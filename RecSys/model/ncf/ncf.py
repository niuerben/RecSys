# -*- coding: utf-8 -*-
"""
Neural Collaborative Filtering (NeuMF) for MovieLens 1M

参考论文: He et al. "Neural Collaborative Filtering" (WWW 2017)
模型结构: GMF + MLP 双分支融合
"""

import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def parse_hidden_layers(value):
    """解析 mlp_hidden 配置字符串，如 '64,32,16' -> [64, 32, 16]"""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


class NeuMF(nn.Module):
    """GMF + MLP 双分支融合模型"""

    def __init__(self, num_users, num_items, embedding_size=64,
                 mlp_hidden=(32, 16, 8), dropout=0.0):
        super().__init__()
        self.embedding_size = embedding_size

        # GMF 分支
        self.gmf_user_emb = nn.Embedding(num_users, embedding_size)
        self.gmf_item_emb = nn.Embedding(num_items, embedding_size)

        # MLP 分支
        self.mlp_user_emb = nn.Embedding(num_users, embedding_size)
        self.mlp_item_emb = nn.Embedding(num_items, embedding_size)

        # MLP 层
        mlp_modules = []
        input_dim = embedding_size * 2
        for out_dim in mlp_hidden:
            mlp_modules.append(nn.Linear(input_dim, out_dim))
            mlp_modules.append(nn.ReLU())
            if dropout > 0:
                mlp_modules.append(nn.Dropout(dropout))
            input_dim = out_dim
        self.mlp = nn.Sequential(*mlp_modules)

        # 融合层
        fusion_dim = embedding_size + (mlp_hidden[-1] if mlp_hidden else embedding_size)
        self.predict = nn.Sequential(
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, user, item):
        gmf_out = self.gmf_user_emb(user) * self.gmf_item_emb(item)
        mlp_out = self.mlp(torch.cat([self.mlp_user_emb(user), self.mlp_item_emb(item)], dim=-1))
        return self.predict(torch.cat([gmf_out, mlp_out], dim=-1)).squeeze()


class NCF(object):
    """Neural Collaborative Filtering 推荐器"""

    def __init__(
        self,
        recommendation_topn=100,
        rating_threshold=3,
        train_ratio=0.8,
        valid_ratio=0.1,
        embedding_size=64,
        mlp_hidden=(32, 16, 8),
        dropout=0.0,
        epochs=500,
        batch_size=4096,
        learning_rate=1e-4,
        num_neg=4,
        seed=2020,
        valid_interval=1,
        early_stop_patience=10,
        min_delta=1e-6,
        save_epoch_recommendations=False,
        epoch_recommendation_dir="./outputs/ncf_epoch_recommendations",
    ):
        self.recommendation_topn = recommendation_topn
        self.rating_threshold = rating_threshold
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.embedding_size = embedding_size
        self.mlp_hidden = tuple(mlp_hidden)
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_neg = num_neg
        self.seed = seed
        self.valid_interval = valid_interval
        self.early_stop_patience = early_stop_patience
        self.min_delta = min_delta
        self.save_epoch_recommendations = save_epoch_recommendations
        self.epoch_recommendation_dir = epoch_recommendation_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.user_ids = []
        self.item_ids = []
        self.user2idx = {}
        self.item2idx = {}
        self.idx2item = {}
        self.train_items_by_user = defaultdict(set)
        self.valid_items_by_user = defaultdict(set)
        self.test_items_by_user = defaultdict(set)
        self.all_items = None
        self.train_pairs = None
        self.valid_pairs = None
        self.test_pairs = None
        self.model = None
        self.best_epoch = 0
        self.best_valid_mrr = -1.0

    def generate_dataset(self, ratingsfile, usersfile=None):
        """加载评分数据并按 8:1:1 划分训练/验证/测试集"""
        print("使用设备:%s" % self.device)
        print("加载 NCF 数据...")

        if usersfile is None:
            candidate = os.path.join(os.path.dirname(ratingsfile), "users.dat")
            if os.path.exists(candidate):
                usersfile = candidate

        interactions = []
        with open(ratingsfile, "r", encoding="latin-1") as f:
            for line in f:
                user, item, rating, _ = line.rstrip("\n").split("::")
                if int(rating) < self.rating_threshold:
                    continue
                interactions.append((int(user), int(item)))

        self.rng.shuffle(interactions)

        if usersfile and os.path.exists(usersfile):
            self.user_ids = self._load_user_ids(usersfile)
        else:
            self.user_ids = sorted({user for user, _ in interactions})
        self.item_ids = sorted({item for _, item in interactions})
        self.user2idx = {user: idx for idx, user in enumerate(self.user_ids)}
        self.item2idx = {item: idx for idx, item in enumerate(self.item_ids)}
        self.idx2item = {idx: item for item, idx in self.item2idx.items()}
        self.all_items = np.arange(len(self.item_ids), dtype=np.int64)

        interactions_by_user = defaultdict(list)
        for user, item in interactions:
            interactions_by_user[user].append(item)

        train_pairs = []
        valid_pairs = []
        test_pairs = []
        ratios = [self.train_ratio, self.valid_ratio, 1 - self.train_ratio - self.valid_ratio]

        for user in self.user_ids:
            items = interactions_by_user.get(user, [])
            if not items:
                continue
            train_items, valid_items, test_items = self._split_user_items(items, ratios)
            user_idx = self.user2idx[user]

            for item in train_items:
                item_idx = self.item2idx[item]
                self.train_items_by_user[user_idx].add(item_idx)
                train_pairs.append((user_idx, item_idx))
            for item in valid_items:
                item_idx = self.item2idx[item]
                self.valid_items_by_user[user_idx].add(item_idx)
                valid_pairs.append((user_idx, item_idx))
            for item in test_items:
                item_idx = self.item2idx[item]
                self.test_items_by_user[user_idx].add(item_idx)
                test_pairs.append((user_idx, item_idx))

        self.train_pairs = np.asarray(train_pairs, dtype=np.int64)
        self.valid_pairs = np.asarray(valid_pairs, dtype=np.int64)
        self.test_pairs = np.asarray(test_pairs, dtype=np.int64)

        print(
            "用户数:%d，电影数:%d，交互数:%d，训练:%d，Top%d 推荐列: %d"
            % (
                len(self.user_ids),
                len(self.item_ids),
                len(interactions),
                len(self.train_pairs),
                self.recommendation_topn,
                len(self.user_ids) * self.recommendation_topn,
            )
        )

    def _load_user_ids(self, usersfile):
        user_ids = []
        with open(usersfile, "r", encoding="latin-1") as f:
            for line in f:
                user_id = line.rstrip("\n").split("::", 1)[0]
                user_ids.append(int(user_id))
        return sorted(user_ids)

    def _split_user_items(self, items, ratios):
        tot = len(items)
        norm_ratios = [ratio / sum(ratios) for ratio in ratios]
        cnt = [int(norm_ratios[i] * tot) for i in range(len(norm_ratios))]
        cnt[0] = tot - sum(cnt[1:])
        for i in range(1, len(norm_ratios)):
            if cnt[0] <= 1:
                break
            if 0 < norm_ratios[-i] * tot < 1:
                cnt[-i] += 1
                cnt[0] -= 1
        train_end = cnt[0]
        valid_end = cnt[0] + cnt[1]
        return items[:train_end], items[train_end:valid_end], items[valid_end:]

    def calc_movie_sim(self):
        """训练模型（兼容接口命名）"""
        self.train()

    def train(self):
        """训练 NCF 模型，带早停机制"""
        print("加载模型 NCF...")
        self.model = NeuMF(
            num_users=len(self.user_ids),
            num_items=len(self.item_ids),
            embedding_size=self.embedding_size,
            mlp_hidden=self.mlp_hidden,
            dropout=self.dropout,
        ).to(self.device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_state_dict = None
        stale_validations = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_count = 0

            # 随机打乱训练数据
            order = self.rng.permutation(len(self.train_pairs))
            for start in range(0, len(order), self.batch_size):
                batch_idx = order[start:start + self.batch_size]
                users = self.train_pairs[batch_idx, 0]
                pos_items = self.train_pairs[batch_idx, 1]

                # 负采样
                neg_items = self._sample_negative_items(users, self.num_neg)

                # 构建 batch: 正样本 + 负样本
                batch_users = np.concatenate([users] * (1 + self.num_neg))
                batch_items = np.concatenate([pos_items] + [neg_items[:, i] for i in range(self.num_neg)])
                batch_labels = np.concatenate([np.ones(len(users)), np.zeros(len(users) * self.num_neg)])

                users_t = torch.LongTensor(batch_users).to(self.device)
                items_t = torch.LongTensor(batch_items).to(self.device)
                labels_t = torch.FloatTensor(batch_labels).to(self.device)

                optimizer.zero_grad()
                preds = self.model(users_t, items_t)
                loss = criterion(preds, labels_t)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_users)
                total_count += len(batch_users)

            avg_loss = total_loss / total_count if total_count > 0 else 0.0
            message = "Epoch %d/%d - loss: %.4f" % (epoch, self.epochs, avg_loss)

            # 验证
            if self.valid_interval > 0 and epoch % self.valid_interval == 0:
                valid_metrics = self.evaluate_split(
                    self.valid_items_by_user,
                    self.train_items_by_user,
                    label="Valid",
                    verbose=False,
                )
                valid_mrr = valid_metrics["mrr"]
                message += " - valid_mrr@10: %.4f" % valid_mrr
                if valid_mrr > self.best_valid_mrr + self.min_delta:
                    self.best_valid_mrr = valid_mrr
                    self.best_epoch = epoch
                    stale_validations = 0
                    best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    message += " *best*"
                else:
                    stale_validations += 1
                    message += " - stale:%d/%d" % (stale_validations, self.early_stop_patience)

            if self.save_epoch_recommendations:
                filename = "ncf_recommendation_%03d.csv" % epoch
                filepath = os.path.join(self.epoch_recommendation_dir, filename)
                self.generate_recommendation(filepath=filepath, progress=False)
                message += " - saved %s" % filepath

            print(message)

            if self.early_stop_patience > 0 and stale_validations >= self.early_stop_patience:
                print(
                    "早停触发: valid MRR@10 连续 %d 次没有提升，停止于 epoch %d。"
                    % (self.early_stop_patience, epoch)
                )
                break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            print(
                "加载验证集最佳 checkpoint: epoch %d, Valid MRR@10=%.4f"
                % (self.best_epoch, self.best_valid_mrr)
            )

    def _sample_negative_items(self, users, num_neg):
        """为每个用户采样 num_neg 个未交互的负样本"""
        neg_items = np.empty((len(users), num_neg), dtype=np.int64)
        for idx, user_idx in enumerate(users):
            for n in range(num_neg):
                while True:
                    item_idx = int(self.rng.integers(0, len(self.item_ids)))
                    if item_idx not in self.train_items_by_user[int(user_idx)]:
                        neg_items[idx, n] = item_idx
                        break
        return neg_items

    def evaluate(self):
        """在测试集上评估模型"""
        return self.evaluate_split(
            self.test_items_by_user,
            self.train_items_by_user,
            label="Test",
            verbose=True,
        )

    def evaluate_split(self, eval_items_by_user, mask_items_by_user, label="Test", verbose=True):
        """在指定数据集上评估排序指标"""
        N = 10  # TopN 评估固定为 10
        precision_sum = 0.0
        recall_sum = 0.0
        ndcg_sum = 0.0
        mrr_sum = 0.0
        hit_user_count = 0
        eval_user_count = 0

        self.model.eval()
        with torch.no_grad():
            for user_idx in range(len(self.user_ids)):
                if verbose and user_idx % 500 == 0:
                    print("%s topn evaluate for %d users" % (label.lower(), user_idx), file=sys.stderr)

                eval_items = eval_items_by_user.get(user_idx, set())
                if not eval_items:
                    continue

                # 对所有物品打分
                user_tensor = torch.LongTensor([user_idx] * len(self.item_ids)).to(self.device)
                item_tensor = torch.LongTensor(self.all_items).to(self.device)
                scores = self.model(user_tensor, item_tensor).cpu().numpy()

                # 屏蔽训练集物品
                for item_idx in mask_items_by_user.get(user_idx, set()):
                    scores[item_idx] = -np.inf

                top_k = min(N, len(scores))
                top_idx = np.argpartition(scores, -top_k)[-top_k:]
                top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

                dcg = 0.0
                user_hit = 0
                for rank, item_idx in enumerate(top_idx, start=1):
                    if int(item_idx) in eval_items:
                        user_hit += 1
                        dcg += 1 / math.log2(rank + 1)
                        if user_hit == 1:
                            mrr_sum += 1 / rank

                ideal_hits = min(len(eval_items), N)
                idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
                precision_sum += user_hit / N
                recall_sum += user_hit / len(eval_items)
                ndcg_sum += dcg / idcg if idcg else 0
                if user_hit > 0:
                    hit_user_count += 1
                eval_user_count += 1

        precision = precision_sum / eval_user_count if eval_user_count else 0
        recall = recall_sum / eval_user_count if eval_user_count else 0
        ndcg = ndcg_sum / eval_user_count if eval_user_count else 0
        mrr = mrr_sum / eval_user_count if eval_user_count else 0
        hit_rate = hit_user_count / eval_user_count if eval_user_count else 0

        metrics = {
            "recall": recall,
            "mrr": mrr,
            "ndcg": ndcg,
            "hit": hit_rate,
            "precision": precision,
            "users": eval_user_count,
        }

        if verbose:
            print(
                "测试集 %s RECALL@%d : %.4f    MRR@%d : %.4f    NDCG@%d : %.4f    HIT@%d : %.4f    PRECISION@%d : %.4f"
                % (label, N, recall, N, mrr, N, ndcg, N, hit_rate, N, precision)
            )

        return metrics

    def generate_recommendation(self, filepath="./RecSys/outputs/ncf_recommendation.csv", topn=None, progress=True):
        """生成用户推荐表"""
        topn = topn or self.recommendation_topn
        print("generating NCF recommendation result: %s" % filepath)
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["user_id"] + ["rec%d" % idx for idx in range(1, topn + 1)])

                for user_idx, user_id in enumerate(self.user_ids):
                    if progress and user_idx % 500 == 0:
                        print("generate NCF recommendation for %d users" % user_idx, file=sys.stderr)

                    user_tensor = torch.LongTensor([user_idx] * len(self.item_ids)).to(self.device)
                    item_tensor = torch.LongTensor(self.all_items).to(self.device)
                    scores = self.model(user_tensor, item_tensor).cpu().numpy()

                    # 屏蔽训练集物品
                    for item_idx in self.train_items_by_user.get(user_idx, set()):
                        scores[item_idx] = -np.inf

                    top_k = min(topn, len(scores))
                    top_idx = np.argpartition(scores, -top_k)[-top_k:]
                    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
                    items = [str(self.idx2item[int(idx)]) for idx in top_idx]
                    if len(items) < topn:
                        items.extend([""] * (topn - len(items)))
                    writer.writerow([user_id] + items)

    def gernate_recommendation(self):
        """兼容旧接口"""
        self.generate_recommendation()
