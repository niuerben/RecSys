# -*- coding: utf-8 -*-
"""
Lightweight XSimGCL recommender for MovieLens 1M.

This module is intentionally self-contained for the course project. It follows
the same data convention as the RecBole-GNN ML-1M setting used in the report:
filter rating < 3, split interactions by 8:1:1, and evaluate Top-10 ranking.
"""

import argparse
import copy
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class XSimGCLModel(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        edge_index,
        embedding_dim=64,
        n_layers=2,
        eps=0.2,
        layer_cl=1,
    ):
        super().__init__()
        if layer_cl < 1 or layer_cl > n_layers:
            raise ValueError("layer_cl must be in [1, n_layers]")
        self.n_users = n_users
        self.n_items = n_items
        self.n_nodes = n_users + n_items
        self.n_layers = n_layers
        self.eps = eps
        self.layer_cl = layer_cl
        self.embedding = nn.Embedding(self.n_nodes, embedding_dim)
        row, col = edge_index
        deg = torch.bincount(row, minlength=self.n_nodes).float().clamp(min=1)
        edge_weight = torch.rsqrt(deg[row]) * torch.rsqrt(deg[col])
        adj = torch.sparse_coo_tensor(
            edge_index,
            edge_weight,
            (self.n_nodes, self.n_nodes),
            device=edge_index.device,
        ).coalesce()
        self.register_buffer("adj", adj)
        nn.init.xavier_uniform_(self.embedding.weight)

    def _propagate(self, perturbed=False):
        all_emb = self.embedding.weight
        embeddings = []
        all_emb_cl = all_emb
        for layer_idx in range(self.n_layers):
            all_emb = torch.sparse.mm(self.adj, all_emb)
            if perturbed:
                noise = F.normalize(torch.rand_like(all_emb), dim=1)
                all_emb = all_emb + torch.sign(all_emb) * noise * self.eps
            embeddings.append(all_emb)
            if layer_idx == self.layer_cl - 1:
                all_emb_cl = all_emb
        output = torch.stack(embeddings, dim=0).mean(dim=0)
        if perturbed:
            user_emb, item_emb = output[: self.n_users], output[self.n_users :]
            user_cl, item_cl = all_emb_cl[: self.n_users], all_emb_cl[self.n_users :]
            return user_emb, item_emb, user_cl, item_cl
        return output[: self.n_users], output[self.n_users :]

    def forward(self, perturbed=False):
        return self._propagate(perturbed=perturbed)


class XSimGCL(object):
    def __init__(
        self,
        topn=10,
        recommendation_topn=100,
        rating_threshold=3,
        train_ratio=0.8,
        valid_ratio=0.1,
        embedding_dim=64,
        n_layers=2,
        epochs=500,
        batch_size=4096,
        learning_rate=0.002,
        reg_weight=1e-4,
        ssl_weight=0.1,
        temperature=0.2,
        eps=0.2,
        layer_cl=1,
        seed=2020,
        valid_interval=1,
        early_stop_patience=10,
        min_delta=1e-6,
        save_epoch_recommendations=False,
        epoch_recommendation_dir="./outputs/xsimgcl_epoch_recommendations",
    ):
        self.topn = topn
        self.recommendation_topn = recommendation_topn
        self.rating_threshold = rating_threshold
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.reg_weight = reg_weight
        self.ssl_weight = ssl_weight
        self.temperature = temperature
        self.eps = eps
        self.layer_cl = layer_cl
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
        print("使用设备:%s" % self.device)
        print("加载 XSimGCL 数据...")

        if usersfile is None:
            candidate_usersfile = os.path.join(os.path.dirname(ratingsfile), "users.dat")
            if os.path.exists(candidate_usersfile):
                usersfile = candidate_usersfile

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
        self.edge_index = self._build_edge_index(train_pairs)
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

    def gernate_dataset(self, ratingsfile, usersfile=None):
        self.generate_dataset(ratingsfile, usersfile=usersfile)

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

    def _build_edge_index(self, train_pairs):
        rows = []
        cols = []
        item_offset = len(self.user_ids)
        for user_idx, item_idx in train_pairs:
            item_node = item_offset + item_idx
            rows.extend([user_idx, item_node])
            cols.extend([item_node, user_idx])
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        return edge_index.to(self.device)

    def calc_movie_sim(self):
        self.train()

    def train(self):
        print("加载模型 XSimGCL...")
        self.model = XSimGCLModel(
            n_users=len(self.user_ids),
            n_items=len(self.item_ids),
            edge_index=self.edge_index,
            embedding_dim=self.embedding_dim,
            n_layers=self.n_layers,
            eps=self.eps,
            layer_cl=self.layer_cl,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_state_dict = None
        stale_validations = 0

        for epoch in range(1, self.epochs + 1):
            order = self.rng.permutation(len(self.train_pairs))
            total_loss = 0.0
            total_count = 0
            self.model.train()
            for start in range(0, len(order), self.batch_size):
                batch_idx = order[start : start + self.batch_size]
                users = self.train_pairs[batch_idx, 0]
                pos_items = self.train_pairs[batch_idx, 1]
                neg_items = self._sample_negative_items(users)

                users_t = torch.tensor(users, dtype=torch.long, device=self.device)
                pos_t = torch.tensor(pos_items, dtype=torch.long, device=self.device)
                neg_t = torch.tensor(neg_items, dtype=torch.long, device=self.device)

                user_emb, item_emb, user_cl, item_cl = self.model(perturbed=True)

                u = user_emb[users_t]
                pos = item_emb[pos_t]
                neg = item_emb[neg_t]
                pos_scores = torch.sum(u * pos, dim=1)
                neg_scores = torch.sum(u * neg, dim=1)
                bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
                ego_emb = self.model.embedding.weight
                ego_users = ego_emb[users_t]
                ego_pos = ego_emb[len(self.user_ids) + pos_t]
                ego_neg = ego_emb[len(self.user_ids) + neg_t]
                reg_loss = (
                    ego_users.norm(2).pow(2)
                    + ego_pos.norm(2).pow(2)
                    + ego_neg.norm(2).pow(2)
                ) / len(users_t)
                unique_users = torch.unique(users_t)
                unique_pos = torch.unique(pos_t)
                ssl_loss = self._ssl_loss(user_emb[unique_users], user_cl[unique_users])
                ssl_loss = ssl_loss + self._ssl_loss(item_emb[unique_pos], item_cl[unique_pos])
                loss = bpr_loss + self.reg_weight * reg_loss + self.ssl_weight * ssl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(users_t)
                total_count += len(users_t)
            message = "Epoch %d/%d - loss: %.4f" % (epoch, self.epochs, total_loss / total_count)
            if self.valid_interval > 0 and epoch % self.valid_interval == 0:
                valid_metrics = self.evaluate_split(
                    self.valid_items_by_user,
                    self.train_items_by_user,
                    label="Valid",
                    verbose=False,
                    write_metrics=False,
                )
                valid_mrr = valid_metrics["mrr"]
                message += " - valid_mrr@%d: %.4f" % (self.topn, valid_mrr)
                if valid_mrr > self.best_valid_mrr + self.min_delta:
                    self.best_valid_mrr = valid_mrr
                    self.best_epoch = epoch
                    stale_validations = 0
                    best_state_dict = copy.deepcopy(self.model.state_dict())
                    message += " *best*"
                else:
                    stale_validations += 1
                    message += " - stale:%d/%d" % (stale_validations, self.early_stop_patience)
            if self.save_epoch_recommendations:
                filename = "xsimgcl_recommendation_%03d.csv" % epoch
                filepath = os.path.join(self.epoch_recommendation_dir, filename)
                self.generate_recommendation(filepath=filepath, mask_valid=True, progress=False)
                message += " - saved %s" % filepath
            print(message)
            if self.early_stop_patience > 0 and stale_validations >= self.early_stop_patience:
                print(
                    "早停触发: valid MRR@%d 连续 %d 次没有提升，停止于 epoch %d。"
                    % (self.topn, self.early_stop_patience, epoch)
                )
                break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            print(
                "加载验证集最佳 checkpoint: epoch %d, Valid MRR@%d=%.4f"
                % (self.best_epoch, self.topn, self.best_valid_mrr)
            )

    def _sample_negative_items(self, users):
        neg_items = np.empty(len(users), dtype=np.int64)
        for idx, user_idx in enumerate(users):
            while True:
                item_idx = int(self.rng.integers(0, len(self.item_ids)))
                if item_idx not in self.train_items_by_user[int(user_idx)]:
                    neg_items[idx] = item_idx
                    break
        return neg_items

    def _ssl_loss(self, emb1, emb2):
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.temperature)
        all_score = torch.exp(torch.matmul(emb1, emb2.t()) / self.temperature).sum(dim=1)
        return -torch.log(pos_score / all_score.clamp(min=1e-12)).mean()

    def evaluate(self):
        mask_items_by_user = defaultdict(set)
        for user_idx, items in self.train_items_by_user.items():
            mask_items_by_user[user_idx].update(items)
        for user_idx, items in self.valid_items_by_user.items():
            mask_items_by_user[user_idx].update(items)
        return self.evaluate_split(
            self.test_items_by_user,
            mask_items_by_user,
            label="Test",
            verbose=True,
            write_metrics=False,
        )

    def evaluate_split(self, eval_items_by_user, mask_items_by_user, label="Test", verbose=True, write_metrics=False):
        N = self.topn
        hit = 0
        precision_sum = 0.0
        recall_sum = 0.0
        ndcg_sum = 0.0
        map_sum = 0.0
        mrr_sum = 0.0
        user_hit_count = 0
        eval_user_count = 0
        self.model.eval()
        with torch.no_grad():
            user_emb, item_emb = self.model(perturbed=False)
            user_emb = user_emb.detach()
            item_emb = item_emb.detach()

        for user_idx in range(len(self.user_ids)):
            if verbose and user_idx % 500 == 0:
                print("%s topn evaluate for %d users" % (label.lower(), user_idx), file=sys.stderr)
            eval_items = eval_items_by_user.get(user_idx, set())
            if not eval_items:
                continue
            scores = torch.matmul(item_emb, user_emb[user_idx]).detach().cpu().numpy()
            for item_idx in mask_items_by_user.get(user_idx, set()):
                scores[item_idx] = -np.inf
            top_k = min(N, len(scores))
            top_idx = np.argpartition(scores, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

            dcg = 0.0
            ap = 0.0
            user_hit = 0
            for rank, item_idx in enumerate(top_idx, start=1):
                if int(item_idx) in eval_items:
                    hit += 1
                    user_hit += 1
                    dcg += 1 / math.log2(rank + 1)
                    ap += user_hit / rank
                    if user_hit == 1:
                        mrr_sum += 1 / rank

            ideal_hits = min(len(eval_items), N)
            idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            precision_sum += user_hit / N
            recall_sum += user_hit / len(eval_items)
            ndcg_sum += dcg / idcg if idcg else 0
            map_sum += ap / ideal_hits if ideal_hits else 0
            if user_hit > 0:
                user_hit_count += 1
            eval_user_count += 1

        precision = precision_sum / eval_user_count if eval_user_count else 0
        recall = recall_sum / eval_user_count if eval_user_count else 0
        ndcg = ndcg_sum / eval_user_count if eval_user_count else 0
        mean_ap = map_sum / eval_user_count if eval_user_count else 0
        mrr = mrr_sum / eval_user_count if eval_user_count else 0
        hit_rate = user_hit_count / eval_user_count if eval_user_count else 0

        metrics = {
            "recall": recall,
            "mrr": mrr,
            "ndcg": ndcg,
            "hit": hit_rate,
            "precision": precision,
            "map": mean_ap,
            "users": eval_user_count,
            "hits": hit,
        }

        if verbose:
            print(
                "测试集 %s RECALL@%d : %.4f    MRR@%d : %.4f    NDCG@%d : %.4f    HIT@%d : %.4f    PRECISION@%d : %.4f"
                % (label, N, recall, N, mrr, N, ndcg, N, hit_rate, N, precision)
            )

        return metrics

    def generate_recommendation(self, filepath="./outputs/xsimgcl_recommendation.csv", topn=None, mask_valid=False, progress=True):
        topn = topn or self.recommendation_topn
        print("generating XSimGCL recommendation result: %s" % filepath)
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        user_emb, item_emb = self.model(perturbed=False)
        user_emb = user_emb.detach()
        item_emb = item_emb.detach()
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id"] + ["rec%d" % idx for idx in range(1, topn + 1)])
            for user_idx, user_id in enumerate(self.user_ids):
                if progress and user_idx % 500 == 0:
                    print("generate XSimGCL recommendation for %d users" % user_idx, file=sys.stderr)
                scores = torch.matmul(item_emb, user_emb[user_idx]).detach().cpu().numpy()
                for item_idx in self.train_items_by_user.get(user_idx, set()):
                    scores[item_idx] = -np.inf
                if mask_valid:
                    for item_idx in self.valid_items_by_user.get(user_idx, set()):
                        scores[item_idx] = -np.inf
                top_k = min(topn, len(scores))
                top_idx = np.argpartition(scores, -top_k)[-top_k:]
                top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
                items = [str(self.idx2item[int(idx)]) for idx in top_idx]
                if len(items) < topn:
                    items.extend([""] * (topn - len(items)))
                writer.writerow([user_id] + items)

    def gernate_recommendation(self):
        self.generate_recommendation()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Reproduce XSimGCL on MovieLens-1M.")
    parser.add_argument("--ratings-file", default="./data/ml-1m/ml-1m/ratings.dat")
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--recommendation-topn", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--reg-weight", type=float, default=1e-4)
    parser.add_argument("--ssl-weight", type=float, default=0.1, help="RecBole-GNN XSimGCL lambda.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--layer-cl", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--users-file", default="./data/ml-1m/ml-1m/users.dat")
    parser.add_argument("--valid-interval", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--save-epoch-recommendations", action="store_true")
    parser.add_argument("--epoch-recommendation-dir", default="./outputs/xsimgcl_epoch_recommendations")
    parser.add_argument("--skip-recommendation", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    xsimgcl = XSimGCL(
        topn=args.topn,
        recommendation_topn=args.recommendation_topn,
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        reg_weight=args.reg_weight,
        ssl_weight=args.ssl_weight,
        temperature=args.temperature,
        eps=args.eps,
        layer_cl=args.layer_cl,
        seed=args.seed,
        valid_interval=args.valid_interval,
        early_stop_patience=args.early_stop_patience,
        min_delta=args.min_delta,
        save_epoch_recommendations=args.save_epoch_recommendations,
        epoch_recommendation_dir=args.epoch_recommendation_dir,
    )
    xsimgcl.generate_dataset(args.ratings_file, usersfile=args.users_file)
    xsimgcl.calc_movie_sim()
    xsimgcl.evaluate()
    if not args.skip_recommendation:
        xsimgcl.generate_recommendation(topn=args.recommendation_topn, mask_valid=True)


if __name__ == "__main__":
    main()
