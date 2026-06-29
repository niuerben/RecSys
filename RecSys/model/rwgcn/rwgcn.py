# -*- coding: utf-8 -*-
"""
RWGCN: Rating-Weighted Graph Convolution Network
=================================================

在 LightGCN 基础上引入**评分加权卷积（Rating-Weighted Convolution）**：
  - 用评分映射的实数权重替代传统 0/1 邻接矩阵
  - 图卷积形式不变（torch.sparse.mm 天然兼容实数权重）
  - 提供两种损失方案（通过 ``loss_scheme`` 切换）

损失方案
--------
  **方案 A**：标准 BPR Loss + 加权图传播
    图传播使用评分加权矩阵，BPR 仍将 rating≥threshold 的交互视为正样本。

  **方案 B**：加权 BPR Loss（Weighted BPR）
    L = -Σ (r_{ui} - r_{uj}) · ln σ(ŷ_{ui} - ŷ_{uj})
    其中 r_{ui} 为真实评分，r_{uj}=0（未交互）。

启动方式
--------
    python main.py --model rwgcn --config RecSys/config/rwgcn.yaml       # 方案 A
    python main.py --model rwgcn --config RecSys/config/rwgcn_b.yaml     # 方案 B

参考
----
  评分加权卷积.md — 完整设计文档
"""

import argparse
import copy
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn


# ==============================================================================
#  1.  Rating → Edge Weight 映射
# ==============================================================================

def map_rating_to_weight(ratings, w_min=0.2):
    """
    线性缩放：评分 [1, 5] → 权重 [w_min, 1.0]

    Parameters
    ----------
    ratings : np.ndarray  原始评分 (1~5)
    w_min   : float       最低权重，防止低分交互彻底断连

    Returns
    -------
    weights : np.ndarray  shape 同 ratings, dtype float32
    """
    r_min, r_max = 1.0, 5.0
    weights = w_min + (1.0 - w_min) * (ratings - r_min) / (r_max - r_min)
    return weights.astype(np.float32)


# ==============================================================================
#  2.  构建加权对称归一化邻接矩阵  D^{-½} A D^{-½}
# ==============================================================================

def build_weighted_adj(user_indices, item_indices, ratings,
                        num_users, num_items, w_min=0.2):
    """
    构建评分加权的对称归一化邻接矩阵。

    Parameters
    ----------
    user_indices : np.ndarray [E]  0-based 用户索引
    item_indices : np.ndarray [E]  0-based 物品索引
    ratings      : np.ndarray [E]  原始评分 (1~5)
    num_users    : int             用户总数
    num_items    : int             物品总数
    w_min        : float           最小边权重

    Returns
    -------
    sparse_adj : torch.sparse_coo_tensor  (N+M)×(N+M)
    """
    # 1. 评分 → 权重
    weights = map_rating_to_weight(ratings, w_min)

    # 2. 加权交互矩阵 R (CSR)
    R = sp.csr_matrix((weights, (user_indices, item_indices)),
                       shape=(num_users, num_items))

    # 3. 对称二分图邻接矩阵 A = [[0, R], [R^T, 0]]
    zero_u = sp.csr_matrix((num_users, num_users))
    zero_i = sp.csr_matrix((num_items, num_items))
    adj = sp.bmat([[zero_u, R], [R.T, zero_i]], format='csr')

    # 4. 加权度矩阵 D (按行求和)
    rowsum = np.array(adj.sum(axis=1)).flatten()
    rowsum[rowsum == 0] = 1e-12

    # 5. D^{-½}
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_mat_inv = sp.diags(d_inv_sqrt)

    # 6. 对称归一化: D^{-½} A D^{-½}
    norm_adj = d_mat_inv.dot(adj).dot(d_mat_inv)

    # 7. COO → PyTorch Sparse Tensor
    norm_adj = norm_adj.tocoo()
    indices = torch.from_numpy(
        np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
    values = torch.from_numpy(norm_adj.data.astype(np.float32))
    shape = torch.Size(norm_adj.shape)

    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


# ==============================================================================
#  3.  RWGCN nn.Module（加权 LightGCN 传播）
# ==============================================================================

class RWGCNModel(nn.Module):
    """
    评分加权 LightGCN 传播模块。

    与标准 LightGCN 的结构完全相同，区别仅在于 self.adj 中的非零元素
    是实数评分权重而非统一的 1。
    """

    def __init__(self, n_users, n_items, weighted_adj, embedding_dim=64, n_layers=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_nodes = n_users + n_items
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim

        # 共享 Embedding 表
        self.embedding = nn.Embedding(self.n_nodes, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

        # 预计算的加权归一化邻接矩阵（冻结为 buffer）
        self.register_buffer("adj", weighted_adj)

    def computer(self):
        """
        多层加权图传播 + 均值池化。

        Returns
        -------
        users : Tensor [n_users, dim]
        items : Tensor [n_items, dim]
        """
        all_emb = self.embedding.weight                       # E^(0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.adj, all_emb)      # 加权传播
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)                       # [N, K+1, d]
        light_out = torch.mean(embs, dim=1)                   # 均值池化
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items

    def forward(self, users, items):
        """预测 (user, item) 得分。"""
        all_users, all_items = self.computer()
        return torch.sum(all_users[users] * all_items[items], dim=1)

    def getUsersRating(self, users):
        """返回 [len(users), n_items] 得分矩阵。"""
        all_users, all_items = self.computer()
        return torch.matmul(all_users[users], all_items.t())


# ==============================================================================
#  4.  RWGCN 包装器（RecSys-master 统一接口）
# ==============================================================================

class RWGCN(object):
    """
    评分加权图卷积推荐模型。

    Parameters
    ----------
    topn : int                    Recall@N / MRR@N 的 N
    recommendation_topn : int     输出推荐列表长度
    rating_threshold : int        正样本评分阈值（用于 BPR 采样与评估）
    embedding_dim : int           Embedding 维度
    n_layers : int                图卷积层数
    w_min : float                 评分→权重的最小权重
    loss_scheme : str             "A" = 标准 BPR, "B" = 加权 BPR
    epochs : int
    batch_size : int
    learning_rate : float
    reg_weight : float            L2 正则化系数
    seed : int
    valid_interval : int
    early_stop_patience : int
    min_delta : float
    save_epoch_recommendations : bool
    epoch_recommendation_dir : str
    """

    def __init__(
        self,
        topn=10,
        recommendation_topn=100,
        rating_threshold=3,
        train_ratio=0.8,
        valid_ratio=0.1,
        embedding_dim=64,
        n_layers=3,
        w_min=0.2,
        loss_scheme="A",
        epochs=500,
        batch_size=2048,
        learning_rate=0.001,
        reg_weight=1e-4,
        seed=2020,
        valid_interval=1,
        early_stop_patience=10,
        min_delta=1e-6,
        save_epoch_recommendations=False,
        epoch_recommendation_dir="./outputs/rwgcn_epoch_recommendations",
    ):
        # ---- 超参 ----------------------------------------------------------------
        self.topn = topn
        self.recommendation_topn = recommendation_topn
        self.rating_threshold = rating_threshold
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.w_min = w_min
        self.loss_scheme = loss_scheme.upper()
        if self.loss_scheme not in ("A", "B"):
            raise ValueError(f"loss_scheme 必须是 'A' 或 'B'，收到: {loss_scheme}")
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.reg_weight = reg_weight
        self.seed = seed
        self.valid_interval = valid_interval
        self.early_stop_patience = early_stop_patience
        self.min_delta = min_delta
        self.save_epoch_recommendations = save_epoch_recommendations
        self.epoch_recommendation_dir = epoch_recommendation_dir

        # ---- 设备 & 随机种子 ----------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # ---- 数据容器 ------------------------------------------------------------
        self.user_ids = []
        self.item_ids = []
        self.user2idx = {}
        self.item2idx = {}
        self.idx2item = {}
        self.train_items_by_user = defaultdict(set)
        self.valid_items_by_user = defaultdict(set)
        self.test_items_by_user = defaultdict(set)
        self.all_items = None
        self.train_pairs = None         # (user_idx, item_idx)
        self.train_ratings = None       # 并行数组，原始评分
        self.valid_pairs = None
        self.test_pairs = None
        self.weighted_adj = None

        # ---- 模型状态 ------------------------------------------------------------
        self.model: RWGCNModel = None
        self.best_epoch = 0
        self.best_valid_mrr = -1.0

    # ==================================================================
    #  数据加载
    # ==================================================================

    def generate_dataset(self, ratingsfile, usersfile=None):
        """
        加载 MovieLens-1M 评分数据。

        Parameters
        ----------
        ratingsfile : str   ratings.dat 路径（user::item::rating::ts）
        usersfile   : str   可选 users.dat 路径
        """
        print("使用设备: %s" % self.device)
        print("加载 RWGCN 数据（评分加权卷积）...")
        print("  损失方案: %s  |  最小权重 w_min: %.2f" % (self.loss_scheme, self.w_min))

        # ---- 读取交互（保留所有评分，不做阈值过滤）------------------------------
        interactions = []          # (user_id, item_id, rating)
        with open(ratingsfile, "r", encoding="latin-1") as f:
            for line in f:
                user, item, rating, _ = line.rstrip("\n").split("::")
                interactions.append((int(user), int(item), float(rating)))

        self.rng.shuffle(interactions)

        # ---- 用户 / 物品列表 ----------------------------------------------------
        if usersfile and os.path.exists(usersfile):
            self.user_ids = self._load_user_ids(usersfile)
        else:
            self.user_ids = sorted({u for u, _, _ in interactions})

        self.item_ids = sorted({i for _, i, _ in interactions})
        self.user2idx = {u: idx for idx, u in enumerate(self.user_ids)}
        self.item2idx = {i: idx for idx, i in enumerate(self.item_ids)}
        self.idx2item = {idx: i for i, idx in self.item2idx.items()}
        self.all_items = np.arange(len(self.item_ids), dtype=np.int64)

        # ---- 按用户分组 ---------------------------------------------------------
        interactions_by_user = defaultdict(list)
        for user, item, rating in interactions:
            interactions_by_user[user].append((item, rating))

        # ---- 8:1:1 划分 ---------------------------------------------------------
        train_triplets = []
        valid_pairs_list = []
        test_pairs_list = []

        for user_id in self.user_ids:
            items_with_ratings = interactions_by_user.get(user_id, [])
            if not items_with_ratings:
                continue
            self.rng.shuffle(items_with_ratings)
            items_arr = items_with_ratings
            tot = len(items_arr)
            n_train = max(1, int(round(self.train_ratio * tot)))
            n_valid = max(0, int(round(self.valid_ratio * tot)))
            n_test = tot - n_train - n_valid
            if n_test <= 0:
                n_train = max(1, tot - 1)
                n_valid = max(0, tot - n_train - 1)
                n_test = tot - n_train - n_valid

            train_items = items_arr[:n_train]
            valid_items = items_arr[n_train:n_train + n_valid]
            test_items  = items_arr[n_train + n_valid:]

            user_idx = self.user2idx[user_id]

            for item_id, rating in train_items:
                item_idx = self.item2idx[item_id]
                self.train_items_by_user[user_idx].add(item_idx)
                train_triplets.append((user_idx, item_idx, rating))
            for item_id, rating in valid_items:
                item_idx = self.item2idx[item_id]
                self.valid_items_by_user[user_idx].add(item_idx)
                valid_pairs_list.append((user_idx, item_idx))
            for item_id, rating in test_items:
                item_idx = self.item2idx[item_id]
                self.test_items_by_user[user_idx].add(item_idx)
                test_pairs_list.append((user_idx, item_idx))

        self.train_pairs = np.array([(u, i) for u, i, _ in train_triplets],
                                     dtype=np.int64)
        self.train_ratings = np.array([r for _, _, r in train_triplets],
                                       dtype=np.float32)
        self.valid_pairs = np.array(valid_pairs_list, dtype=np.int64)
        self.test_pairs = np.array(test_pairs_list, dtype=np.int64)

        # ---- 构建加权邻接矩阵（使用 ALL 交互）---------------------------------
        all_user_idx = np.array([self.user2idx[u] for u, _, _ in interactions],
                                 dtype=np.int64)
        all_item_idx = np.array([self.item2idx[i] for _, i, _ in interactions],
                                 dtype=np.int64)
        all_ratings = np.array([r for _, _, r in interactions], dtype=np.float32)

        self.weighted_adj = build_weighted_adj(
            all_user_idx, all_item_idx, all_ratings,
            len(self.user_ids), len(self.item_ids),
            w_min=self.w_min,
        )

        # ---- 日志 ---------------------------------------------------------------
        n_pos_train = int((self.train_ratings >= self.rating_threshold).sum())
        print(
            "用户数: %d  电影数: %d  交互数: %d\n"
            "训练正样本 (rating>=%d): %d / %d  |  Top%d 推荐列: %d"
            % (
                len(self.user_ids), len(self.item_ids), len(interactions),
                self.rating_threshold, n_pos_train, len(self.train_pairs),
                self.recommendation_topn,
                len(self.user_ids) * self.recommendation_topn,
            )
        )

    def gernate_dataset(self, ratingsfile, usersfile=None):
        """别名（兼容旧拼写）。"""
        self.generate_dataset(ratingsfile, usersfile=usersfile)

    # ==================================================================
    #  辅助方法
    # ==================================================================

    @staticmethod
    def _load_user_ids(usersfile):
        user_ids = []
        with open(usersfile, "r", encoding="latin-1") as f:
            for line in f:
                user_id = line.rstrip("\n").split("::", 1)[0]
                user_ids.append(int(user_id))
        return sorted(user_ids)

    def _sample_negative_items(self, users):
        """为每个用户采样一个未交互的负样本物品。"""
        neg_items = np.empty(len(users), dtype=np.int64)
        for idx, user_idx in enumerate(users):
            while True:
                item_idx = int(self.rng.integers(0, len(self.item_ids)))
                if item_idx not in self.train_items_by_user[int(user_idx)]:
                    neg_items[idx] = item_idx
                    break
        return neg_items

    # ==================================================================
    #  训练
    # ==================================================================

    def calc_movie_sim(self):
        """别名（兼容 ItemCF / UserCF 命名习惯）。"""
        self.train()

    def train(self):
        """
        训练 RWGCN。

        方案 A：标准 BPR Loss + 加权图
        方案 B：加权 BPR Loss = -(r_pos - r_neg) · ln σ(ŷ_pos - ŷ_neg)
        """
        print("加载模型 RWGCN（评分加权卷积）...")
        print("  损失方案: %s" % self.loss_scheme)

        # ---- 确定正样本 ---------------------------------------------------------
        if self.loss_scheme == "A":
            mask = self.train_ratings >= self.rating_threshold
            pos_users = self.train_pairs[mask, 0]
            pos_items = self.train_pairs[mask, 1]
            pos_ratings = self.train_ratings[mask]
        else:  # B
            pos_users = self.train_pairs[:, 0]
            pos_items = self.train_pairs[:, 1]
            pos_ratings = self.train_ratings

        num_pos = len(pos_users)
        print("  训练正样本数: %d" % num_pos)

        # ---- 初始化模型 ---------------------------------------------------------
        self.model = RWGCNModel(
            n_users=len(self.user_ids),
            n_items=len(self.item_ids),
            weighted_adj=self.weighted_adj.to(self.device),
            embedding_dim=self.embedding_dim,
            n_layers=self.n_layers,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_state_dict = None
        stale_validations = 0

        # ---- 训练循环 -----------------------------------------------------------
        for epoch in range(1, self.epochs + 1):
            order = self.rng.permutation(num_pos)
            total_loss = 0.0
            total_count = 0
            self.model.train()

            for start in range(0, num_pos, self.batch_size):
                batch_idx = order[start:start + self.batch_size]
                users = pos_users[batch_idx]
                pos_items_batch = pos_items[batch_idx]
                neg_items = self._sample_negative_items(users)

                users_t = torch.tensor(users, dtype=torch.long, device=self.device)
                pos_t = torch.tensor(pos_items_batch, dtype=torch.long, device=self.device)
                neg_t = torch.tensor(neg_items, dtype=torch.long, device=self.device)

                # ---- 图传播 ------------------------------------------------------
                all_users_emb, all_items_emb = self.model.computer()
                users_emb = all_users_emb[users_t]
                pos_emb = all_items_emb[pos_t]
                neg_emb = all_items_emb[neg_t]

                pos_scores = torch.sum(users_emb * pos_emb, dim=1)
                neg_scores = torch.sum(users_emb * neg_emb, dim=1)

                # ---- 损失计算 ----------------------------------------------------
                if self.loss_scheme == "A":
                    # 标准 BPR: L = -ln σ(pos − neg) = softplus(neg − pos)
                    bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))
                else:  # B
                    # 加权 BPR: L = -(r_pos - r_neg) · ln σ(pos − neg)
                    #               = (r_pos - 0) · softplus(neg − pos)
                    pos_r = torch.tensor(
                        pos_ratings[batch_idx], dtype=torch.float32, device=self.device)
                    weight = pos_r  # r_neg = 0  for unobserved items
                    sample_loss = weight * F.softplus(neg_scores - pos_scores)
                    bpr_loss = torch.mean(sample_loss)

                # ---- L2 正则化（仅对初始 Embedding）-------------------------------
                ego_emb = self.model.embedding.weight
                ego_users = ego_emb[users_t]
                ego_pos = ego_emb[len(self.user_ids) + pos_t]
                ego_neg = ego_emb[len(self.user_ids) + neg_t]
                reg_loss = (ego_users.norm(2).pow(2) +
                            ego_pos.norm(2).pow(2) +
                            ego_neg.norm(2).pow(2)) / float(len(users_t))

                loss = bpr_loss + self.reg_weight * reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(users_t)
                total_count += len(users_t)

            # ---- 验证 -----------------------------------------------------------
            message = "Epoch %d/%d - loss: %.4f" % (
                epoch, self.epochs, total_loss / max(total_count, 1))

            if self.valid_interval > 0 and epoch % self.valid_interval == 0:
                valid_metrics = self._evaluate_split(
                    self.valid_items_by_user,
                    self.train_items_by_user,
                    label="Valid",
                    verbose=False,
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
                    message += " - stale:%d/%d" % (
                        stale_validations, self.early_stop_patience)

            # ---- 保存每轮推荐（可选）-----------------------------------------------
            if self.save_epoch_recommendations:
                filename = "rwgcn_recommendation_%03d.csv" % epoch
                filepath = os.path.join(self.epoch_recommendation_dir, filename)
                self.generate_recommendation(filepath=filepath, mask_valid=True,
                                              progress=False)
                message += " - saved %s" % filepath

            print(message)

            # ---- 早停 -----------------------------------------------------------
            if self.early_stop_patience > 0 and stale_validations >= self.early_stop_patience:
                print("早停触发: valid MRR@%d 连续 %d 次未提升，停止于 epoch %d。"
                      % (self.topn, self.early_stop_patience, epoch))
                break

        # ---- 恢复最佳 checkpoint ------------------------------------------------
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            print("加载最佳 checkpoint: epoch %d, Valid MRR@%d=%.4f"
                  % (self.best_epoch, self.topn, self.best_valid_mrr))

    # ==================================================================
    #  评估
    # ==================================================================

    def evaluate(self):
        """测试集评估（mask 掉 train + valid 物品）。"""
        mask_items_by_user = defaultdict(set)
        for user_idx, items in self.train_items_by_user.items():
            mask_items_by_user[user_idx].update(items)
        for user_idx, items in self.valid_items_by_user.items():
            mask_items_by_user[user_idx].update(items)
        return self._evaluate_split(
            self.test_items_by_user, mask_items_by_user,
            label="Test", verbose=True)

    def _evaluate_split(self, eval_items_by_user, mask_items_by_user,
                         label="Test", verbose=True):
        """
        计算 Recall, MRR, NDCG, Hit, Precision @ topn。
        """
        N = self.topn
        total_hits = 0
        precision_sum = 0.0
        recall_sum = 0.0
        ndcg_sum = 0.0
        mrr_sum = 0.0
        user_hit_count = 0
        eval_user_count = 0

        self.model.eval()
        with torch.no_grad():
            all_users_emb, all_items_emb = self.model.computer()
            user_emb = all_users_emb.detach()
            item_emb = all_items_emb.detach()

        for user_idx in range(len(self.user_ids)):
            if verbose and user_idx % 500 == 0:
                print("%s topn evaluate for %d users" % (label.lower(), user_idx),
                      file=sys.stderr)

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
            user_hit = 0
            for rank, item_idx in enumerate(top_idx, start=1):
                if int(item_idx) in eval_items:
                    total_hits += 1
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
                user_hit_count += 1
            eval_user_count += 1

        precision = precision_sum / eval_user_count if eval_user_count else 0
        recall = recall_sum / eval_user_count if eval_user_count else 0
        ndcg = ndcg_sum / eval_user_count if eval_user_count else 0
        mrr = mrr_sum / eval_user_count if eval_user_count else 0
        hit_rate = user_hit_count / eval_user_count if eval_user_count else 0

        metrics = {
            "recall": recall, "mrr": mrr, "ndcg": ndcg,
            "hit": hit_rate, "precision": precision,
            "users": eval_user_count, "hits": total_hits,
        }

        if verbose:
            print(
                "测试集 %s  RECALL@%d : %.4f    MRR@%d : %.4f    NDCG@%d : %.4f    "
                "HIT@%d : %.4f    PRECISION@%d : %.4f"
                % (label, N, recall, N, mrr, N, ndcg, N, hit_rate, N, precision))

        return metrics

    # ==================================================================
    #  推荐生成
    # ==================================================================

    def generate_recommendation(self, filepath="./outputs/rwgcn_recommendation.csv",
                                 topn=None, mask_valid=False, progress=True):
        """
        输出 per-user Top-N 推荐 CSV。

        CSV 格式: user_id, rec1, rec2, ..., recN
        """
        topn = topn or self.recommendation_topn
        print("generating RWGCN recommendation result: %s" % filepath)
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            all_users_emb, all_items_emb = self.model.computer()
            user_emb = all_users_emb.detach()
            item_emb = all_items_emb.detach()

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id"] + ["rec%d" % i for i in range(1, topn + 1)])
            for user_idx, user_id in enumerate(self.user_ids):
                if progress and user_idx % 500 == 0:
                    print("generate RWGCN recommendation for %d users" % user_idx,
                          file=sys.stderr)

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

        print("RWGCN recommendation written to %s" % filepath)

    def gernate_recommendation(self):
        """别名。"""
        self.generate_recommendation()


# ==============================================================================
#  CLI（独立运行）
# ==============================================================================

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="RWGCN: Rating-Weighted Graph Convolution Network")
    parser.add_argument("--ratings-file",
                        default="./data/ml-1m/ratings.dat")
    parser.add_argument("--users-file",
                        default="./data/ml-1m/users.dat")
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--recommendation-topn", type=int, default=100)
    parser.add_argument("--rating-threshold", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--w-min", type=float, default=0.2)
    parser.add_argument("--loss-scheme", type=str, default="A",
                        help="A (标准 BPR) 或 B (加权 BPR)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--reg-weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--valid-interval", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--save-epoch-recommendations", action="store_true")
    parser.add_argument("--epoch-recommendation-dir",
                        default="./outputs/rwgcn_epoch_recommendations")
    parser.add_argument("--skip-recommendation", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    rwgcn = RWGCN(
        topn=args.topn,
        recommendation_topn=args.recommendation_topn,
        rating_threshold=args.rating_threshold,
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers,
        w_min=args.w_min,
        loss_scheme=args.loss_scheme,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        reg_weight=args.reg_weight,
        seed=args.seed,
        valid_interval=args.valid_interval,
        early_stop_patience=args.early_stop_patience,
        min_delta=args.min_delta,
        save_epoch_recommendations=args.save_epoch_recommendations,
        epoch_recommendation_dir=args.epoch_recommendation_dir,
    )
    rwgcn.generate_dataset(args.ratings_file, usersfile=args.users_file)
    rwgcn.calc_movie_sim()
    rwgcn.evaluate()
    if not args.skip_recommendation:
        rwgcn.generate_recommendation(topn=args.recommendation_topn, mask_valid=True)


if __name__ == "__main__":
    main()
