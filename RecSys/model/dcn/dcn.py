# -*- coding: utf-8 -*-
"""
Project-level DCN runner for MovieLens 1M.

This wrapper keeps the same call style as the collaborative filtering modules:
generate dataset -> train/calc -> evaluate -> generate recommendations.
"""

import argparse
import copy
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=3):
        super().__init__()
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.randn(input_dim, 1) * 0.01) for _ in range(num_layers)]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)]
        )

    def forward(self, x0):
        xl = x0
        for weight, bias in zip(self.weights, self.biases):
            xw = xl @ weight
            xl = x0 * xw + bias + xl
        return xl


class DCNModel(nn.Module):
    def __init__(self, n_users, n_movies, n_genders, n_ages, n_occupations, genre_features):
        super().__init__()
        self.register_buffer("genre_features", genre_features)

        self.user_emb = nn.Embedding(n_users, 16)
        self.movie_emb = nn.Embedding(n_movies, 24)
        self.gender_emb = nn.Embedding(n_genders, 4)
        self.age_emb = nn.Embedding(n_ages, 4)
        self.occupation_emb = nn.Embedding(n_occupations, 8)
        self.genre_layer = nn.Linear(genre_features.shape[1], 16)

        input_dim = 16 + 24 + 4 + 4 + 8 + 16
        self.cross = CrossNetwork(input_dim, num_layers=3)
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output = nn.Linear(input_dim + 64, 1)

    def forward(self, user_idx, movie_idx, gender_idx, age_idx, occupation_idx):
        genre_vec = self.genre_features[movie_idx]
        x = torch.cat(
            [
                self.user_emb(user_idx),
                self.movie_emb(movie_idx),
                self.gender_emb(gender_idx),
                self.age_emb(age_idx),
                self.occupation_emb(occupation_idx),
                torch.relu(self.genre_layer(genre_vec)),
            ],
            dim=1,
        )
        x = torch.cat([self.cross(x), self.deep(x)], dim=1)
        return self.output(x).squeeze(1)


class DCN(object):
    def __init__(
        self,
        topn=10,
        rating_threshold=3,
        train_ratio=0.8,
        valid_ratio=0.1,
        train_neg_per_positive=2,
        epochs=5,
        batch_size=8192,
        learning_rate=1e-3,
        weight_decay=1e-6,
        seed=0,
        recommendation_topn=100,
        valid_interval=1,
        early_stop_patience=10,
        min_delta=1e-6,
        save_epoch_recommendations=False,
        epoch_recommendation_dir="./outputs",
    ):
        self.topn = topn
        self.rating_threshold = rating_threshold
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.train_neg_per_positive = train_neg_per_positive
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.recommendation_topn = recommendation_topn
        self.valid_interval = valid_interval
        self.early_stop_patience = early_stop_patience
        self.min_delta = min_delta
        self.save_epoch_recommendations = save_epoch_recommendations
        self.epoch_recommendation_dir = epoch_recommendation_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.metrics = None

        self.user_ids = []
        self.movie_ids = []
        self.user2idx = {}
        self.movie2idx = {}
        self.idx2movie = {}
        self.user_features = {}
        self.user_feature_matrix = None
        self.movie_index_lookup = None
        self.rated_by_user = defaultdict(set)
        self.train_rated_by_user = defaultdict(set)
        self.valid_positive_by_user = defaultdict(set)
        self.test_positive_by_user = defaultdict(set)
        self.best_epoch = 0
        self.best_valid_mrr = -1.0

    def generate_dataset(self, mergedfile):
        print("使用设备:%s" % self.device)
        print("加载 DCN 数据...")
        full_data = pd.read_csv(mergedfile)
        data = full_data[full_data["rating"] >= self.rating_threshold].copy()
        rng = np.random.default_rng(self.seed)

        self.user_ids = sorted(full_data["user_id"].unique())
        self.movie_ids = sorted(data["movie_id"].unique())
        self.user2idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.movie2idx = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
        self.idx2movie = {idx: movie_id for movie_id, idx in self.movie2idx.items()}
        self.movie_index_lookup = np.full(max(self.movie_ids) + 1, -1, dtype=np.int64)
        for movie_id, movie_idx in self.movie2idx.items():
            self.movie_index_lookup[int(movie_id)] = movie_idx

        gender2idx = {value: idx for idx, value in enumerate(sorted(full_data["gender"].unique()))}
        age2idx = {value: idx for idx, value in enumerate(sorted(full_data["age"].unique()))}
        occupation2idx = {
            value: idx for idx, value in enumerate(sorted(full_data["occupation"].unique()))
        }

        users = full_data[["user_id", "gender", "age", "occupation"]].drop_duplicates("user_id")
        for row in users.itertuples(index=False):
            self.user_features[int(row.user_id)] = (
                gender2idx[row.gender],
                age2idx[row.age],
                occupation2idx[row.occupation],
            )
        self.user_feature_matrix = np.zeros((len(self.user_ids), 3), dtype=np.int64)
        for user_id, user_idx in self.user2idx.items():
            self.user_feature_matrix[user_idx] = self.user_features[int(user_id)]

        movie_genres = data[["movie_id", "genres"]].drop_duplicates("movie_id")
        genres = sorted({genre for text in movie_genres["genres"] for genre in str(text).split("|")})
        genre2idx = {genre: idx for idx, genre in enumerate(genres)}
        genre_features = np.zeros((len(self.movie_ids), len(genres)), dtype=np.float32)
        for row in movie_genres.itertuples(index=False):
            movie_idx = self.movie2idx[int(row.movie_id)]
            for genre in str(row.genres).split("|"):
                genre_features[movie_idx, genre2idx[genre]] = 1.0
        self.genre_features = torch.tensor(genre_features, dtype=torch.float32)

        rows = data[["user_id", "movie_id", "rating"]].to_numpy(dtype=np.int32)
        rng.shuffle(rows)
        train_split = int(len(rows) * self.train_ratio)
        valid_split = int(len(rows) * (self.train_ratio + self.valid_ratio))
        train_rows = rows[:train_split]
        valid_rows = rows[train_split:valid_split]
        test_rows = rows[valid_split:]

        for user_id, movie_id, _ in rows:
            self.rated_by_user[int(user_id)].add(int(movie_id))
        for user_id, movie_id, _ in train_rows:
            self.train_rated_by_user[int(user_id)].add(int(movie_id))
        for user_id, movie_id, rating in valid_rows:
            self.valid_positive_by_user[int(user_id)].add(int(movie_id))
        for user_id, movie_id, rating in test_rows:
            self.test_positive_by_user[int(user_id)].add(int(movie_id))

        self.train_arrays = self._build_train_arrays(train_rows, rng)

        print(
            "用户数:%d，电影数:%d，交互数:%d，训练:%d，Top%d 推荐列: %d"
            % (
                len(self.user_ids),
                len(self.movie_ids),
                len(rows),
                len(train_rows),
                self.recommendation_topn,
                len(self.user_ids) * self.recommendation_topn,
            )
        )

    def gernate_dataset(self, mergedfile):
        self.generate_dataset(mergedfile)

    def _build_train_arrays(self, rows, rng):
        user_arr = []
        movie_arr = []
        label_arr = []

        positives_by_user = defaultdict(int)
        for user_id, movie_id, rating in rows:
            user_arr.append(self.user2idx[int(user_id)])
            movie_arr.append(self.movie2idx[int(movie_id)])
            label_arr.append(1)
            positives_by_user[int(user_id)] += 1

        all_movies = np.array(self.movie_ids, dtype=np.int32)
        for user_id in self.user_ids:
            rated = self.rated_by_user[int(user_id)]
            candidates = np.array([movie_id for movie_id in all_movies if int(movie_id) not in rated], dtype=np.int32)
            if len(candidates) == 0:
                continue
            sample_size = positives_by_user[int(user_id)] * self.train_neg_per_positive
            if sample_size <= 0:
                continue
            sampled = rng.choice(candidates, size=sample_size, replace=sample_size > len(candidates))
            user_arr.extend([self.user2idx[int(user_id)]] * len(sampled))
            movie_arr.extend(self.movie_index_lookup[sampled].tolist())
            label_arr.extend([0] * len(sampled))

        user_arr = np.asarray(user_arr, dtype=np.int64)
        movie_arr = np.asarray(movie_arr, dtype=np.int64)
        label_arr = np.asarray(label_arr, dtype=np.float32)
        order = rng.permutation(len(label_arr))
        return user_arr[order], movie_arr[order], label_arr[order]

    def _make_loader(self, arrays, shuffle=False):
        user_arr, movie_arr, label_arr = arrays
        user_feature_arr = self.user_feature_matrix[user_arr]
        gender_arr = user_feature_arr[:, 0]
        age_arr = user_feature_arr[:, 1]
        occupation_arr = user_feature_arr[:, 2]

        dataset = TensorDataset(
            torch.tensor(user_arr, dtype=torch.long),
            torch.tensor(movie_arr, dtype=torch.long),
            torch.tensor(gender_arr, dtype=torch.long),
            torch.tensor(age_arr, dtype=torch.long),
            torch.tensor(occupation_arr, dtype=torch.long),
            torch.tensor(label_arr, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def calc_movie_sim(self):
        print("加载模型 DCN...")
        self.model = DCNModel(
            n_users=len(self.user_ids),
            n_movies=len(self.movie_ids),
            n_genders=2,
            n_ages=len({features[1] for features in self.user_features.values()}),
            n_occupations=len({features[2] for features in self.user_features.values()}),
            genre_features=self.genre_features,
        ).to(self.device)

        loader = self._make_loader(self.train_arrays, shuffle=True)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()
        best_state_dict = None
        stale_validations = 0

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            total_count = 0
            for user_idx, movie_idx, gender_idx, age_idx, occupation_idx, labels in loader:
                user_idx = user_idx.to(self.device)
                movie_idx = movie_idx.to(self.device)
                gender_idx = gender_idx.to(self.device)
                age_idx = age_idx.to(self.device)
                occupation_idx = occupation_idx.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(user_idx, movie_idx, gender_idx, age_idx, occupation_idx)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                total_count += labels.size(0)
            message = "Epoch %d/%d - loss: %.4f" % (epoch, self.epochs, total_loss / total_count)
            if self.valid_interval > 0 and epoch % self.valid_interval == 0:
                valid_metrics = self._evaluate_topn_split(
                    self.valid_positive_by_user,
                    self.train_rated_by_user,
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
                    message += " - stale:%d/%d" % (stale_validations, self.early_stop_patience)

            if self.save_epoch_recommendations:
                filepath = os.path.join(
                    self.epoch_recommendation_dir,
                    "dcn_epoch_%03d_recommender.csv" % epoch,
                )
                self.generate_recommendation(filepath=filepath, topn=self.recommendation_topn, progress=False)
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

    def evaluate(self):
        self.evaluate_topn()

    def evaluate_classification(self, threshold=0.5, eval_neg_per_user=1900):
        rng = np.random.default_rng(self.seed + 1)
        eval_rows = []
        for user_id, movies in self.test_positive_by_user.items():
            for movie_id in movies:
                eval_rows.append([user_id, movie_id, self.rating_threshold])
        if not eval_rows:
            print("No positive test samples for classification evaluation.")
            return

        user_arr = []
        movie_arr = []
        label_arr = []
        all_movies = np.array(self.movie_ids, dtype=np.int32)
        for user_id, movie_id, _ in eval_rows:
            user_arr.append(self.user2idx[int(user_id)])
            movie_arr.append(self.movie2idx[int(movie_id)])
            label_arr.append(1)
            rated = self.rated_by_user[int(user_id)]
            candidates = np.array([mid for mid in all_movies if int(mid) not in rated], dtype=np.int32)
            if len(candidates) == 0:
                continue
            sample_size = min(eval_neg_per_user, len(candidates))
            sampled = rng.choice(candidates, size=sample_size, replace=False)
            user_arr.extend([self.user2idx[int(user_id)]] * len(sampled))
            movie_arr.extend(self.movie_index_lookup[sampled].tolist())
            label_arr.extend([0] * len(sampled))

        eval_arrays = (
            np.asarray(user_arr, dtype=np.int64),
            np.asarray(movie_arr, dtype=np.int64),
            np.asarray(label_arr, dtype=np.float32),
        )
        loader = self._make_loader(eval_arrays, shuffle=False)
        y_true, y_score = self._predict_loader(loader)
        y_pred = (y_score >= threshold).astype(np.int32)

        self.metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0,
        }

        print("======================")
        print("【分类指标评估】(threshold=%.1f)" % threshold)
        print("======================")
        print("Accuracy: %.4f" % self.metrics["accuracy"])
        print("Precision: %.4f" % self.metrics["precision"])
        print("Recall:  %.4f" % self.metrics["recall"])
        print("F1-Score: %.4f" % self.metrics["f1"])
        print("AUC: %.4f" % self.metrics["auc"])

    def evaluate_topn(self):
        mask_items_by_user = defaultdict(set)
        for user_id, items in self.train_rated_by_user.items():
            mask_items_by_user[user_id].update(items)
        for user_id, items in self.valid_positive_by_user.items():
            mask_items_by_user[user_id].update(items)
        return self._evaluate_topn_split(self.test_positive_by_user, mask_items_by_user, label="Test")

    def _evaluate_topn_split(self, eval_items_by_user, mask_items_by_user, label="Test", verbose=True):
        N = self.topn
        precision_sum = 0.0
        recall_sum = 0.0
        ndcg_sum = 0
        mrr_sum = 0.0
        hit_user_count = 0
        eval_user_count = 0
        all_movie_idx = torch.arange(len(self.movie_ids), dtype=torch.long, device=self.device)

        for i, user_id in enumerate(self.user_ids):
            if verbose and i % 500 == 0:
                print("topn evaluate for %d users" % i, file=sys.stderr)
            eval_movies = eval_items_by_user.get(int(user_id), set())
            if not eval_movies:
                continue
            scores = self._predict_user_movies(user_id, all_movie_idx)
            for movie_id in mask_items_by_user.get(int(user_id), set()):
                scores[self.movie2idx[int(movie_id)]] = -1.0
            top_idx = np.argpartition(scores, -N)[-N:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            rec_movies = [self.idx2movie[int(idx)] for idx in top_idx]

            dcg = 0
            user_hit = 0
            reciprocal_rank = 0.0
            for rank, movie_id in enumerate(rec_movies, start=1):
                if movie_id in eval_movies:
                    user_hit += 1
                    dcg += 1 / np.log2(rank + 1)
                    if reciprocal_rank == 0.0:
                        reciprocal_rank = 1 / rank

            ideal_hits = min(len(eval_movies), N)
            idcg = sum(1 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            precision_sum += user_hit / N
            recall_sum += user_hit / len(eval_movies)
            ndcg_sum += dcg / idcg if idcg else 0
            mrr_sum += reciprocal_rank
            if user_hit > 0:
                hit_user_count += 1
            eval_user_count += 1

        metrics = {
            "recall": recall_sum / eval_user_count if eval_user_count else 0,
            "mrr": mrr_sum / eval_user_count if eval_user_count else 0,
            "ndcg": ndcg_sum / eval_user_count if eval_user_count else 0,
            "hit": hit_user_count / eval_user_count if eval_user_count else 0,
            "precision": precision_sum / eval_user_count if eval_user_count else 0,
        }
        if verbose:
            print(
                "测试集 %s RECALL@%d : %.4f    MRR@%d : %.4f    NDCG@%d : %.4f    HIT@%d : %.4f    PRECISION@%d : %.4f"
                % (label, N, metrics["recall"], N, metrics["mrr"], N, metrics["ndcg"], N, metrics["hit"], N, metrics["precision"])
            )
        return metrics

    def _predict_loader(self, loader):
        self.model.eval()
        y_true = []
        y_score = []
        with torch.no_grad():
            for user_idx, movie_idx, gender_idx, age_idx, occupation_idx, labels in loader:
                logits = self.model(
                    user_idx.to(self.device),
                    movie_idx.to(self.device),
                    gender_idx.to(self.device),
                    age_idx.to(self.device),
                    occupation_idx.to(self.device),
                )
                scores = torch.sigmoid(logits).detach().cpu().numpy()
                y_score.append(scores)
                y_true.append(labels.numpy())
        return np.concatenate(y_true).astype(np.int32), np.concatenate(y_score)

    def generate_recommendation(
        self,
        filepath="./outputs/dcn_recommendation.csv",
        topn=None,
        progress=True,
    ):
        topn = topn or self.recommendation_topn
        print("generating DCN recommendation result: %s" % filepath)
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        all_movie_idx = torch.arange(len(self.movie_ids), dtype=torch.long, device=self.device)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id"] + ["rec%d" % idx for idx in range(1, topn + 1)])
            for i, user_id in enumerate(self.user_ids):
                if progress and i % 500 == 0:
                    print("generate DCN recommendation for %d users" % i, file=sys.stderr)
                scores = self._predict_user_movies(user_id, all_movie_idx)
                rated = self.train_rated_by_user[int(user_id)]
                for movie_id in rated:
                    scores[self.movie2idx[int(movie_id)]] = -1.0
                top_k = min(topn, len(scores))
                top_idx = np.argpartition(scores, -top_k)[-top_k:]
                top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
                movies = [str(self.idx2movie[int(idx)]) for idx in top_idx]
                if len(movies) < topn:
                    movies.extend([""] * (topn - len(movies)))
                writer.writerow([user_id] + movies)
        print("generate DCN recommendation result succ", file=sys.stderr)

    def gernate_recommendation(self):
        self.generate_recommendation()

    def _predict_user_movies(self, user_id, all_movie_idx):
        self.model.eval()
        gender_idx, age_idx, occupation_idx = self.user_features[int(user_id)]
        user_idx = torch.full_like(all_movie_idx, self.user2idx[int(user_id)])
        gender_tensor = torch.full_like(all_movie_idx, gender_idx)
        age_tensor = torch.full_like(all_movie_idx, age_idx)
        occupation_tensor = torch.full_like(all_movie_idx, occupation_idx)
        with torch.no_grad():
            logits = self.model(user_idx, all_movie_idx, gender_tensor, age_tensor, occupation_tensor)
            return torch.sigmoid(logits).detach().cpu().numpy()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run DCN and export Spec recommender tables.")
    parser.add_argument("--merged-file", default="./data/ml-1m/ml-1m/merged.dat")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--train-neg-per-positive", type=int, default=2)
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--recommendation-topn", type=int, default=100)
    parser.add_argument("--rating-threshold", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--valid-interval", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--save-epoch-recommendations", action="store_true")
    parser.add_argument("--epoch-recommendation-dir", default="./outputs")
    parser.add_argument("--skip-final-recommendation", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    dcn = DCN(
        topn=args.topn,
        rating_threshold=args.rating_threshold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_neg_per_positive=args.train_neg_per_positive,
        seed=args.seed,
        recommendation_topn=args.recommendation_topn,
        valid_interval=args.valid_interval,
        early_stop_patience=args.early_stop_patience,
        min_delta=args.min_delta,
        save_epoch_recommendations=args.save_epoch_recommendations,
        epoch_recommendation_dir=args.epoch_recommendation_dir,
    )
    dcn.generate_dataset(args.merged_file)
    dcn.calc_movie_sim()
    if not args.skip_evaluation:
        dcn.evaluate()
    if not args.skip_final_recommendation:
        dcn.generate_recommendation(topn=args.recommendation_topn)


if __name__ == "__main__":
    main()
