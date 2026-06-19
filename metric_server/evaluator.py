import csv
import io
import math
import os
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

REFERENCE_ROWS = [
    {"method": "BPR", "recall": 0.1776, "mrr": 0.4187, "ndcg": 0.2401, "hit": 0.7199, "precision": 0.1779},
    {"method": "NeuMF", "recall": 0.1651, "mrr": 0.4020, "ndcg": 0.2271, "hit": 0.7029, "precision": 0.1700},
    {"method": "NGCF", "recall": 0.1814, "mrr": 0.4354, "ndcg": 0.2508, "hit": 0.7239, "precision": 0.1850},
    {"method": "LightGCN", "recall": 0.1861, "mrr": 0.4388, "ndcg": 0.2538, "hit": 0.7330, "precision": 0.1863},
    {"method": "LightGCL", "recall": 0.1867, "mrr": 0.4283, "ndcg": 0.2479, "hit": 0.7370, "precision": 0.1815},
    {"method": "SGL", "recall": 0.1889, "mrr": 0.4315, "ndcg": 0.2505, "hit": 0.7392, "precision": 0.1843},
    {"method": "HMLET", "recall": 0.1847, "mrr": 0.4297, "ndcg": 0.2490, "hit": 0.7305, "precision": 0.1836},
    {"method": "NCL", "recall": 0.2021, "mrr": 0.4599, "ndcg": 0.2702, "hit": 0.7565, "precision": 0.1962},
    {"method": "SimGCL", "recall": 0.2029, "mrr": 0.4550, "ndcg": 0.2667, "hit": 0.7640, "precision": 0.1933},
    {"method": "XSimGCL", "recall": 0.2116, "mrr": 0.4638, "ndcg": 0.2750, "hit": 0.7743, "precision": 0.1987},
]


@dataclass
class SubmittedFile:
    name: str
    content: bytes


class ML1MEvaluator:
    def __init__(self, data_dir: str, topk: int = 10, seed: int = 2020):
        self.data_dir = data_dir
        self.topk = topk
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.user_ids: List[int] = []
        self.item_ids: List[int] = []
        self.train_items_by_user: Dict[int, set] = defaultdict(set)
        self.valid_items_by_user: Dict[int, set] = defaultdict(set)
        self.test_items_by_user: Dict[int, set] = defaultdict(set)
        self.stats = {}
        self._load_and_split()

    def _load_and_split(self) -> None:
        users_file = os.path.join(self.data_dir, "users.dat")
        ratings_file = os.path.join(self.data_dir, "ratings.dat")

        with open(users_file, "r", encoding="latin-1") as f:
            self.user_ids = sorted(int(line.rstrip("\n").split("::", 1)[0]) for line in f)

        interactions: List[Tuple[int, int]] = []
        with open(ratings_file, "r", encoding="latin-1") as f:
            for line in f:
                user, item, rating, _ = line.rstrip("\n").split("::")
                if int(rating) >= 3:
                    interactions.append((int(user), int(item)))

        self.rng.shuffle(interactions)
        self.item_ids = sorted({item for _, item in interactions})

        interactions_by_user: Dict[int, List[int]] = defaultdict(list)
        for user, item in interactions:
            interactions_by_user[user].append(item)

        train_count = 0
        valid_count = 0
        test_count = 0
        ratios = [0.8, 0.1, 1 - 0.8 - 0.1]
        for user in self.user_ids:
            items = interactions_by_user.get(user, [])
            if not items:
                continue
            train_items, valid_items, test_items = self._split_user_items(items, ratios)
            self.train_items_by_user[user].update(train_items)
            self.valid_items_by_user[user].update(valid_items)
            self.test_items_by_user[user].update(test_items)
            train_count += len(train_items)
            valid_count += len(valid_items)
            test_count += len(test_items)

        self.stats = {
            "users": len(self.user_ids),
            "items": len(self.item_ids),
            "interactions": len(interactions),
            "train": train_count,
            "valid": valid_count,
            "test": test_count,
            "topk": self.topk,
            "seed": self.seed,
        }

    @staticmethod
    def _split_user_items(items: Sequence[int], ratios: Sequence[float]) -> Tuple[Sequence[int], Sequence[int], Sequence[int]]:
        total = len(items)
        norm_ratios = [ratio / sum(ratios) for ratio in ratios]
        counts = [int(norm_ratios[i] * total) for i in range(len(norm_ratios))]
        counts[0] = total - sum(counts[1:])
        for i in range(1, len(norm_ratios)):
            if counts[0] <= 1:
                break
            if 0 < norm_ratios[-i] * total < 1:
                counts[-i] += 1
                counts[0] -= 1
        train_end = counts[0]
        valid_end = counts[0] + counts[1]
        return items[:train_end], items[train_end:valid_end], items[valid_end:]

    def evaluate_submission_files(self, submitted_files: Sequence[SubmittedFile], method_name: str = "Your Result") -> dict:
        csv_files = self._expand_files(submitted_files)
        if not csv_files:
            raise ValueError("没有找到 CSV 文件。请上传 .csv，或包含 .csv 的 .zip。")

        grouped = {}
        for filename, content in csv_files:
            epoch, split = self._classify_file(filename)
            grouped[(epoch, split)] = self.parse_recommendations(content)

        valid_epochs = sorted(epoch for (epoch, split) in grouped if split == "valid")
        test_epochs = sorted(epoch for (epoch, split) in grouped if split == "test")

        if valid_epochs:
            candidates = []
            for epoch in valid_epochs:
                if (epoch, "test") not in grouped:
                    continue
                valid_metrics = self.evaluate_recommendations(
                    grouped[(epoch, "valid")],
                    self.valid_items_by_user,
                    self.train_items_by_user,
                )
                candidates.append((valid_metrics["mrr"], epoch, valid_metrics))
            if not candidates:
                raise ValueError("找到了 valid CSV，但没有找到同 epoch 的 test CSV。请使用 epoch_001_valid.csv 和 epoch_001_test.csv 这样的命名。")
            candidates.sort(key=lambda row: (-row[0], row[1]))
            _, best_epoch, valid_metrics = candidates[0]
            mask = self._merge_masks(self.train_items_by_user, self.valid_items_by_user)
            test_metrics = self.evaluate_recommendations(grouped[(best_epoch, "test")], self.test_items_by_user, mask)
            selection = {
                "mode": "best_valid_mrr",
                "best_epoch": best_epoch,
                "valid_mrr": valid_metrics["mrr"],
                "evaluated_epochs": len(candidates),
            }
        else:
            if len(test_epochs) != 1:
                raise ValueError("没有 valid CSV 时，只能上传 1 个最终 test CSV；若上传多 epoch，请同时提供 valid/test CSV 以便按 valid MRR 选 best。")
            best_epoch = test_epochs[0]
            mask = self._merge_masks(self.train_items_by_user, self.valid_items_by_user)
            test_metrics = self.evaluate_recommendations(grouped[(best_epoch, "test")], self.test_items_by_user, mask)
            selection = {
                "mode": "single_test_file",
                "best_epoch": best_epoch,
                "valid_mrr": None,
                "evaluated_epochs": 1,
            }

        user_row = {
            "method": method_name or "Your Result",
            "recall": test_metrics["recall"],
            "mrr": test_metrics["mrr"],
            "ndcg": test_metrics["ndcg"],
            "hit": test_metrics["hit"],
            "precision": test_metrics["precision"],
        }
        return {
            "standard": self.standard_summary(),
            "selection": selection,
            "metrics": test_metrics,
            "table": REFERENCE_ROWS + [user_row],
        }

    def standard_summary(self) -> dict:
        return {
            "dataset": "MovieLens-1M",
            "filtering": "rating >= 3",
            "evaluation": "ratio-based 8:1:1, per-user split, full sort",
            "metrics": ["Recall@10", "MRR@10", "NDCG@10", "Hit@10", "Precision@10"],
            "valid_metric": "MRR@10",
            "embedding_size": 64,
            "epochs": 500,
            "train_batch_size": 4096,
            "eval_batch_size": 4096000,
            "stats": self.stats,
        }

    @staticmethod
    def _merge_masks(*maps: Dict[int, set]) -> Dict[int, set]:
        merged: Dict[int, set] = defaultdict(set)
        for item_map in maps:
            for user, items in item_map.items():
                merged[user].update(items)
        return merged

    @staticmethod
    def _expand_files(files: Sequence[SubmittedFile]) -> List[Tuple[str, bytes]]:
        expanded = []
        for file in files:
            lower_name = file.name.lower()
            if lower_name.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(file.content)) as zf:
                    for info in zf.infolist():
                        if info.is_dir() or not info.filename.lower().endswith(".csv"):
                            continue
                        expanded.append((info.filename, zf.read(info)))
            elif lower_name.endswith(".csv"):
                expanded.append((file.name, file.content))
        return expanded

    @staticmethod
    def _classify_file(filename: str) -> Tuple[int, str]:
        basename = os.path.basename(filename).lower()
        split = "test"
        if "valid" in basename or re.search(r"(^|[_\-.])val([_\-.]|$)", basename):
            split = "valid"
        elif "test" in basename:
            split = "test"

        epoch = 1
        match = re.search(r"epoch[_\-.]?(\d+)", basename)
        if not match:
            match = re.search(r"(^|[_\-.])e(\d+)([_\-.]|$)", basename)
        if match:
            epoch = int(match.group(1) if match.lastindex == 1 else match.group(2))
        return epoch, split

    def parse_recommendations(self, content: bytes) -> Dict[int, List[int]]:
        text = content.decode("utf-8-sig", errors="ignore")
        sample = text[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            dialect = csv.excel

        rows = list(csv.reader(io.StringIO(text), dialect))
        rows = [row for row in rows if row and any(cell.strip() for cell in row)]
        if not rows:
            return {}

        header = [cell.strip().lower() for cell in rows[0]]
        if "user_id" in header and "item_id" in header:
            return self._parse_long_recommendations(rows[1:], header)

        start_idx = 0
        if not self._is_int(rows[0][0]):
            start_idx = 1

        recs: Dict[int, List[int]] = {}
        for row in rows[start_idx:]:
            if not row or not self._is_int(row[0]):
                continue
            user = int(row[0])
            items = []
            for cell in row[1:]:
                cell = cell.strip()
                if self._is_int(cell):
                    items.append(int(cell))
            recs[user] = items
        return recs

    def _parse_long_recommendations(self, rows: Iterable[List[str]], header: List[str]) -> Dict[int, List[int]]:
        user_idx = header.index("user_id")
        item_idx = header.index("item_id")
        rank_idx = header.index("rank") if "rank" in header else None
        score_idx = header.index("score") if "score" in header else None

        grouped = defaultdict(list)
        for order, row in enumerate(rows):
            if len(row) <= max(user_idx, item_idx):
                continue
            if not self._is_int(row[user_idx]) or not self._is_int(row[item_idx]):
                continue
            user = int(row[user_idx])
            item = int(row[item_idx])
            if rank_idx is not None and len(row) > rank_idx and self._is_float(row[rank_idx]):
                key = float(row[rank_idx])
            elif score_idx is not None and len(row) > score_idx and self._is_float(row[score_idx]):
                key = -float(row[score_idx])
            else:
                key = float(order)
            grouped[user].append((key, item))

        return {user: [item for _, item in sorted(values)] for user, values in grouped.items()}

    def evaluate_recommendations(self, recommendations: Dict[int, List[int]], targets: Dict[int, set], masks: Dict[int, set]) -> dict:
        precision_sum = 0.0
        recall_sum = 0.0
        ndcg_sum = 0.0
        mrr_sum = 0.0
        hit_sum = 0.0
        hit_count = 0
        eval_users = 0

        for user in self.user_ids:
            target_items = targets.get(user, set())
            if not target_items:
                continue
            ranked_items = self._filtered_topk(recommendations.get(user, []), masks.get(user, set()))
            user_hits = 0
            dcg = 0.0
            first_hit = 0.0
            for rank, item in enumerate(ranked_items, start=1):
                if item in target_items:
                    user_hits += 1
                    hit_count += 1
                    dcg += 1 / math.log2(rank + 1)
                    if first_hit == 0.0:
                        first_hit = 1 / rank

            ideal_hits = min(len(target_items), self.topk)
            idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            precision_sum += user_hits / self.topk
            recall_sum += user_hits / len(target_items)
            ndcg_sum += dcg / idcg if idcg else 0.0
            mrr_sum += first_hit
            hit_sum += 1.0 if user_hits > 0 else 0.0
            eval_users += 1

        if eval_users == 0:
            raise ValueError("评估集中没有可评估用户。请检查 split 或上传文件。")

        return {
            "recall": recall_sum / eval_users,
            "mrr": mrr_sum / eval_users,
            "ndcg": ndcg_sum / eval_users,
            "hit": hit_sum / eval_users,
            "precision": precision_sum / eval_users,
            "hit_count": hit_count,
            "eval_users": eval_users,
        }

    def _filtered_topk(self, items: Sequence[int], masked_items: set) -> List[int]:
        seen = set()
        filtered = []
        item_universe = set(self.item_ids)
        for item in items:
            if item in seen or item in masked_items or item not in item_universe:
                continue
            seen.add(item)
            filtered.append(item)
            if len(filtered) >= self.topk:
                break
        return filtered

    @staticmethod
    def _is_int(value: str) -> bool:
        try:
            int(str(value).strip())
            return True
        except ValueError:
            return False

    @staticmethod
    def _is_float(value: str) -> bool:
        try:
            float(str(value).strip())
            return True
        except ValueError:
            return False
