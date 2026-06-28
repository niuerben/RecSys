# MovieLens-1M Metric Server

FastAPI 测评机，按 RecBole-GNN ML-1M 设置统一计算：

- `rating >= 3`
- ratio-based `8:1:1`
- full sort
- valid `MRR@10` 选择 best epoch
- `Recall@10 / MRR@10 / NDCG@10 / Hit@10 / Precision@10`

## 启动

```bash
pip install -r metric_server/requirements.txt
uvicorn metric_server.main:app --reload --host 0.0.0.0 --port 8000
```

```ps1
pip install -r metric_server/requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

打开：
```text
http://127.0.0.1:8000
```

## 上传格式

支持两类。

### 单个最终 test CSV

适合只测最终推荐结果。格式可以是宽表：

```csv
user_id,recommendations1,recommendations2,...,recommendations100
1,1193,661,...,260
2,1357,3068,...,1198
```

也支持无表头：

```csv
1,1193,661,...,260
2,1357,3068,...,1198
```

### 多 epoch valid/test CSV 或 ZIP

适合测评机统一选择 best checkpoint。推荐命名：

```text
epoch_001_valid.csv
epoch_001_test.csv
epoch_002_valid.csv
epoch_002_test.csv
...
epoch_500_valid.csv
epoch_500_test.csv
```

后端会：

1. 对每个 epoch 的 valid 推荐算 `MRR@10`
2. 选择 valid `MRR@10` 最高的 epoch
3. 用同 epoch 的 test 推荐算最终指标

CSV 可以使用宽表，也可以使用长表：

```csv
user_id,item_id,rank
1,1193,1
1,661,2
```

或：

```csv
user_id,item_id,score
1,1193,9.8
1,661,9.1
```

## Mask 规则

测评机侧统一执行：

- valid：mask train items
- test：mask train + valid items

因此推荐上传 Top100 或更长列表。若只上传 Top10，mask 后可能不足 10 个候选，会影响结果。

当前测评机会只使用每个用户提交的前 100 个推荐 item；之后执行 mask，再取过滤后的 Top10 计算 `Precision@10 / Recall@10 / MRR@10 / NDCG@10 / Hit@10`。

合法 item 集合来自 `ratings.dat` 中出现过的全部 item。`rating < 3` 的 item 不会被当作命中目标，但如果被推荐，会保留在候选列表中并占用 Top10 位置。
