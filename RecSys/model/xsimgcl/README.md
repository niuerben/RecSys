# XSimGCL

Lightweight PyTorch implementation for this course project, aligned with the
RecBole-GNN MovieLens-1M result page:

- filter interactions with `rating < 3`
- split filtered interactions by user with `8:1:1`
- use full-sort Top-10 evaluation
- evaluate `Recall@10`, `MRR@10`, `NDCG@10`, `Hit@10`, and `Precision@10`

Official ML-1M XSimGCL hyper-parameters used here:

```text
epochs=500
train_batch_size=4096
embedding_size=64
learning_rate=0.002
n_layers=2
reg_weight=0.0001
temperature=0.2
lambda=0.1
eps=0.2
layer_cl=1
valid_metric=MRR@10
user nodes=6040 from users.dat
```

Run the aligned reproduction:

```bash
python RS-torch/XSimGCL/xsimgcl.py
```

For a quick smoke test:

```bash
python RS-torch/XSimGCL/xsimgcl.py --epochs 1 --skip-recommendation
```

To export every epoch recommendation:

```bash
python RS-torch/XSimGCL/xsimgcl.py --save-epoch-recommendations
```

It will write files like `outputs/xsimgcl_epoch_recommendations/xsimgcl_recommendation_001.csv`.

The official RecBole-GNN page reports XSimGCL on ML-1M as:

```text
Recall@10=0.2116
MRR@10=0.4638
NDCG@10=0.2750
Hit@10=0.7743
Precision@10=0.1987
```

This remains a compact standalone wrapper rather than a full RecBole-GNN
replacement, so exact numbers can differ because RecBole's internal data
batching and sampler implementations are not reused directly. The local pipeline
does align the main protocol knobs: `rating>=3`, user-grouped ratio split,
6040 user nodes from `users.dat`, validation by `MRR@10`, best-checkpoint test
evaluation, and mean-per-user ranking metrics.

New Top-N rows are written to `outputs/xsimgcl_metrics.csv` with the aligned
metric columns.
