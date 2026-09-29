# Dota2-BP-Predection

## 清洗最近下载的比赛

清洗脚本只读取本地 JSON，不访问网络。默认输入为 `data/raw/match_details`，生成严格最小字段的 Parquet 文件到 `data/processed/test`，并重建 `data/database/dota2.duckdb` 中带 `test_` 前缀的四张表：

```powershell
py -m src.normalize --input-dir data/raw/match_details --output-dir data/processed/test --database data/database/dota2.duckdb --force
```

输出包括 `matches.parquet`、`match_players.parquet`、`draft_actions.parquet`、`normalization_manifest.parquet` 和 `normalization_summary.json`。重复运行会整表重建，不会产生重复行。可用以下命令进行只读核验：

```powershell
py scripts/verify_normalization.py --database data/database/dota2.duckdb
```

其他参数包括 `--limit N`、`--match-id MATCH_ID`、`--no-parquet`、`--no-database` 和 `--log-level`。不能同时使用两个 `--no-*` 参数。

## 构建模型数据与统计基线

完成标准化后，按顺序运行：

```powershell
python scripts/profile_dataset.py
python scripts/build_dataset_splits.py
python scripts/build_draft_samples.py
python scripts/build_match_rosters.py
python scripts/evaluate_statistical_baselines.py
```

数据质量分析输出到 `data/processed/profile`，时间划分、逐手 BP 样本和阵容输出到 `data/processed/model`，统计基线结果输出到 `data/processed/baselines`。所有基线统计只使用 `train`；`validation` 用于选择组合方式，`test` 用于最终评估。

可运行以下命令检查数据划分和逐手历史是否一致：

```powershell
python scripts/verify_dataset_pipeline.py
python -m pytest -q
```

## 训练神经网络

训练只更新 `train`，并根据 `validation` Top-5 保存最佳 checkpoint。三个变体分别是：Model A 使用 BP 状态与基础环境；Model B 增加双方队伍 Embedding；Model C 再增加双方五名选手 Embedding。

```powershell
python -m scripts.train_draft_mlp --model a --epochs 15 --batch-size 2048 --patience 3
python -m scripts.train_draft_mlp --model b --epochs 15 --batch-size 2048 --patience 3
python -m scripts.train_draft_mlp --model c --epochs 15 --batch-size 2048 --patience 3
python -m scripts.train_draft_mlp --model b --output-dir data/processed/neural/model_b_tuned_1 --epochs 30 --batch-size 2048 --patience 5 --scheduler-patience 2
python -m scripts.train_draft_mlp --model b --output-dir data/processed/neural/model_b_plus --epochs 30 --batch-size 2048 --patience 5 --scheduler-patience 2 --lineup-encoder
python -m scripts.train_draft_mlp --model b --output-dir data/processed/neural/model_b_plus_recency120 --epochs 30 --batch-size 2048 --patience 5 --scheduler-patience 2 --lineup-encoder --recency-half-life-days 120
python -m scripts.train_draft_mlp --model b --output-dir data/processed/neural/model_b_plus_recency120_attention --epochs 30 --batch-size 2048 --patience 5 --scheduler-patience 2 --lineup-attention --recency-half-life-days 120
python -m scripts.build_hero_role_priors
python -m scripts.train_draft_mlp --model b --output-dir data/processed/neural/model_b_plus_recency120_roles --epochs 40 --batch-size 2048 --patience 5 --scheduler-patience 2 --lineup-encoder --recency-half-life-days 120 --role-coverage
python -m scripts.train_draft_mlp --model b --output-dir data/processed/neural/model_b_plus_recency60_candidate --epochs 40 --batch-size 2048 --patience 5 --scheduler-patience 2 --lineup-encoder --recency-half-life-days 60 --candidate-scorer
python -m scripts.train_draft_mlp --model b --output-dir data/processed/neural/model_b_plus_recency60_sequence --epochs 40 --batch-size 512 --accumulation-steps 4 --patience 5 --scheduler-patience 2 --lineup-encoder --recency-half-life-days 60 --sequence-encoder
python -m scripts.train_draft_mlp --model b --output-dir data/processed/neural/model_b_plus_outcome_weighted --epochs 40 --batch-size 512 --accumulation-steps 4 --patience 5 --scheduler-patience 2 --lineup-encoder --recency-half-life-days 60 --outcome-weighted-pick-loss --winning-pick-weight 1.5 --losing-pick-weight 0.5 --seed 42
python -m scripts.train_draft_value --output-dir data/processed/neural/draft_value --epochs 40 --batch-size 512 --accumulation-steps 4 --patience 5 --scheduler-patience 2 --recency-half-life-days 60 --seed 42
python -m scripts.train_draft_transformer --output-dir data/processed/neural/model_d --epochs 30 --batch-size 2048 --patience 5 --scheduler-patience 2
python scripts/compare_neural_models.py
```

输出位于 `data/processed/neural` 下对应的模型目录，统一验证集对比写入 `model_comparison.json`。模型开发期间不使用 `test`；确定结构和超参数后，再使用 `train + validation` 重训并进行一次最终测试。
