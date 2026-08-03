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
