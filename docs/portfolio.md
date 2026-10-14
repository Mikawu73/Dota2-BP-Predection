# Portfolio Copy

## Project Card

**Dota 2 Draft Prediction**

Built an end-to-end machine learning pipeline for professional Dota 2 pick/ban prediction, covering 25,158 eligible matches and 603,640 draft actions. Explored team embeddings, DeepSets, Transformers, and auxiliary tasks under chronological evaluation. The B+ behavior model recorded 31.6% validation Top-5 accuracy.

**Stack:** Python, PyTorch, DuckDB, Apache Parquet, pytest.

## Highlights

- Validated raw API responses and transformed them into normalized analytical tables.
- Split matches chronologically before expanding them into action-level examples.
- Tested compact lineup representations against more complex sequence models.
- Recorded unsuccessful experiments and distinguished behavior prediction from outcome prediction.
- Added synthetic pipeline/model tests and a Windows/Linux CI workflow.
- Built a checkpoint-only inference CLI and an interactive local BP sandbox with shared feature encoding.
- Added a data/results homepage with temporal splits, filterable experiment metrics, configuration details and a linked sandbox.

## Reporting Notes

Use "validation Top-5" wherever 31.6% appears. Do not call it match win rate or final test accuracy. Do not claim a 10.7-point baseline improvement until the baseline is rerun on the same snapshot. The independent value model is exploratory, not a demonstrated win-rate optimizer.

The README contains a Mermaid architecture diagram for GitHub. The results JSON can supply portfolio tables. The interactive BP demo runs locally using a trusted checkpoint; a publicly hosted demo and downloadable model release are not available yet.
