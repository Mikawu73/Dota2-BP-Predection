# Recorded Validation Results

Generated with `python -m scripts.export_portfolio_report`. See [results.json](results.json) for configuration and source-summary hashes.

Historical statistical baseline and old model_comparison.json are excluded: their data snapshot predates this cohort. Seed 42 original B+ did not record physical batch size; later runs used 512 x 4 accumulation. These are exploratory comparisons, not a fully controlled multi-seed benchmark.

These are validation model-selection results, not a final test benchmark. A single-seed maximum is not an estimate of typical performance.

| Run | Seed | Best epoch | Top-5 (%) | MRR |
| --- | ---: | ---: | ---: | ---: |
| decay_current_none | 42 | 14 | 30.5061 | 0.2194 |
| decay_current_60 | 42 | 23 | 31.5928 | 0.2271 |
| decay_current_120 | 42 | 23 | 31.4471 | 0.2264 |
| decay_current_240 | 42 | 23 | 30.8982 | 0.2235 |
| model_b_plus_recency60_seed7 | 7 | 21 | 30.7336 | 0.2224 |
| model_b_plus_recency60_seed2026 | 2026 | 17 | 31.3907 | 0.2257 |
| model_b_plus_recency60_sequence | 42 | 14 | 31.1721 | 0.2249 |
| model_b_plus_recency60_sequence_seed7 | 7 | 19 | 31.3720 | 0.2257 |
| model_b_plus_recency60_sequence_seed2026 | 2026 | 17 | 31.2339 | 0.2248 |
| model_b_plus_recency60_sequence_multitask | 42 | 19 | 31.2085 | 0.2249 |
| model_b_plus_recency60_sequence_role_multitask | 42 | 13 | 31.3355 | 0.2258 |
| model_b_plus_outcome_weighted | 42 | 19 | 31.1654 | 0.2251 |

## Value Model

Best epoch 1: AUC 0.5664, log loss 0.69151, Brier 0.24860.
Evaluated on actual historical Picks. This does not establish the benefit of choosing a different hero.

## Interpretation

- B+ with 60-day decay is the current behavior-model candidate, pending final test evaluation.
- Ordered Transformer results vary by seed; there is no established improvement in the mean.
- Outcome and position auxiliary tasks have only single-seed evidence in this cohort.
- Outcome-weighted imitation is not a win-rate estimator.
- Value-model validation loss worsened after epoch 1. Candidate contribution has not been isolated from team and draft-context effects.
- Rows from the same match are dependent. Future uncertainty estimates should resample matches, not individual actions.
