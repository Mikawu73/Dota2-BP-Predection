# Model and Data Notes

## Intended Use

Research and portfolio demonstration of next-action prediction for professional Dota 2 drafts. The behavior model imitates observed picks and bans. It does not establish optimal drafting or a causal increase in win probability.

## Dataset Snapshot

The recorded snapshot spans 2025-08-07 through 2026-08-23 UTC, with 26,852 normalized matches and 25,158 eligible matches. The profile excludes 1,694 matches for overlapping quality reasons, including invalid teams, incomplete drafts, and disagreement between picked heroes and final players.

| Split | Matches | Draft actions |
| --- | ---: | ---: |
| Train | 17,610 | 422,532 |
| Validation | 3,774 | 90,549 |
| Test | 3,774 | 90,559 |

Splitting is chronological at match level. Every sample contains only preceding draft actions plus the known current action type, side and order. The vocabulary is fitted on train; unseen patches and teams map to unknown embeddings.

Recorded training patches are 58 and 59. Patch 60 is absent from train and occurs in validation/test. Evaluation therefore includes temporal and version shift. API patch identifiers are retained without claiming a mapping to named game releases.

## Behavior Model

B+ concatenates four hero-presence vectors, context/team embeddings and two DeepSets lineup encodings, then ranks heroes through an MLP. Used heroes are masked. A 60-day half-life weights training examples relative to the newest training timestamp, not today's date. Validation is unweighted and does not update parameters.

Top-5 means the actual next hero appears among the five highest-ranked available heroes. It is not match win rate. The recorded seed-42 maximum is 31.59%, selected using validation. Many experiments reused this split; final generalization remains unmeasured for the selected neural model.

## Findings and Limitations

- Sequence Transformers did not show a clear consistent gain in the recorded runs.
- Multi-seed configurations have incomplete metadata. Physical batch size and accumulation differ or are undocumented; these are not perfectly controlled comparisons.
- Positions are heuristic assignments from post-match lane/resource data. Assignment scores are not calibrated label-confidence probabilities. Role accuracy measures agreement with those inferred labels.
- Outcome-weighted imitation cannot establish that recommended heroes increase win rate.
- The value model's best validation AUC is 0.5664 at epoch 1. Later loss increases indicate overfitting. Team/context-only ablation and probability calibration remain necessary.
- Value evaluation covers actual choices, not unobserved alternatives. Candidate re-ranking remains experimental.
- The old statistical baseline predates the current dataset. Do not subtract it from the current B+ result to advertise a lift.
- Historical statistical test results were exposed. Do not describe test data as never inspected.
- Actions within one match are correlated. Future confidence intervals should resample matches.

## Reproducibility and Distribution

The public report stores aggregate metrics, available configuration, and hashes of source summary files. These hashes do not fingerprint original training data or prove identical rows across runs. Exact dependencies and dataset hashes were not captured for historical runs.

Raw data and checkpoints remain local. Test fixtures are synthetic. No player account IDs or individual match records are included in the report. A code license and future data/weight distribution terms must be specified separately.
