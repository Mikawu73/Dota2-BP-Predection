# Inference and Local Demo

## Run

From the repository root, after `python -m pip install -e .`:

```bash
python -m scripts.predict_draft --input examples/draft_state.json
python -m scripts.serve_demo --port 8765
```

The CLI reads JSON from stdin when `--input` is omitted. Both commands accept
`--checkpoint PATH`; the default is the existing `decay_current_60/model_b_best.pt`
under `data/processed/neural`. Use checkpoints you trust. Loading uses
`torch.load(weights_only=True)`, CPU execution, strict state-dict validation,
evaluation mode, and inference mode. No optimizer or data pipeline is run.

Open the printed loopback URL. Stop a foreground server with Ctrl+C. If the
port is occupied, choose another with `--port 8766`; `--port 0` chooses a free port.
Do not expose this development server to the internet.

The root URL is the data and experiment overview. `/sandbox` is the interactive
BP page, linked from the homepage. `/api/report` serves only the configured
public report (`reports/results.json` by default, override with `--report PATH`).
The homepage reads this snapshot, not raw match files or live training outputs;
its reference metrics need not describe a custom checkpoint loaded into the
sandbox. Installed wheel users must provide the public report separately.

The overview contains chronological split counts, recency comparisons, a
filterable experiment table with configuration details, repeat-seed means,
the separate factual outcome-model results, and provenance/limitations.
All behavioral metrics are explicitly validation metrics. Statistical baselines
from a different snapshot remain excluded. Missing reports return HTTP 503;
this does not disable sandbox inference. Downloading the JSON exposes only the
same aggregate report already used for the public project documentation.

## Input Contract

Use `examples/draft_state.json` as a minimal example. The object has:

| Field | Meaning |
| --- | --- |
| `patch_id` | Positive integer OpenDota patch ID, not a named game version |
| `radiant_team_id`, `dire_team_id` | Positive integer IDs or null/omitted |
| `acting_side` | `radiant` or `dire`, for the action being predicted |
| `is_pick` | JSON boolean; true = pick, false = ban |
| `history` | Ordered preceding actions, each with `hero_id`, `side`, `is_pick` |
| `top_k` | Optional integer, 1 to 20; default 5 |

The action order is derived from history length. Do not include the target hero,
outcome, future actions, or post-match statistics. Extra fields are rejected.
Hero IDs must belong to the checkpoint vocabulary. Duplicate heroes and more
than five picks on a side are rejected. A complete history returns an empty
ranking with `status: complete`; a history beyond the model's action limit fails.

The UI does not enforce Captain's Mode phases or automatically switch sides:
the next side and action are explicit controls. Invalid official draft schedules
can therefore be explored, but their scores may be out of distribution. Hero
availability by patch is not enforced beyond the training vocabulary. The sample
button and bundled JSON are synthetic examples, not benchmark matches.

Team fields accept a searchable name, a `Name (#ID)` catalog entry, or a numeric
ID. Duplicate names must be disambiguated by ID. Imports, exports and model
features remain numeric. Clearing a team field uses the unknown-team embedding.
Teams absent from the checkpoint vocabulary are labelled as unseen in training.

`src/web/teams.json` contains display-only names extracted from downloaded match
details, not a live roster directory. It uses the latest observed name per ID
and retains earlier names as searchable aliases. These names may be historical;
they do not identify current players. Refresh after downloading matches with:

```bash
python -m scripts.build_team_catalog
```

Restart the demo server after rebuilding the catalog. This does not change
normalized tables, training vocabularies, features or checkpoint weights.

## Outputs and Interpretation

The output includes hero IDs, sorted probabilities, context warnings, action
order, candidate count, and the checkpoint SHA-256. The softmax denominator
contains all available vocabulary heroes, not just the returned top five.
Used heroes are excluded. Ties use vocabulary order. Scores are uncalibrated
likelihoods of historical behavior, not predicted win rates or recommendations
proven to improve wins.

Unseen patches and teams map to the same zero-index embeddings used by training
and validation. No nearest-patch substitution occurs. Changing a patch between
two unseen IDs therefore does not add patch-specific knowledge. Missing team IDs
remove learned team-specific context. A hero not in train cannot be scored.

The inference feature tensors are checked against `DraftDataset` using synthetic
fixtures, including relative team order when Dire acts. Local CLI and HTTP tests
do not read or evaluate the held-out test dataset.

## HTTP and Distribution

- `GET /api/metadata`: architecture, vocabulary, hero display data and checkpoint hash.
- `POST /api/predict`: the input JSON contract above; returns the same ranking as the CLI.
- Invalid input returns 400; unsupported content type returns 415; oversized input returns 413.
- Request bodies are capped at 16 KB. Host and Origin checks restrict local requests.
- Static file paths are allowlisted; raw data, checkpoints and `.env` are never served.
- Model execution is serialized across HTTP threads; the model is loaded once on CPU.

The site uses plain HTML/CSS/JavaScript and a Python standard-library HTTP server.
It cannot be deployed as a functional static-only GitHub Pages site. Public
hosting needs a separate production Python service, request/concurrency limits,
appropriate origin configuration and deployment secrets handling. Keep the
local server's restrictions intact; use a production server adapter instead.

No hosted demo or downloadable weights are included in this change. Repository
tests use tiny synthetic checkpoints. Package builds include the page assets,
but not private training data or weights. Hero portraits load directly from
Valve's CDN; names and rankings still work if images cannot load. See
`src/web/ASSETS.md` for asset provenance and the preserved Lucide license.
