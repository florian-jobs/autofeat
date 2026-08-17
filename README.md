# AutoFeat: Transitive Feature Discovery over Join Paths

Codebase for [AutoFeat: Transitive Feature Discovery over Join Paths](ICDE_FeatureDiscovery.pdf), reworked to run
without a live Neo4j instance — graph lookups are simulated in-process over a `join_paths.csv` DataFrame. Neo4j is
only needed by the legacy thesis-reproduction CLI (`cli.py`, `run.py`), not by the baseline adapter (`baseline.py`).

## Quick start (server)

```bash
cd ~/data/baselines/autofeat
git pull
uv sync   # picks up setuptools<81 (autogluon needs pkg_resources) and lightgbm/xgboost for GBM/XGB

# Reproduce a paper number (writes results/thesis/credit_AutoFeat.csv)
uv run python src/feature_discovery/experiments/ablation.py --dataset credit
uv run python summarize_results.py     # collapses to one best-accuracy row per dataset+algorithm

# Or run every locally-present dataset at once
./run_all_ablation.sh          # --algorithm LR (default), --top-k 15 (default)
./run_all_ablation.sh GBM 10   # algorithm + top_k are optional positional args

# Test the beluga integration adapter (baseline.py) against a fake server fixture
uv run python setup_baseline_test_fixture.py
uv run python test_baseline.py --verbose --min-rows 500 --max-null-ratio 0.9
```

Expected: `credit` reaches all 5 joinable tables and tops out around **0.735 accuracy** (paper reports ~0.75 for
"Linear", Figure 5). If your run instead stops at 2-3 tables / ~0.65 accuracy, `git pull` didn't pick up the BFS
fix — check `git log -1 -- src/feature_discovery/autofeat_pipeline/autofeat.py`.

No Neo4j, no Java/Valentine, and no data-ingestion step is required for any of the above.

## Repository layout

| Path | Role |
| --- | --- |
| `baseline.py` | `AutoFeatBaseline` — the integration adapter. Duck-typed `run(config) -> polars.DataFrame` |
| `discover_join_paths.py` | Generates `join_paths.csv` for a base table with none yet (Valentine schema-matching over a corpus) |
| `setup_baseline_test_fixture.py` | Builds a local `tmp/queries` + `tmp/corpus` fixture from `data/benchmark/credit` |
| `test_baseline.py` | Runs `AutoFeatBaseline.run(config)` against a fake or real `queries`/`corpus` layout, with optional correctness checks |
| `src/feature_discovery/autofeat_pipeline/autofeat.py` | `AutoFeat` — BFS traversal over join paths, core ranking/state |
| `src/feature_discovery/experiments/ablation.py` | Thesis ablation study entry point; `--dataset` selects `data/benchmark/<name>` |
| `build_benchmark_dataset.py` | Splits a wide CSV into a snowflake-schema benchmark dataset under `data/benchmark/<name>/` |
| `summarize_results.py` | Collapses `results/thesis/*.csv` down to one best-accuracy row per dataset+algorithm |
| `run_all_ablation.sh` | Runs `ablation.py` over every dataset under `data/benchmark/`, then `summarize_results.py` |

## The `baseline.py` contract

`AutoFeatBaseline.run(config)` expects a duck-typed config object (`base_table`, `queries_dir`, `data_dir`,
`corpus` required; `target_column_id`, `downstream_task`, `connections_csv_path`, `base_table_sep` optional —
see docstring in `baseline.py` for details). `join_paths.csv` is always supplied externally (precomputed via
Blend) — there's no in-process discovery fallback. Use `discover_join_paths.py` to generate one if missing.

`run(config)`: read base table → validate → materialize base table into lake folder → BFS-score every reachable
join path via `AutoFeat.streaming_feature_selection` → pick best-ranked path (or the base table itself, i.e. no
join) → materialize and return as `polars.DataFrame`.

## Generating `join_paths.csv`

```bash
uv run python discover_join_paths.py \
    --corpora ~/data/corpora/open_data/joinable_tables/ --limit 5 \
    --input ~/data/corpora/open_data/joinable_tables/nyc/nyc-finance-39g5-gbp3/table.csv \
    --query_column agency_name --target_column total_current_budget_amount
```

Matches on `--query_column` above `--threshold` (default `0.55`) are kept; prints the exact `test_baseline.py`
command to run against the result. Only `--sample-rows` rows (default `5000`) are read for matching.

## Reproducing the paper's numbers with `ablation.py`

Compares against `docs/assets/papers/ICDE_FeatureDiscovery.pdf` (Table II / Figure 4-7). `data/benchmark/` is
gitignored; which of the 8 paper datasets have a runnable corpus locally varies (`school` commonly missing).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dataset` | `credit` | `base_table_label` in `data/benchmark/datasets.csv` |
| `--value-ratio` | `0.65` | Join-quality pruning threshold (τ in the paper) |
| `--top-k` | `15` | Max features retained per table (κ in the paper) |
| `--algorithm` | `LR` | One of `LR`, `RF`, `GBM`, `XT`, `XGB`, `KNN` |
| `--sample-size` | `3000` | Rows sampled for BFS scoring — raise for very large base tables (e.g. `covertype`) |

Compare `summarize_results.py` output against the green `AutoFeat` bars in Figure 4 (`GBM`/`XGB`) or 5 (`LR`), not
Table II's "Best accuracy (OpenML.org)" column. `./run_all_ablation.sh [algorithm] [top_k]` runs every locally
available dataset and summarizes in one go.

New benchmark dataset from a raw CSV:

```bash
uv run python build_benchmark_dataset.py \
    --input path/to/openml_dataset.csv --target-column <label_column> --name mydataset --num-tables 6 --max-depth 2
uv run python src/feature_discovery/experiments/ablation.py --dataset mydataset
```

**Known-fixed bugs** (older results will be shallower/lower and possibly mislabeled by algorithm): BFS used to
stop after one hop regardless of `--top-k`; `--algorithm` was silently ignored (always trained `LR`); stratified
sampling never triggered for classification; runs weren't reproducible across invocations (`PYTHONHASHSEED` fix).
All fixed — see `git log` on `autofeat_pipeline/autofeat.py` / `experiments/ablation.py` for details.

**Not a bug:** BFS sometimes joins far more tables than the paper reports (e.g. `bioresponse`: ~40 vs. paper's 1).
Verified byte-for-byte against upstream `delftdata/autofeat` — ranking never penalizes chain depth, and which
tables end up chained depends on Python's hash-seed-dependent `set()` iteration order. Our runs are internally
reproducible (`PYTHONHASHSEED=42`) but land on a different (equally valid) draw than the paper's original seed.

## Server-side testing (`test_baseline.py`)

No real `beluga` package in this checkout — `test_baseline.py` fakes `config` as a `SimpleNamespace`, which
`baseline.py` handles the same way (see `try/except ImportError` in `_read_base_table`).

```bash
uv run python setup_baseline_test_fixture.py   # materializes tmp/queries/credit and tmp/corpus/credit_lake
uv run python test_baseline.py                 # runs AutoFeatBaseline.run(config) against that fixture
```

Point the flags below at a real `corpora/`/`queries/` tree to test against actual server data — no fixture step
needed. Success prints the augmented `polars.DataFrame` and exits 0.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--queries-dir` | `tmp/queries` | Contains `<base_table>/` with the base table CSV + `join_paths.csv` |
| `--data-dir` | `tmp/corpus` | Directory containing the lake corpus |
| `--corpus` | `credit_lake` | Corpus subfolder name under `--data-dir` |
| `--base-table` | `credit` | Base table id |
| `--target-column-id` | `0` | Target column index |
| `--downstream-task` | `classification` | `classification` or `regression` |
| `--limit` | none | Max lake tables to traverse (BFS-bounded) — use against a huge corpus to sample a bounded subgraph |
| `--value-ratio` / `--top-k` / `--sample-size` | `0.65` / `15` / `3000` | Same meaning as in `ablation.py` |
| `--verbose` | off | Prints BFS candidate count, chosen path's rank vs. top-5, per-phase timing |
| `--min-rows` | none | Exit non-zero if result has fewer rows — catches a join silently dropping most of the base table |
| `--max-null-ratio` | none | Exit non-zero if any joined-in column's null ratio exceeds this — catches a join matching mostly empty cells |

A failed run raises `ValueError` (missing required config field, unsupported `downstream_task`, non-numeric
regression target, ambiguous base table file, or a stale/mismatched `join_paths.csv`).

### Example: end-to-end against real server data

```bash
# 1. Discover a join_paths.csv (small --limit first on a huge corpus)
uv run python discover_join_paths.py \
    --corpora ~/data/corpora/open_data/joinable_tables/ --limit 5 \
    --input ~/data/corpora/open_data/joinable_tables/nyc/nyc-finance-39g5-gbp3/table.csv \
    --query_column agency_name --target_column total_current_budget_amount

# 2. Run test_baseline.py with the command discover_join_paths.py printed
uv run python test_baseline.py \
    --queries-dir tmp/queries --data-dir tmp/corpus --corpus nyc-finance-39g5-gbp3_lake \
    --base-table nyc-finance-39g5-gbp3 --target-column-id 12 --downstream-task regression
```

Check match quality (similarity ≠ semantic joinability, spot-check values) and `--downstream-task` (defaults to
`classification`; pick `regression` explicitly for continuous targets).
