# AutoFeat: Transitive Feature Discovery over Join Paths

Codebase for [AutoFeat: Transitive Feature Discovery over Join Paths](ICDE_FeatureDiscovery.pdf), reworked to run
without a live Neo4j instance. Neo4j is only needed by the legacy CLI (`cli.py`, `run.py`), not by the integration
adapter (`baseline.py`).

## Quick start (server)

```bash
cd ~/data/baselines/autofeat && git pull && uv sync
uv run python src/feature_discovery/experiments/ablation.py --dataset credit
uv run python summarize_results.py
# or every locally-present dataset: ./run_all_ablation.sh [algorithm] [top_k]
```

Expected: `credit` reaches 6 tables, ~**0.735 accuracy** (paper: ~0.75, "Linear", Fig. 5). Stuck at 2-3 tables /
~0.65? Stale BFS fix — check `git log -1 -- src/feature_discovery/autofeat_pipeline/autofeat.py`.

## Server-side testing (`test_baseline.py`)

No real `beluga` package in this checkout — `test_baseline.py` fakes `config` as a `SimpleNamespace`; `baseline.py`
handles that identically (see `try/except ImportError` in `_read_base_table`).

```bash
uv run python setup_baseline_test_fixture.py   # materializes tmp/queries/credit, tmp/corpus/credit_lake
uv run python test_baseline.py --verbose --min-rows 500 --max-null-ratio 0.9
```

Point `--queries-dir`/`--data-dir`/`--corpus` at a real `corpora/`/`queries/` tree to test against actual server
data (no fixture step needed). Success prints the augmented `polars.DataFrame` and exits 0.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--queries-dir` / `--data-dir` | `tmp/queries` / `tmp/corpus` | Base table + `join_paths.csv` / lake corpus |
| `--corpus` / `--base-table` | `credit_lake` / `credit` | Corpus subfolder / base table id |
| `--target-column-id` | `0` | Target column index |
| `--downstream-task` | `classification` | or `regression` — pick explicitly for continuous targets |
| `--limit` | none | Max lake tables to traverse — bound a huge corpus |
| `--value-ratio` / `--top-k` / `--sample-size` | `0.65` / `15` / `3000` | Same meaning as in `ablation.py` |
| `--verbose` | off | BFS candidate count, chosen path's rank vs. top-5, per-phase timing |
| `--min-rows` / `--max-null-ratio` | none | Exit non-zero if the join dropped most rows / matched mostly empty cells |

No `join_paths.csv` for a base table yet? Generate one first:

```bash
uv run python discover_join_paths.py --corpora ~/data/corpora/open_data/joinable_tables/ --limit 5 \
    --input ~/data/corpora/open_data/joinable_tables/nyc/nyc-finance-39g5-gbp3/table.csv \
    --query_column agency_name --target_column total_current_budget_amount
# prints the exact test_baseline.py command to run against the result
```

A failed run raises `ValueError`: missing required config field, unsupported `downstream_task`, non-numeric
regression target, ambiguous base table file, or a stale/mismatched `join_paths.csv`.

## Errors & fixes

| Symptom | Cause | Status |
| --- | --- | --- |
| `OpenBLAS: Program is Terminated. ... too many memory regions` on hercules | Many-core server: `joblib`'s `Parallel(n_jobs=-1)` (schema matching) and AutoGluon's own worker processes each open an OpenBLAS thread pool sized to core count — total exceeds OpenBLAS's hardcoded 128-thread build limit | **Fixed** — `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS` pinned to 1 via `os.environ.setdefault(...)` before `numpy` import in `cli.py`, `ablation.py`, `dataset_discovery.py`. Override with e.g. `OPENBLAS_NUM_THREADS=4 uv run python ...` if 1 is too conservative |
| Results not reproducible run-to-run (same dataset, different tables/accuracy) | `os.environ['PYTHONHASHSEED']` set *inside* an already-running process is a no-op (CPython reads it once, at interpreter startup); BFS traversal uses `set()` iteration order, which depends on it | **Fixed** — `ablation.py` re-launches itself via `subprocess` with `PYTHONHASHSEED` set in the real environment first. Override with `PYTHONHASHSEED=<n>` to explore other seeds |
| `IndexError`/`ValueError` in `measure_redundancy` on a fresh dataset's first hop | Empty `selected_features` → `np.array([])` defaults to `float64`, which numpy refuses for fancy indexing; `apply_along_axis` also refuses a zero-width axis | **Fixed** (`21660ef`) — empty selections short-circuit to zero redundancy contribution |
| `./run_all_ablation.sh` crashed every dataset on `argparse` int conversion | `${1:15}` is bash substring expansion (not `${1:-15}` default-value syntax) and reused `$1` (the algorithm arg) for `top_k` too, always passing `--top-k ""` | **Fixed** (`17b01ee`) — `TOP_K="${2:-15}"`, algorithm and top_k are now independent positional args |
| BFS stopped after 1 hop regardless of `--top-k`; `--algorithm` silently ignored (always trained `LR`); stratified sampling never triggered for classification | Older `autofeat.py`/`ablation.py` bugs | **Fixed** — see `git log` on those files |
| `bioresponse` (and others) join far more tables than the paper reports (~40 vs. paper's 1) | **Not a bug.** Verified byte-for-byte against upstream `delftdata/autofeat`: ranking never penalizes join-path depth, and which tables get chained depends on Python's hash-seed-dependent `set()` iteration order — inherent to the published algorithm, not this port. Runs are internally reproducible (`PYTHONHASHSEED=42`) but land on a different, equally valid draw than the paper's original (undocumented) seed | Use `sweep_hashseed.py` to quantify the seed-to-seed variance instead of chasing a single "correct" seed |
| Confusion: is κ (`--top-k`) 5 or 15? | Paper text states **κ=15** explicitly ("maximum selected features from a table"); upstream's class constructor default of 5 is an unused Python fallback, not what the paper's reported experiments used | `ablation.py`'s CLI default is `15`, matching the paper |

## Repository layout

| Path | Role |
| --- | --- |
| `baseline.py` | `AutoFeatBaseline` — integration adapter, duck-typed `run(config) -> polars.DataFrame` |
| `discover_join_paths.py` | Generates `join_paths.csv` for a base table with none yet |
| `setup_baseline_test_fixture.py` / `test_baseline.py` | Local test fixture / server-side test runner (see above) |
| `src/feature_discovery/autofeat_pipeline/autofeat.py` | `AutoFeat` — BFS traversal over join paths, core ranking/state |
| `src/feature_discovery/experiments/ablation.py` | Thesis ablation entry point; `--dataset` selects `data/benchmark/<name>` |
| `build_benchmark_dataset.py` | Splits a wide CSV into a snowflake-schema benchmark dataset under `data/benchmark/<name>/` |
| `summarize_results.py` / `run_all_ablation.sh` | Collapse `results/thesis/*.csv` to best-per-dataset+algorithm / run every local dataset then summarize |
| `sweep_hashseed.py` | Quantifies BFS's `PYTHONHASHSEED` sensitivity across seeds (see Errors table) |

`ablation.py` flags: `--dataset` (`credit`), `--value-ratio` (τ, `0.65`), `--top-k` (κ, `15`), `--algorithm`
(`LR`/`RF`/`GBM`/`XT`/`XGB`/`KNN`), `--sample-size` (`3000`, raise for large base tables e.g. `covertype`).
Compare `summarize_results.py` output against Figure 4 (`GBM`/`XGB`) or 5 (`LR`) in the paper, not Table II's
"Best accuracy (OpenML.org)" column.
