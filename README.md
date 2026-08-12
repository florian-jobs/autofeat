# AutoFeat: Transitive Feature Discovery over Join Paths

Codebase for [AutoFeat: Transitive Feature Discovery over Join Paths](ICDE_FeatureDiscovery.pdf), reworked to run
without a live Neo4j instance. All graph lookups are now simulated in-process over a `join_paths.csv` DataFrame —
see `src/feature_discovery/graph_processing/neo4j_transactions.py`. Neo4j is only still needed by the legacy
thesis-reproduction CLI (`src/feature_discovery/cli.py`, `run.py`), not by the baseline adapter (`baseline.py`).

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

No Neo4j, no Java/Valentine, and no data-ingestion step is required for any of the above. Everything past this
point is reference detail for the pieces above.

## Repository layout

| Path | Role |
| --- | --- |
| `baseline.py` | `AutoFeatBaseline` — the integration adapter. Duck-typed `run(config) -> polars.DataFrame`, same shape as the sibling `QCRBaseline`/`ARDABaseline` |
| `discover_join_paths.py` | Generates `join_paths.csv` for a base table with none yet — Valentine schema-matches it against up to `--limit` tables under `--corpora` (no Neo4j, no full-corpus scan) and stages the result for `test_baseline.py` |
| `setup_baseline_test_fixture.py` | Builds a local `tmp/queries` + `tmp/corpus` fixture from `data/benchmark/credit`, mimicking what a server supplies at request time |
| `test_baseline.py` | Runs `AutoFeatBaseline.run(config)` against a fake or real `queries`/`corpus` layout, with optional correctness checks |
| `src/feature_discovery/autofeat_pipeline/autofeat.py` | `AutoFeat` — BFS traversal over join paths (`streaming_feature_selection`), core ranking/state |
| `src/feature_discovery/autofeat_pipeline/join_path_utils.py` | Join-path name encoding/decoding, path length |
| `src/feature_discovery/experiments/evaluate_join_paths.py` | `resolve_path_list`/`build_hop_list`, `join_from_path`, `evaluate_paths` — turns a ranked join path into an actual joined DataFrame |
| `src/feature_discovery/graph_processing/neo4j_transactions.py` | Neo4j-free graph simulation over `join_paths_df`, plus the raw-CSV read cache |
| `src/feature_discovery/experiments/ablation.py` | Thesis ablation study entry point; `--dataset` selects which `data/benchmark/<name>` to run |
| `build_benchmark_dataset.py` | Splits a single wide CSV (e.g. an OpenML table) into a snowflake-schema benchmark dataset under `data/benchmark/<name>/`, registers it in `datasets.csv` |
| `summarize_results.py` | Collapses `results/thesis/*.csv` down to one best-accuracy row per dataset+algorithm |
| `run_all_ablation.sh` | Runs `ablation.py` over every dataset under `data/benchmark/`, then `summarize_results.py` |

## The `baseline.py` contract

`AutoFeatBaseline.run(config)` expects a duck-typed config object (the eventual `beluga.config.schema.Config`):

| Attribute | Required | Meaning |
| --- | --- | --- |
| `base_table` | yes | Table id, e.g. `"credit"` (directory name under `queries_dir`, no `.csv`) |
| `queries_dir` | yes | Directory containing `<base_table>/` with the base table CSV and `join_paths.csv` |
| `data_dir`, `corpus` | yes | Location of the external lake corpus (`data_dir/corpus/*.csv`) |
| `target_column_id` | no | Column index of the label; defaults to the last column |
| `downstream_task` | no | `"classification"` or `"regression"` (default `"classification"`); anything else raises |
| `connections_csv_path` | no | Override for the join-paths CSV path (defaults to `queries_dir/base_table/join_paths.csv`) |
| `base_table_sep` | no | Base table's own CSV separator (default `,`); lake corpus is always read as `,`, matching ARDA |

`join_paths.csv` is always supplied externally (precomputed via Blend) — `baseline.py` has no in-process discovery
fallback. If you don't have one yet for a given base table, use `discover_join_paths.py` to generate one.

`run(config)` does, in order: read the base table → validate `target_column_id`/`downstream_task` → materialize
the base table into the lake folder as `<base_table>.csv` → run `AutoFeat.streaming_feature_selection` (BFS over
`join_paths.csv`, scoring every reachable join path) → pick the best-ranked path (base table itself if that wins,
i.e. no join) → materialize the winning join and return it as a `polars.DataFrame`.

## Generating `join_paths.csv` with `discover_join_paths.py`

For a base table with no precomputed `join_paths.csv`, Valentine schema-matches it against up to `--limit`
candidate tables under `--corpora` (bounding cost up front, unlike `test_baseline.py --limit`, which only bounds
traversal *after* `join_paths.csv` exists) and writes a `join_paths.csv` + staged corpus `test_baseline.py` can
run against directly:

```bash
uv run python discover_join_paths.py \
    --corpora ~/data/corpora/open_data/joinable_tables/ --limit 5 \
    --input ~/data/corpora/open_data/joinable_tables/nyc/nyc-finance-39g5-gbp3/table.csv \
    --query_column agency_name --target_column total_current_budget_amount
```

It searches candidates under `--input`'s own parent directory first, then expands to the rest of `--corpora` if
`--limit` isn't filled. Matches on `--query_column` above `--threshold` (default `0.55`) are kept; the command
prints the exact `test_baseline.py` invocation to run against the result, or (if nothing clears `--threshold`)
`No matches found above threshold` — rerun with `--verbose` to see each candidate's best similarity regardless of
threshold. Only `--sample-rows` rows (default `5000`) are read for matching; the full files are copied/joined
once a table is actually selected. Table/column ids containing dashes (e.g. `nyc-finance-39g5-gbp3`) are handled
correctly — see the join-path name encoding note below.

Join paths are internally named by chaining hops together (`join_path_utils.compute_join_name`), using
non-printable separators (`\x1e`/`\x1f`) rather than `"-"`, since real corpus ids often already contain dashes.

## Reproducing the paper's numbers with `ablation.py`

`ablation.py` runs the AutoFeat pipeline against the paper's own benchmark datasets, for comparing against
`docs/assets/papers/ICDE_FeatureDiscovery.pdf` (Table II / Figure 4-7). `data/benchmark/datasets.csv` lists the
paper's 8 datasets; `data/benchmark/` itself is gitignored, so which ones have a runnable corpus locally
(`table_0_0.csv` + `connections.csv`) varies — `school` is commonly missing, the other 7 typically present.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dataset` | `credit` | `base_table_label` in `data/benchmark/datasets.csv` |
| `--value-ratio` | `0.65` | Join-quality pruning threshold (τ in the paper) |
| `--top-k` | `15` | Max features retained per table (κ in the paper) |
| `--algorithm` | `LR` | One of `LR`, `RF`, `GBM`, `XT`, `XGB`, `KNN` |
| `--sample-size` | `3000` | Rows sampled for BFS relevance/redundancy scoring. Only affects ranking, not final training. Worth raising for very large base tables (e.g. `covertype`, 423K rows) |

Each run writes `results/thesis/<dataset>_AutoFeat.csv`, one row per evaluated join path. `GBM`/`XGB` match the
paper's tree-based headline numbers (Figure 4); `LR` matches the "Linear" chart (Figure 5). `summarize_results.py`
collapses these to one best-accuracy row per dataset+algorithm — compare against the green `AutoFeat` bars in
Figure 4/5, not Table II's "Best accuracy (OpenML.org)" column (that's the best anyone's ever reported on OpenML's
leaderboard, not what AutoFeat achieves).

`./run_all_ablation.sh [algorithm] [top_k]` runs every dataset with a local corpus and summarizes them in one go.

Building a new benchmark dataset from scratch (no ground-truth join graph, since you define the schema yourself):

```bash
uv run python build_benchmark_dataset.py \
    --input path/to/openml_dataset.csv --target-column <label_column> --name mydataset --num-tables 6 --max-depth 2
uv run python src/feature_discovery/experiments/ablation.py --dataset mydataset
```

**Known-fixed bugs** (results generated before these fixes will be lower, less deep, and possibly mislabeled by
algorithm):
- BFS used to stop after exactly one hop from the base table regardless of `--top-k` (a recursion guard in
  `autofeat.py` treated "seen as a neighbour" as "fully explored"). Now correctly continues into deeper hops.
- `--algorithm` was silently ignored — every run trained `LR`, and `GBM`/`XT`/`XGB`/`KNN` raised `BadParameter` if
  requested directly. Now all six genuinely run (`GBM`/`XGB` need `lightgbm`/`xgboost`, pulled in by `uv sync`;
  `xgboost` is pinned `<1.8` for `autogluon.tabular==0.7.0` compatibility).
- Stratified sampling never triggered for classification (`autofeat.py` compared the task against the literal
  string `"classification"`, but the actual value is `"binary"`/`"regression"`). Only matters once row count
  exceeds `sample_size` (default 3000) — affects `covertype`/`miniboone`/`jannis`/`bioresponse`, not `credit`.
- Runs weren't reproducible: `PYTHONHASHSEED` was set from inside the already-running process, which CPython
  ignores (only read at interpreter startup). Since BFS iterates Python `set()`s whose order depends on string
  hashing, and every edge has `weight=1` (so ties lean entirely on that order), each invocation could rank a
  different join path "best." `ablation.py` now re-execs itself via `subprocess.run` with `PYTHONHASHSEED` set in
  the real environment first.

**Not a bug, but worth documenting:** BFS joins far more tables than the paper reports for some datasets (e.g.
`bioresponse`: ~37-41 tables vs. the paper's 1). `streaming_feature_selection`'s per-sibling loop reads and writes
one shared `previous_queue`, so each sibling neighbour chains onto the *previous sibling's* result instead of
branching independently — collapsing what looks like a tree into one long sequential path. This looks like a bug
(a local fix snapshotting `previous_queue` per sibling shrank `bioresponse` from 37 to 4 tables), but it's
**not**: byte-for-byte diffed against live upstream `delftdata/autofeat` (`autofeat.py`, `evaluate_join_paths.py`,
`join_path_feature_selection.py`), including the ranking sort key and the `top_k=15`/`value_ratio=0.65` defaults
the paper's own authors call `ablation.py` with — all identical or behaviorally equivalent. The local "fix" was
reverted.

Why it produces such deep chains: each table added to the chain gets its own rank score based only on the
features it adds, never normalized by chain depth — so a long chain isn't penalized, and if each new table keeps
contributing decent features, rank keeps climbing the deeper it goes. Empirically (BFS-only run, `ranking`
inspected directly, no training): `bioresponse` (~41 tables) produced exactly 41 candidates at depths 0–40,
`corr(rank_score, depth) = 0.26`; `jannis` (13 tables) produced exactly 13 candidates at depths 0–12,
`corr = 0.37` — weakly positive, confirming depth is never penalized. A depth-37 `bioresponse` candidate ranked
3rd overall. For `jannis`, `top_k=15` exceeds the total candidate count (13), so *every* candidate — including the
full-corpus chain — gets trained regardless of rank, and more joined columns generally helps a linear model's raw
accuracy on a fixed split, so the deepest chain tends to win on both rank and accuracy. Which tables end up
chained together (and in what order) depends on Python `set()` iteration order, i.e. the hash seed — a different
seed plausibly produces a different "winning" path, including a shallow one, as the paper reports. We fixed our
own runs to be internally reproducible (`PYTHONHASHSEED=42`), but have no way to reproduce whichever seed the
paper's authors' original run landed on, so this is plausibly just a different (equally valid) draw from the same
nondeterministic process, not a fixable discrepancy.

## Server-side testing (`test_baseline.py`)

Since there is no real `beluga` package in this checkout, testing the adapter means faking what the server would
hand it: a base table directory, a `join_paths.csv`, and a lake corpus directory. `test_baseline.py` builds
`config` as a plain `SimpleNamespace` — `baseline.py` never imports `beluga` directly (see the `try/except
ImportError` in `_read_base_table`), so this is a faithful stand-in for how the real harness calls it.

```bash
uv run python setup_baseline_test_fixture.py   # materializes tmp/queries/credit and tmp/corpus/credit_lake
uv run python test_baseline.py                 # runs AutoFeatBaseline.run(config) against that fixture
```

To test against a real server layout (an actual `corpora/`/`queries/` tree), just point the flags below at it —
no fixture-building step required. A successful run prints the augmented `polars.DataFrame` and exits 0; a base
table that outranks every join is returned unchanged (no `_lake.`-prefixed columns).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--queries-dir` | `tmp/queries` | Contains `<base_table>/` with the base table CSV + `join_paths.csv` |
| `--data-dir` | `tmp/corpus` | Directory containing the lake corpus |
| `--corpus` | `credit_lake` | Corpus subfolder name under `--data-dir` |
| `--base-table` | `credit` | Base table id |
| `--target-column-id` | `0` | Target column index |
| `--downstream-task` | `classification` | `classification` or `regression` |
| `--limit` | none (full graph) | Max lake tables to traverse from the base table, BFS-bounded over `join_paths.csv`. Use against a large corpus (e.g. 60GB) to sample a bounded subgraph |
| `--value-ratio` | `0.65` | Pruning threshold passed through to `AutoFeatBaseline` |
| `--top-k` | `15` | Max ranked join-path candidates kept (κ in the paper) |
| `--sample-size` | `3000` | Rows sampled for BFS scoring — raise for a large real-world base table |
| `--verbose` | off | Prints BFS candidate count, chosen path's rank vs. top-5 runner-ups, and per-phase timing |
| `--min-rows` | none | Exit non-zero if the result has fewer rows than this — catches a join silently dropping most of the base table |
| `--max-null-ratio` | none | Exit non-zero if any joined-in column's null ratio (0-1) exceeds this — catches a join that "succeeded" but only matched empty cells |

`--verbose`/`--min-rows`/`--max-null-ratio` print a row/column-count and per-column null-coverage report, e.g.:

```
[AutoFeatBaseline] join graph: 7 edges, 6 tables reachable
[AutoFeatBaseline] BFS discovered 6 candidate join paths in 0.8s
[AutoFeatBaseline] chosen path: rank=0.9815, tables_joined=5 (#1 of 6 by rank)
[check] 1000 rows, 13 columns (9 joined in from lake tables)
[check] highest-null joined columns:
[check]     0.0%  table_1_1.csv.other_parties
```

This is what makes a huge, unfamiliar corpus (e.g. the full webtables/open-data lake) verifiable without staring
at the final `DataFrame`: did the join pull in real data, or match on IDs that mostly don't overlap and pad with
nulls? A good combo: `--verbose --max-null-ratio 0.9 --min-rows <90% of base table row count>`.

`--limit` walks `join_paths.csv` from the base table outward (BFS) and writes a temporary join-paths file
restricted to the first `limit` tables reached, passed to `baseline.py` via `connections_csv_path` — no change to
`baseline.py` itself. It bounds *which* lake tables can be read, not how much of any single table is read.

A failed run raises `ValueError` instead of printing a DataFrame:

| Message contains | Meaning |
| --- | --- |
| `"must be set"` | A required config field (`base_table`, `queries_dir`, `data_dir`) was missing |
| `"downstream_task ... not supported"` | `--downstream-task` was neither `classification` nor `regression` |
| `"Target column ... not numeric"` | `--downstream-task regression` but the target column isn't numeric |
| `"Cannot resolve base table file in ..."` | `queries_dir/<base_table>/` has zero or more than one CSV (besides `join_paths.csv`) |
| `"Join failed for path"` | The best-ranked join path couldn't actually be materialized (e.g. a stale/mismatched `join_paths.csv`) |

### Example: end-to-end against real server data

Starting from a base table with no `join_paths.csv` yet, against a 60GB NYC open-data corpus:

```bash
# 1. Discover a join_paths.csv (small --limit first — the corpus is huge)
uv run python discover_join_paths.py \
    --corpora ~/data/corpora/open_data/joinable_tables/ --limit 5 \
    --input ~/data/corpora/open_data/joinable_tables/nyc/nyc-finance-39g5-gbp3/table.csv \
    --query_column agency_name --target_column total_current_budget_amount

# 2. Run test_baseline.py with the command discover_join_paths.py printed
#    (total_current_budget_amount is a dollar amount -> --downstream-task regression)
uv run python test_baseline.py \
    --queries-dir tmp/queries --data-dir tmp/corpus --corpus nyc-finance-39g5-gbp3_lake \
    --base-table nyc-finance-39g5-gbp3 --target-column-id 12 --downstream-task regression
```

Two things worth checking on a real run like this before trusting the result:
- **Match quality** — `--threshold 0.55` gates weak matches, but similarity alone doesn't guarantee the columns
  are *semantically* joinable. Spot-check a joined column's values against the base table.
- **`--downstream-task`** — defaults to `classification`; pick `regression` explicitly for a continuous target.
