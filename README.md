# AutoFeat: Transitive Feature Discovery over Join Paths

Codebase for [AutoFeat: Transitive Feature Discovery over Join Paths](ICDE_FeatureDiscovery.pdf), reworked to run
without a live Neo4j instance. All graph lookups (`get_adjacent_nodes`, `get_node_by_id`,
`get_relation_properties_node_name`, ...) are now simulated in-process over a `join_paths.csv` DataFrame — see
`src/feature_discovery/graph_processing/neo4j_transactions.py`. Neo4j is only still needed by the legacy
thesis-reproduction CLI (`src/feature_discovery/cli.py`, `run.py`), not by the baseline adapter below.

## Repository layout

| Path | Role |
| --- | --- |
| `baseline.py` | `AutoFeatBaseline` — the integration adapter. Duck-typed `run(config) -> polars.DataFrame`, same shape as the sibling `QCRBaseline`/`ARDABaseline` |
| `discover_join_paths.py` | Generates `join_paths.csv` for a base table with none yet — Valentine schema-matches it against up to `--limit` tables under `--corpora` (no Neo4j, no full-corpus scan) and stages the result for `test_baseline.py` |
| `setup_baseline_test_fixture.py` | Builds a local `tmp/queries` + `tmp/corpus` fixture from `data/benchmark/credit`, mimicking what a server supplies at request time |
| `test_baseline.py` | Minimal smoke test: builds a fake `config` (`SimpleNamespace`) and calls `AutoFeatBaseline.run(config)` |
| `src/feature_discovery/autofeat_pipeline/autofeat.py` | `AutoFeat` — BFS traversal over join paths (`streaming_feature_selection`), core ranking/state (`ranking`, `partial_join_selected_features`) |
| `src/feature_discovery/autofeat_pipeline/join_path_utils.py` | Join-path name encoding/decoding, path length |
| `src/feature_discovery/experiments/evaluate_join_paths.py` | `resolve_path_list`/`build_hop_list`, `join_from_path`, `evaluate_paths` — turns a ranked join path into an actual joined DataFrame |
| `src/feature_discovery/graph_processing/neo4j_transactions.py` | Neo4j-free graph simulation over `join_paths_df`, plus the raw-CSV read cache (`get_df_with_prefix`) |
| `src/feature_discovery/experiments/ablation.py` | Thesis ablation study entry point, exercises the same pipeline end-to-end |

## Setup

```bash
pip install -e .
```

No Neo4j, no Java/Valentine, no data ingestion step is required to run or test `baseline.py`.

## The `baseline.py` contract

`AutoFeatBaseline.run(config)` expects a duck-typed config object (the eventual
`beluga.config.schema.Config`) with:

| Attribute | Required | Meaning |
| --- | --- | --- |
| `base_table` | yes | Table id, e.g. `"credit"` (directory name under `queries_dir`, no `.csv`) |
| `queries_dir` | yes | Directory containing `<base_table>/` with the base table CSV and `join_paths.csv` |
| `data_dir`, `corpus` | yes | Location of the external lake corpus (`data_dir/corpus/*.csv`) |
| `target_column_id` | no | Column index of the label; defaults to the last column |
| `downstream_task` | no | `"classification"` or `"regression"` (default `"classification"`); anything else raises |
| `connections_csv_path` | no | Override for the join-paths CSV path (defaults to `queries_dir/base_table/join_paths.csv`) |
| `base_table_sep` | no | Base table's own CSV separator (default `,`); the lake corpus is always read as `,`, matching the ARDA baseline |

`join_paths.csv` is always supplied externally (precomputed via Blend) — `baseline.py` has no in-process
discovery fallback. If you don't have one yet for a given base table, use `discover_join_paths.py` (below) to
generate one.

## Generating `join_paths.csv` with `discover_join_paths.py`

For a base table with no precomputed `join_paths.csv`, this script Valentine schema-matches it against up to
`--limit` candidate tables under `--corpora` (bounding the cost up front, unlike `test_baseline.py --limit` which
only bounds traversal *after* `join_paths.csv` exists) and writes a `join_paths.csv` + staged corpus that
`test_baseline.py` can run against directly — no Neo4j involved.

```bash
python discover_join_paths.py \
    --corpora ~/data/corpora/open_data/joinable_tables/ --limit 5 \
    --input ~/data/corpora/open_data/joinable_tables/nyc/nyc-finance-39g5-gbp3/table.csv \
    --query_column agency_name --target_column total_current_budget_amount
```

It walks `--corpora` recursively for `*.csv` files (up to `--limit`), uses each candidate's parent folder name as
its table id (matching layouts like `.../nyc-finance-39g5-gbp3/table.csv`), keeps Valentine matches on
`--query_column` above `--threshold` (default `0.55`), and prints the exact `test_baseline.py` command to run
against the result. Table/column ids containing dashes (e.g. `nyc-finance-39g5-gbp3`) are handled correctly —
see the join-path name encoding note below.

## How `baseline.py` works

`AutoFeatBaseline.run(config)` does, in order:

1. **Read the base table** (`_read_base_table`) — via `beluga.online.base_table.read_base_table` if importable,
   else a local CSV fallback (dev/testing only).
2. **Validate** `target_column_id`/`downstream_task`, and that a regression target is numeric.
3. **Materialize the base table into the lake folder** as `<base_table>.csv`, so it's addressable as just another
   node during traversal (`base_table_id = config.base_table + ".csv"`).
4. **Run `AutoFeat.streaming_feature_selection`** — BFS over `join_paths.csv` starting from the base table,
   scoring every reachable join path and caching each path's selected features in
   `partial_join_selected_features`.
5. **Pick the best-ranked path** (`bfs_traversal.ranking`, tie-broken by shortest path then name). If that's the
   base table itself, return it as-is — no join needed.
6. **Materialize the winning join** (`resolve_path_list` + `join_from_path`) and return the base table joined
   with the selected features from that path as a `polars.DataFrame`.

Join paths are internally named by chaining hops together (`join_path_utils.compute_join_name`). Table/column ids
from real corpora often contain dashes (Socrata-style ids like `nyc-finance-39g5-gbp3`), so the encoding uses
non-printable separators (`\x1e` between hops, `\x1f` between fields) rather than `"-"`/`"--"`, which would
otherwise misparse such ids.

## Server-side testing

Since there is no real `beluga` package in this checkout yet, testing the adapter means faking what the server
would hand it: a base table directory, a `join_paths.csv`, and a lake corpus directory.

```bash
python setup_baseline_test_fixture.py   # materializes tmp/queries/credit and tmp/corpus/credit_lake
python test_baseline.py                 # runs AutoFeatBaseline.run(config) against that fixture
```

`test_baseline.py` builds `config` as a plain `SimpleNamespace` rather than importing `beluga.config.schema.Config`
— `baseline.py` never imports `beluga` directly (see the `try/except ImportError` in `_read_base_table`), so this
is a faithful stand-in for how the real harness will call it. A successful run prints the augmented `polars.DataFrame`
(base table columns + selected joined features).

None of `test_baseline.py`'s paths are hardcoded — they're CLI flags, falling back to the local fixture above:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--queries-dir` | `tmp/queries` | Queries directory (contains `<base_table>/` with the base table CSV + `join_paths.csv`) |
| `--data-dir` | `tmp/corpus` | Directory containing the lake corpus |
| `--corpus` | `credit_lake` | Corpus subfolder name under `--data-dir` |
| `--base-table` | `credit` | Base table id |
| `--target-column-id` | `0` | Target column index |
| `--downstream-task` | `classification` | `classification` or `regression` |
| `--limit` | none (full graph) | Max number of lake tables to traverse from the base table, BFS-bounded over `join_paths.csv`. Use this against a large corpus (e.g. 60GB) to sample a bounded subgraph instead of scanning it all |

To test against a real server layout (an actual `corpora/`/`queries/` directory tree), just point these at it —
no code or fixture-building step required.

`--limit` works by walking `join_paths.csv` from the base table outward (BFS over `from_id`/`to_id`) and writing
a temporary join-paths file restricted to the first `limit` tables reached, passed to `baseline.py` via its
existing `connections_csv_path` override — no change to `baseline.py` itself. It bounds *which* lake tables can be
read, not how much of any single table is read.

### Examples

```bash
# 1. Local fixture, defaults (classification on the credit dataset)
python setup_baseline_test_fixture.py
python test_baseline.py

# 2. Local fixture, explicit flags (equivalent to the defaults above)
python test_baseline.py \
    --queries-dir tmp/queries --data-dir tmp/corpus --corpus credit_lake \
    --base-table credit --target-column-id 0 --downstream-task classification

# 3. Regression task, numeric target in a different column
# (requires a fixture for that base table first — setup_baseline_test_fixture.py only
#  builds the "credit" one; point --queries-dir/--data-dir at your own tmp/queries/<table>
#  + tmp/corpus/<table>_lake, or adapt setup_baseline_test_fixture.py's SOURCE)
python test_baseline.py --base-table steel --target-column-id 3 --downstream-task regression

# 4. Real server layout, unbounded (small/known corpus)
python test_baseline.py \
    --queries-dir /srv/queries --data-dir /srv/corpora --corpus my_lake --base-table my_table

# 5. Real server layout, bounded traversal (large corpus, e.g. 60GB)
python test_baseline.py \
    --queries-dir /srv/queries --data-dir /srv/corpora --corpus my_lake --base-table my_table \
    --limit 20
```

A successful run prints the augmented `polars.DataFrame` and exits 0; a failure raises a `ValueError` naming the
missing config field, unresolved base table, or failed join.
