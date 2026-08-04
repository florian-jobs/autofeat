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
| `src/feature_discovery/experiments/ablation.py` | Thesis ablation study entry point, exercises the same pipeline end-to-end; `--dataset` selects which `data/benchmark/<name>` to run |
| `build_benchmark_dataset.py` | Splits a single wide CSV (e.g. an OpenML table) into a snowflake-schema benchmark dataset under `data/benchmark/<name>/`, registers it in `datasets.csv` |

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
uv run python discover_join_paths.py \
    --corpora ~/data/corpora/open_data/joinable_tables/ --limit 5 \
    --input ~/data/corpora/open_data/joinable_tables/nyc/nyc-finance-39g5-gbp3/table.csv \
    --query_column agency_name --target_column total_current_budget_amount
```

Expected output on a successful match:

```
Matching .../table.csv ('agency_name') against 5 candidate tables under ... ...
  match: nyc-some-other-table.some_column (similarity=0.81)
  skip nyc-broken-table: <read error, if any>

2 join path(s) written to .../queries/nyc-finance-39g5-gbp3/join_paths.csv
Base table staged at: .../queries/nyc-finance-39g5-gbp3/table.csv
Matched lake tables staged at: .../corpus/nyc-finance-39g5-gbp3_lake

Run:
  python test_baseline.py --queries-dir ... --data-dir ... --corpus nyc-finance-39g5-gbp3_lake --base-table nyc-finance-39g5-gbp3 --target-column-id N
```

If no candidate clears `--threshold`, it instead prints `No matches found above threshold; no join_paths.csv written.` and exits without staging anything — rerun with `--verbose` (see below) to see why.

It searches candidates under `--input`'s own parent directory first (e.g. other tables under `.../nyc/` if
`--input` is `.../nyc/nyc-finance-39g5-gbp3/table.csv`) — far more likely to be genuinely joinable than an
arbitrary alphabetical slice of the whole corpus — and only expands to the rest of `--corpora` if `--limit`
isn't filled by nearby tables. It uses each candidate's parent folder name as its table id, keeps Valentine
matches on `--query_column` above `--threshold` (default `0.55`), and prints the exact `test_baseline.py`
command to run against the result. Table/column ids containing dashes (e.g. `nyc-finance-39g5-gbp3`) are
handled correctly — see the join-path name encoding note below.

If nothing matches, add `--verbose` to see each candidate's best similarity for `--query_column` even when it's
below `--threshold` — useful for telling apart "no real join partner in this sample" from "just needs a lower
`--threshold`".

Matching only reads `--sample-rows` rows (default `5000`) from the base table and each candidate — schema
matching needs a representative slice of values, not the whole file, and corpus tables can be multi-million-row.
The full files are still copied/joined once a table is actually selected; only the discovery step is sampled.

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

## Reproducing the paper's numbers with `ablation.py`

Separate from `baseline.py` (the integration adapter), `src/feature_discovery/experiments/ablation.py` runs the
same AutoFeat pipeline against the paper's own benchmark datasets, for comparing reproduced accuracy against the
numbers in `docs/assets/papers/ICDE_FeatureDiscovery.pdf` (Table II / Figure 4-7). `data/benchmark/datasets.csv`
lists the paper's 8 datasets (`credit`, `steel`, `jannis`, `miniboone`, `eyemove`, `bioresponse`, `school`,
`covertype`); only `credit` ships with this checkout.

```bash
uv run python src/feature_discovery/experiments/ablation.py --dataset credit
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dataset` | `credit` | `base_table_label` in `data/benchmark/datasets.csv` |
| `--value-ratio` | `0.65` | Join-quality pruning threshold (τ in the paper) |
| `--top-k` | `15` | Max features retained per table (κ in the paper) |
| `--algorithm` | `LR` | Model passed to `evaluate_all_algorithms` |

Each run writes `results/thesis/<dataset>_AutoFeat.csv` — one row per evaluated join path, with `Result` dataclass
fields (`accuracy`, `train_time`, `feature_selection_time`, `rank`, `join_path_features`, ...). The paper's own
defaults are `τ=0.65, κ=15` (Section VII-B), matching the flags' defaults above.

**Known-fixed bug:** the BFS traversal used to stop after exactly one hop from the base table on every run,
regardless of `--top-k` (`autofeat.py`'s recursion guard treated "already seen as a neighbour" as "already fully
explored," which is trivially always true one level in). This has been fixed — traversal now correctly continues
into deeper hops. If you're comparing against results generated before this fix, expect them to be lower and
less deep than what you get now.

Building a new benchmark dataset from scratch — e.g. an OpenML table the paper didn't cover — is a different
scenario (no ground-truth join graph to match, since you're defining the schema yourself):

```bash
uv run python build_benchmark_dataset.py \
    --input path/to/openml_dataset.csv --target-column <label_column> \
    --name mydataset --num-tables 6 --max-depth 2
uv run python src/feature_discovery/experiments/ablation.py --dataset mydataset
```

### Example: reproducing `credit` on the server

```bash
cd ~/data/baselines/autofeat
git pull
uv sync   # picks up the setuptools<81 pin (autogluon needs pkg_resources, removed in setuptools>=81)
uv run python src/feature_discovery/experiments/ablation.py --dataset credit
```

Expected result — `results/thesis/credit_AutoFeat.csv` gets one row per evaluated join path, reaching all 5
joinable tables (not just the two directly connected to the base table):

| Path | Tables joined | Accuracy |
| --- | --- | --- |
| `credit → table_1_1` | 2 | 0.68 |
| `credit → table_1_1 + table_1_2` | 3 | 0.65 |
| `credit → table_1_1 + table_1_2 → table_2_5` | 4 | 0.695 |
| `credit → table_1_1 (+table_2_3) + table_1_2 (+table_2_5)` | 5 | 0.715 |
| `credit → table_1_1 (+table_2_3, +table_2_4) + table_1_2 (+table_2_5)` (full depth-3 join) | 6 | **0.735** |

Best accuracy (0.735) is close to the paper's reported ~0.75 for `credit` on the "Linear" model (Figure 5). If
your run instead tops out at accuracy 0.65 with only 2-3 tables joined, the BFS fix hasn't landed — re-check
`git log -1 -- src/feature_discovery/autofeat_pipeline/autofeat.py` and rerun `git pull`.

`build_benchmark_dataset.py` mimics the paper's own "Benchmark Setting" construction (Section VII-A): it
vertically splits the source table's columns across a random tree of smaller tables, linking them with synthetic
KFK columns (the same row index copied along each branch) so every join is lossless — the same pattern as the
shipped `data/benchmark/credit/table_*.csv` + `connections.csv`. It also appends a row to `datasets.csv`
automatically. Since the split (which columns land where, tree shape) is randomized by `--seed`, results from a
self-built dataset are a new comparison point, not a reproduction of any paper number.

## Server-side testing

Since there is no real `beluga` package in this checkout yet, testing the adapter means faking what the server
would hand it: a base table directory, a `join_paths.csv`, and a lake corpus directory.

```bash
uv run python setup_baseline_test_fixture.py   # materializes tmp/queries/credit and tmp/corpus/credit_lake
uv run python test_baseline.py                 # runs AutoFeatBaseline.run(config) against that fixture
```

`test_baseline.py` builds `config` as a plain `SimpleNamespace` rather than importing `beluga.config.schema.Config`
— `baseline.py` never imports `beluga` directly (see the `try/except ImportError` in `_read_base_table`), so this
is a faithful stand-in for how the real harness will call it. A successful run prints the augmented `polars.DataFrame`
(base table columns + selected joined features), e.g.:

```
shape: (1_000, 12)
┌─────────┬─────────┬─────┬───────────────────────┬─────────────────────────┐
│ column0 ┆ column1 ┆ ... ┆ credit_lake.feature_x ┆ credit_lake.feature_y   │
│ ---     ┆ ---     ┆     ┆ ---                   ┆ ---                     │
│ i64     ┆ f64     ┆     ┆ f64                   ┆ str                     │
╞═════════╪═════════╪═════╪═══════════════════════╪═════════════════════════╡
│ ...     ┆ ...     ┆ ... ┆ ...                   ┆ ...                     │
└─────────┴─────────┴─────┴───────────────────────┴─────────────────────────┘
```

Exact column names/counts depend on which join path AutoFeat ranks best; if the base table alone outranks every
join, the base table is returned unchanged (no `_lake.` prefixed columns). A failed run raises a `ValueError`
instead — see the note at the end of this section for what each one means.

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

### Example: end-to-end against real server data

Real run against a NYC open-data corpus on the server, starting from a base table with no `join_paths.csv` yet:

```bash
cd ~/data/baselines/autofeat

# 1. Discover a join_paths.csv (small --limit first — the corpus is 60GB)
uv run python discover_join_paths.py \
    --corpora ~/data/corpora/open_data/joinable_tables/ --limit 5 \
    --input ~/data/corpora/open_data/joinable_tables/nyc/nyc-finance-39g5-gbp3/table.csv \
    --query_column agency_name --target_column total_current_budget_amount \
    --sample-rows 5000 --verbose
```

```
Matching .../nyc-finance-39g5-gbp3/table.csv ('agency_name') against 5 candidate tables ...
  match: nyc-education-2pmj-y4p4.school_name (similarity=0.39)
  match: nyc-education-35sw-rdxj.agency (similarity=0.52)
  ...
5 join path(s) written to tmp/queries/nyc-finance-39g5-gbp3/join_paths.csv

Run:
  python test_baseline.py --queries-dir tmp/queries --data-dir tmp/corpus --corpus nyc-finance-39g5-gbp3_lake --base-table nyc-finance-39g5-gbp3 --target-column-id 12
```

```bash
# 2. Run test_baseline.py with the printed command (prefix uv run, and total_current_budget_amount
#    is a dollar amount, so use --downstream-task regression, not the classification default)
uv run python test_baseline.py \
    --queries-dir tmp/queries --data-dir tmp/corpus --corpus nyc-finance-39g5-gbp3_lake \
    --base-table nyc-finance-39g5-gbp3 --target-column-id 12 --downstream-task regression
```

```
shape: (1_256, 31)
┌─────────────────┬─────────────────┬─────┬──────────────────┬──────────────────┐
│ other_categorical_funds  ┆ inter_fund_agreement ┆ ... ┆ unit_appropriation_number ┆ federal_funds_current_...  │
│ ---              ┆ ---               ┆     ┆ ---              ┆ ---              │
│ i64               ┆ i64               ┆     ┆ i64              ┆ i64              │
╞═══════════════════╪═══════════════════╪═════╪══════════════════╪══════════════════╡
│ 0                  ┆ 0                  ┆ ... ┆ 2                 ┆ 0                 │
│ 0                  ┆ 0                  ┆ ... ┆ 122               ┆ 12331687          │
└───────────────────┴───────────────────┴─────┴──────────────────┴──────────────────┘
```

Two things worth checking on a real run like this before trusting the result:
- **Match quality** — `discover_join_paths.py`'s default `--threshold 0.55` gates weak matches, but similarity
  alone doesn't guarantee the columns are *semantically* joinable (e.g. `agency_name` matching `school_name` at
  0.52 is a plausible but not certain real relationship). Spot-check a joined column's values against the base
  table before trusting the result.
- **`--downstream-task`** — `test_baseline.py` defaults to `classification`; pick `regression` explicitly for a
  continuous target like a dollar amount, as above.

`--limit` works by walking `join_paths.csv` from the base table outward (BFS over `from_id`/`to_id`) and writing
a temporary join-paths file restricted to the first `limit` tables reached, passed to `baseline.py` via its
existing `connections_csv_path` override — no change to `baseline.py` itself. It bounds *which* lake tables can be
read, not how much of any single table is read.

### Examples

```bash
# 1. Local fixture, defaults (classification on the credit dataset)
uv run python setup_baseline_test_fixture.py
uv run python test_baseline.py

# 2. Local fixture, explicit flags (equivalent to the defaults above)
uv run python test_baseline.py \
    --queries-dir tmp/queries --data-dir tmp/corpus --corpus credit_lake \
    --base-table credit --target-column-id 0 --downstream-task classification

# 3. Regression task, numeric target in a different column
# (requires a fixture for that base table first — setup_baseline_test_fixture.py only
#  builds the "credit" one; point --queries-dir/--data-dir at your own tmp/queries/<table>
#  + tmp/corpus/<table>_lake, or adapt setup_baseline_test_fixture.py's SOURCE)
uv run python test_baseline.py --base-table steel --target-column-id 3 --downstream-task regression

# 4. Real server layout, unbounded (small/known corpus)
uv run python test_baseline.py \
    --queries-dir /srv/queries --data-dir /srv/corpora --corpus my_lake --base-table my_table

# 5. Real server layout, bounded traversal (large corpus, e.g. 60GB)
uv run python test_baseline.py \
    --queries-dir /srv/queries --data-dir /srv/corpora --corpus my_lake --base-table my_table \
    --limit 20
```

A successful run prints the augmented `polars.DataFrame` and exits 0 (see the sample output above). With `--limit`,
it first prints a line like `Limited join graph to 20 tables (37 of 210 edges) -> tmp/queries/_join_paths_limit_20.csv`
before the run itself starts, confirming how much of the join graph was actually kept.

A failed run raises a `ValueError` instead of printing a DataFrame — the message tells you which check failed:

| Message contains | Meaning |
| --- | --- |
| `"must be set"` | A required config field (`base_table`, `queries_dir`, `data_dir`) was missing |
| `"downstream_task ... not supported"` | `--downstream-task` was neither `classification` nor `regression` |
| `"Target column ... not numeric"` | `--downstream-task regression` but the target column isn't numeric |
| `"Cannot resolve base table file in ..."` | `queries_dir/<base_table>/` has zero or more than one CSV (besides `join_paths.csv`) |
| `"Join failed for path"` | The best-ranked join path couldn't actually be materialized (e.g. a stale/mismatched `join_paths.csv`) |
