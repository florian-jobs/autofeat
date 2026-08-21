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

## Why joined-table counts deviate from the paper

Paper's Figure 4 (benchmark setting, tree-based models), tables joined per approach:

| Dataset | ARDA (paper) | MAB (paper) | AutoFeat (paper) | AutoFeat, this repo (ground-truth `connections.csv`) | AutoFeat, this repo (per-dataset discovered graph) |
| --- | --- | --- | --- | --- | --- |
| bioresponse | 3 | 39 | **1** | 37 | 7 |
| jannis | 3 | 11 | **1** | 13 | 8 |

**The benchmark corpus is the authors' real, published data — not a locally-reconstructed guess.** It's archived on Zenodo: [10.5281/zenodo.12755408](https://zenodo.org/records/12755408) (`autofeat-data.tar`, ~203MB, CC-BY-4.0), linked from upstream's own README under "Data setup". `bioresponse`'s `connections.csv` there is byte-identical to the one this repo already had (79 edges for 41 tables) — confirming it's authentic, not denser-than-intended or stale. **Earlier revisions of this README wrongly diagnosed that 79-edge file as a "stale corpus bug" and recommended regenerating it with `build_benchmark_dataset.py`; that was a mistake** — doing so replaces the real paper data with an unrelated synthetic reconstruction (different tree, different column-to-table split), which is *not* a valid stand-in for reproducing the paper's numbers. **Always source `data/benchmark/<dataset>/` from the Zenodo archive**; only use `build_benchmark_dataset.py` for genuinely new datasets that aren't part of the paper's published set.

So against the authentic ground-truth corpus, the gap is real: **37 tables joined vs. the paper's 1** — no corpus-authenticity issue involved. But the ground-truth `connections.csv` tree is not the only legitimate join graph: it's a *hand-authored* snowflake schema, not the graph AutoFeat's own offline phase would discover. Running the paper's actual discovery mechanism — per-dataset Valentine/Coma schema matching (see "Live Neo4j/Valentine ingestion" below) — instead of reading the ground-truth tree cuts the gap substantially, system-wide, not just for bioresponse:

| Dataset | ground-truth `connections.csv` | per-dataset discovered graph |
| --- | --- | --- |
| bioresponse | 37 | **7** |
| covertype | 12 | **4** |
| credit | 6 | 6 |
| eyemove | 7 | 7 |
| jannis | 13 | **8** |
| miniboone | 14 | **4** |
| school | 6 | 8 |
| steel | 16 | 16 |

Five of eight datasets drop substantially (bioresponse 37→7, covertype 12→4, jannis 13→8, miniboone 14→4); two stay flat because they're small enough that Valentine reconnects nearly every table anyway (credit, eyemove); school ticks up slightly. Net effect: the discovered graph is *consistently* closer to (or no worse than) the paper's own numbers than the ground-truth tree is — this isn't a one-dataset anecdote, and the earlier framing below ("supplementary, not a fix") undersold it. **This does not fully close the gap** (7 and 8 are still well above the paper's 1), and accuracy drops alongside table count for some datasets (bioresponse 0.75→0.60, jannis 0.74→0.51) since fewer joined tables means fewer candidate features — expected, not a regression. What remains after switching to the discovered graph is explained by the same algorithmic mechanism as before:

**Algorithm behavior — BFS never prunes a candidate join on relevance/redundancy, only on the τ null-value-ratio check.** Verified byte-for-byte against upstream `delftdata/autofeat`: `current_queue.add(join_name)` runs unconditionally otherwise, in both this port and upstream. So it chains through most of the *reachable* graph by design — which is exactly why starting from a sparser, correctly-per-dataset-scoped graph (discovered) joins fewer tables than starting from a denser one (ground truth), even though neither graph is corpus-inauthentic. Which specific subset wins as "best" is decided by Python's `set()` iteration order, i.e. `PYTHONHASHSEED` (also a no-op-when-set-mid-process issue, separately fixed — see `ablation.py`'s subprocess relaunch). Local runs are internally reproducible (`PYTHONHASHSEED=42`) but land on a different, equally valid draw than the paper's own undocumented seed. Not fixable without deviating from the paper's own Algorithm 1; `sweep_hashseed.py --dataset bioresponse --mode full --seeds <range>` quantifies the actual seed-to-seed variance instead of chasing a single "correct" seed.

**The authors' own reproducibility gap.** Upstream's `autofeat.py` pins `random_state=42`/`seed=42` in exactly four places — `train_test_split` (base-table sampling, `initialisation()`), AutoGluon's `AutoMLPipelineFeatureGenerator.fit_transform` (`streaming_relevance_redundancy()`), and the polars/pandas row-sampling in `step_join()` — but never touches `PYTHONHASHSEED`. So the authors clearly cared about reproducibility for every sampling/training step they controlled directly, yet the actual traversal order (`queue.pop()`, `previous_queue.pop()`) was left to whatever hash seed their interpreter happened to have at the time — almost certainly unintentional, not a documented experimental choice.

**Why not just replace `set()` with something deterministic ourselves?** Because that would silently change the algorithm being evaluated, not just fix its nondeterminism. The paper's Algorithm 1 and the reference implementation are both genuinely `set`-based (`Rsel = Rsel ∪ Rred`, no ordering guarantee); swapping in a list or a forced sort order would replace one arbitrary tie-break rule (hash order) with a different arbitrary one (insertion order, alphabetical, ...) — no more "correct," and no closer to whatever specific accidental order produced the paper's own numbers, which isn't recoverable either way. Pinning `PYTHONHASHSEED` keeps the algorithm exactly as specified/published while making *our* runs internally reproducible — the closest available thing to a principled fix.

**Related confusion: is κ (`--top-k`) 5 or 15?** Paper text states **κ=15** explicitly ("maximum selected features from a table"); upstream's class constructor default of 5 is an unused Python fallback, not what the paper's reported experiments used. `ablation.py`'s CLI default is `15`, matching the paper — this doesn't explain the joined-table deviation, but was a live source of confusion while investigating it.

## Live Neo4j/Valentine ingestion (the recommended default — closes most of the gap above)

`reingest_dataset.py` + `export_discovered_connections.py` restore the OG pipeline's *other* join-discovery
mechanism: real schema matching (Valentine/Coma) into a live Neo4j instance, instead of reading a
pre-published `connections.csv`. This is what AutoFeat's own offline phase actually does — the ground-truth
`connections.csv` tree is a hand-authored stand-in for it, not a replacement.

**Must be run per-dataset, never across the whole data lake in one pass.** `profile_valentine_all`
(`--discover-connections-data-lake` in the OG CLI) matches every table against every other table
regardless of which benchmark dataset it belongs to; because tables are named generically
(`table_0_0.csv`, `table_1_1.csv`, ...), this produces cross-dataset `RELATED` edges and BFS then leaks
features between unrelated datasets (e.g. from jannis into miniboone) — this is the failure mode a denser,
same-shaped-table corpus is naturally exposed to, and it's why `reingest_dataset.py` always calls
`profile_valentine_dataset` (per-dataset glob, per-dataset node wipe), matching the paper's own
per-benchmark-dataset matching description. If you see suspiciously high cross-dataset feature leakage,
confirm the offending run didn't go through `profile_valentine_all` instead.

Run for all 8 benchmark datasets, this consistently reduces joined-table counts relative to the
ground-truth tree (see table above) — bioresponse 37→7, covertype 12→4, jannis 13→8, miniboone 14→4, with
credit/eyemove/steel roughly unchanged and school ticking up by 2. `run_all_ablation.sh` already prefers
`connections_discovered.csv` over `connections.csv` automatically when present, so once discovery has been
run for a dataset there's nothing else to opt into.

```bash
# One-time: a disposable Neo4j instance (Community Edition works; production uses an Enterprise instance
# with NEO4J_DATABASE=lake — Community only supports its single default db, hence the override below).
# Needs a JRE on PATH (Valentine's Coma matcher shells out to it) and neo4j==4.4.0/valentine==0.1.6/
# joblib==1.2.0, all already in this repo's own dependencies.
NEO4J_HOST="bolt://localhost:7688" NEO4J_DATABASE="neo4j" uv run python reingest_dataset.py --dataset credit
NEO4J_HOST="bolt://localhost:7688" NEO4J_DATABASE="neo4j" uv run python export_discovered_connections.py --dataset credit
uv run python src/feature_discovery/experiments/ablation.py --dataset credit \
    --connections data/benchmark/credit/connections_discovered.csv
# or, once discovery has been run for every dataset you care about: ./run_all_ablation.sh
```

`reingest_dataset.py --no-wipe` adds to an existing graph instead of clearing the dataset's nodes first;
it only ever wipes nodes under `<dataset>/`, never the whole database, so it's safe to point at a shared
instance holding other datasets' graphs too.

**Local Neo4j Community Edition is memory-fragile under this workload.** The parallel `joblib`
`Parallel(n_jobs=-1)` fan-out in `profile_valentine_dataset` spawns one JVM-backed Coma matcher per core
alongside the Neo4j server's own JVM; on a resource-constrained machine the server can die silently
mid-run (no shutdown log entry, no `hs_err` crash dump — just gone) after a few datasets' worth of
ingestion. It comes back cleanly on restart (`bin/neo4j.bat console`, runs transaction-log recovery
automatically) — just re-run `reingest_dataset.py` for whichever dataset was in flight when it died; already
`export`-ed datasets are unaffected since their `connections_discovered.csv` is already on disk.

Bugs fixed while restoring this (all pre-existing, not introduced by this rework): `ingest_data.py`/
`dataset_discovery.py` imported functions (`merge_nodes_relation_tables`, `create_node`) that no longer
existed once `neo4j_transactions.py` became the in-memory simulate layer — both files were silently
unimportable; a Windows path-separator bug (`str.partition` on a literal `"/"` against `glob.glob`'s
backslash paths) collapsed every node ID to `""`; the graph-export query compared `node.label` against
the relationship's `from_label`/`to_label`, which are full paths, not labels — never matched; and
`get_relation_properties_node_name` returned a matched row as-is regardless of which direction it was
stored in, which crashes downstream on any graph with bidirectional edges (a discovered graph always has
both directions; the ground-truth `connections.csv` never did, so this never surfaced before).

Also ported: `MIN_JOIN_KEY_CARDINALITY` (`autofeat.py`) — hard-drops join keys with fewer than 3 distinct
values, preventing the join-path blowup binary indicator columns cause (e.g. covertype's Soil_Type
flags). Applies regardless of which graph source is in use.

## Other fixed bugs

| Symptom | Cause | Fix |
| --- | --- | --- |
| `OpenBLAS: Program is Terminated. ... too many memory regions` on hercules | Many-core server: `joblib`'s `Parallel(n_jobs=-1)` and AutoGluon's own worker processes each open an OpenBLAS thread pool sized to core count — total exceeds OpenBLAS's hardcoded 128-thread build limit | `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS` pinned to 1 via `os.environ.setdefault(...)` before `numpy` import in `cli.py`, `ablation.py`, `dataset_discovery.py` |
| `IndexError`/`ValueError` in `measure_redundancy` on a fresh dataset's first hop | Empty `selected_features` → `np.array([])` defaults to `float64`, which numpy refuses for fancy indexing | `21660ef` — empty selections short-circuit to zero redundancy contribution |
| `./run_all_ablation.sh` crashed every dataset on `argparse` int conversion | `${1:15}` is bash substring expansion, not `${1:-15}` default-value syntax | `17b01ee` — `TOP_K="${2:-15}"`, independent positional args |
| BFS stopped after 1 hop regardless of `--top-k`; `--algorithm` silently ignored; stratified sampling never triggered for classification | Older `autofeat.py`/`ablation.py` bugs | See `git log` on those files |

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
| `sweep_hashseed.py` | Quantifies BFS's `PYTHONHASHSEED` sensitivity across seeds (see above) |
| `reingest_dataset.py` / `export_discovered_connections.py` | Live Neo4j/Valentine join discovery, for datasets without a published ground-truth graph (see above) |
| `src/feature_discovery/graph_processing/neo4j_live.py` | Driver-backed Neo4j client used by the above; `neo4j_transactions.py` stays the in-memory simulate layer `autofeat.py`/`baseline.py` depend on |

`ablation.py` flags: `--dataset` (`credit`), `--value-ratio` (τ, `0.65`), `--top-k` (κ, `15`), `--algorithm`
(`LR`/`RF`/`GBM`/`XT`/`XGB`/`KNN`), `--sample-size` (`3000`, raise for large base tables e.g. `covertype`).
Compare `summarize_results.py` output against Figure 4 (`GBM`/`XGB`) or 5 (`LR`) in the paper, not Table II's
"Best accuracy (OpenML.org)" column.