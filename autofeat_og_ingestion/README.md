# OG AutoFeat per-dataset lake ingestion

This package reproduces the ingestion of the AutoFeat controlled-lake benchmark
tables into a Neo4j graph database, exactly as used in the Fast Data Discovery
ablation experiments. It builds one join-discovery graph per benchmark dataset
inside a single Neo4j database named `lake`, with no cross-dataset edges.

## Contents

| Path | Description |
| --- | --- |
| `reingest_per_dataset.py` | The ingestion driver. Wipes the `lake` database, then for each dataset calls the upstream `ingest_nodes` and `profile_valentine_dataset` (Valentine/Coma, threshold 0.55), and finally verifies that no cross-dataset edges exist. |
| `src/feature_discovery/` | The upstream feature-discovery package (Ionescu et al., ICDE), Apache-2.0, including the modifications used in our experiments. Only its ingestion modules are exercised here. |
| `data/autofeat/` | The 8 benchmark dataset folders (bioresponse, covertype, credit, eyemove, jannis, miniboone, school, steel) containing the OG split-table fragments (`table_i_j.csv`), plus `datasets.csv`. The raw full-table duplicates (`<dataset>.csv`) are intentionally omitted, see Notes. |
| `docker-compose.yml` | A Neo4j Enterprise server with authentication disabled, sufficient to run the ingestion. |
| `requirements-ingest.txt` | Pinned Python dependencies for the ingestion path, matching the environment in which the reference graph was built. |
| `pyproject.toml`, `LICENSE` | Upstream packaging metadata and Apache-2.0 license. |

## Requirements

1. Python 3.12 (the reference environment used 3.12.3).
2. A Java runtime on `PATH`. The Valentine Coma matcher shells out to a bundled
   `coma.jar` (the reference environment used OpenJDK 21).
3. A reachable Neo4j server with multi-database support, holding a database
   named `lake`. The reference environment used Neo4j 2026.04.0. The provided
   `docker-compose.yml` (Neo4j 5.x Enterprise, auth disabled) is sufficient.
   On Community editions, which support only one user database, either set
   `initial.dbms.default_database=lake` in `neo4j.conf` or run the script with
   `NEO4J_DATABASE=neo4j`.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-ingest.txt
docker compose up -d                # or point NEO4J_HOST at an existing server
PYTHONPATH="$PWD/src" python reingest_per_dataset.py
```

The script reads its configuration from environment variables, all optional.
`AUTOFEAT_DATA_ROOT` (default `./data/autofeat`), `NEO4J_HOST` (default
`neo4j://localhost:7687`), `NEO4J_USER`, `NEO4J_PASS` (ignored when
authentication is disabled), and `NEO4J_DATABASE` (default `lake`).

The `feature_discovery` package is used directly from `src/` via `PYTHONPATH`,
the same mechanism used to build the reference graph. Do not `pip install`
from the upstream `pyproject.toml` (kept for provenance only). It pins the
full pipeline stack (AutoGluon 0.7, scikit-learn 1.2, pandas 1.5), which is
neither needed for ingestion nor compatible with Python 3.12. The ingestion
closure in `requirements-ingest.txt` reflects the versions actually used to
build the reference graph.

## Expected outcome

The run wipes the `lake` database, ingests each dataset folder, runs Valentine
matching within each dataset at threshold 0.55, restores any hidden files, and
prints a summary. The final check must report `No cross-dataset edges.` The
Valentine step dominates the runtime because Coma matching is pairwise over the
tables of a dataset. Expect on the order of an hour in total, with covertype
the largest dataset.

## Why per-dataset ingestion

The upstream README suggests the `--discover-connections-data-lake` CLI path,
which runs Valentine on all table pairs across datasets. That produces
cross-dataset RELATED edges, and the BFS join discovery then leaks features
between benchmark datasets (for example from jannis into miniboone). The
AutoFeat paper text describes per-benchmark-dataset matching, which corresponds
to `ingest_nodes` followed by `profile_valentine_dataset`. This package
implements the latter.

## Notes

1. Raw duplicates. The original dataset folders also contain the raw full
   table `<dataset>.csv`. If ingested, it forms intra-dataset Valentine edges
   to `table_0_0.csv` (identical columns) and breaks the BFS. The driver hides
   such files during ingestion and restores them afterwards. The `data/` tree
   shipped here omits them entirely, so the hiding is a no-op. The resulting
   graph is identical either way.
2. school. Its OG base table is `base.csv` rather than `table_0_0.csv` (see
   `data/autofeat/datasets.csv`), and `school.csv` is the raw duplicate that is
   excluded. A freshly ingested graph contains a single base node per dataset.
3. Determinism. Given the same table files and threshold, Coma matching is
   deterministic, so repeated runs produce the same graph.
