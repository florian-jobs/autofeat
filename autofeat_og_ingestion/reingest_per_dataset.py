"""Build the OG AutoFeat ``lake`` Neo4j graph, one benchmark dataset at a time.

Background. The upstream feature-discovery README suggests the
``--discover-connections-data-lake`` ingestion path, which runs Valentine
matching on all table pairs across datasets. That produces cross-dataset
RELATED edges and lets the BFS join discovery leak features between
benchmark datasets (e.g. from jannis into miniboone). The AutoFeat paper
text instead describes per-benchmark-dataset Valentine matching, which
corresponds to ``ingest_nodes`` followed by ``profile_valentine_dataset``.
This script reproduces the latter.

Procedure:

1. ensure the target Neo4j database exists, then wipe it,
2. rebind ``feature_discovery.config.DATA_FOLDER`` to the dataset root,
3. for each dataset folder call ``ingest_nodes`` + ``profile_valentine_dataset``
   (Valentine/Coma threshold 0.55, the value fixed by the AutoFeat paper)
   while temporarily hiding the raw ``<dataset>.csv`` duplicate so that only
   the OG split-table fragments end up in the graph,
4. verify that the resulting graph contains no cross-dataset edges.

Configuration (environment variables, with defaults):

    AUTOFEAT_DATA_ROOT   dataset root        <this dir>/data/autofeat
    NEO4J_HOST           bolt URI            neo4j://localhost:7687
    NEO4J_USER           user                neo4j
    NEO4J_PASS           password            (upstream default; ignored if auth is disabled)
    NEO4J_DATABASE       target database     lake

Run with the package importable, e.g.:

    pip install -r requirements-ingest.txt && pip install -e . --no-deps
    python reingest_per_dataset.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

DATA_ROOT = Path(
    os.getenv("AUTOFEAT_DATA_ROOT", Path(__file__).resolve().parent / "data" / "autofeat")
).resolve()

os.environ.setdefault("NEO4J_DATABASE", "lake")

from feature_discovery import config as fd_config  # noqa: E402

fd_config.DATA_FOLDER = DATA_ROOT

# These modules captured DATA_FOLDER at import time; rebind explicitly.
from feature_discovery.dataset_relation_graph import ingest_data as fd_ingest  # noqa: E402
from feature_discovery.dataset_relation_graph import dataset_discovery as fd_disc  # noqa: E402

fd_ingest.DATA_FOLDER = DATA_ROOT
fd_disc.DATA_FOLDER = DATA_ROOT

from neo4j import GraphDatabase  # noqa: E402


# (label, on-disk raw csv to hide during ingest). The latter is whatever
# OG's datasets.csv calls the "base_table" file. Hiding prevents the raw
# full table being added as a graph node alongside the split fragments,
# which would otherwise produce intra-dataset Valentine edges between the
# raw table and ``table_0_0.csv`` (same columns) and break BFS. The
# shipped data/ tree already omits these duplicates, so the hiding is a
# no-op unless you point AUTOFEAT_DATA_ROOT at a tree that contains them.
DATASETS = [
    ("bioresponse", ["bioresponse.csv"]),
    ("covertype", ["covertype.csv"]),
    ("credit", ["credit.csv"]),
    ("eyemove", ["eyemove.csv"]),
    ("jannis", ["jannis.csv"]),
    ("miniboone", ["miniboone.csv"]),
    # school's OG base is ``base.csv``; the ``school.csv`` file is a raw
    # duplicate that would create intra-dataset edges back to ``base.csv``
    # via the shared ``DBN`` key, breaking BFS the same way.
    ("school", ["school.csv"]),
    ("steel", ["steel.csv"]),
]


def ensure_database() -> None:
    """Create the target database if the server supports multiple databases.

    Neo4j Enterprise (and the calendar releases with multi-database support)
    accept ``CREATE DATABASE``. On Community editions this fails; there,
    set ``initial.dbms.default_database=lake`` in neo4j.conf instead, or
    export NEO4J_DATABASE=neo4j to use the default database.
    """
    drv = GraphDatabase.driver(fd_config.NEO4J_HOST, auth=fd_config.NEO4J_CREDENTIALS)
    try:
        with drv.session(database="system") as s:
            s.run(f"CREATE DATABASE {fd_config.NEO4J_DATABASE} IF NOT EXISTS").consume()
        print(f"Database '{fd_config.NEO4J_DATABASE}' present.")
    except Exception as exc:  # noqa: BLE001
        print(
            f"Could not create database '{fd_config.NEO4J_DATABASE}' ({exc}). "
            "If it already exists this is harmless. On Neo4j Community, set "
            "initial.dbms.default_database or NEO4J_DATABASE instead."
        )
    finally:
        drv.close()


def wipe_database() -> None:
    drv = GraphDatabase.driver(fd_config.NEO4J_HOST, auth=fd_config.NEO4J_CREDENTIALS)
    with drv.session(database=fd_config.NEO4J_DATABASE) as s:
        s.run("MATCH (n) DETACH DELETE n").consume()
        n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        r = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    drv.close()
    print(f"After wipe: nodes={n} rels={r}")


def reingest() -> None:
    for ds, raw_files in DATASETS:
        folder = DATA_ROOT / ds
        if not folder.exists():
            print(f"SKIP {ds} (missing folder)")
            continue
        # Hide each raw duplicate CSV that would otherwise be added as a
        # graph node and Valentine-matched against the OG base table.
        hidden_pairs: list[tuple[Path, Path]] = []
        for raw in raw_files or []:
            raw_path = folder / raw
            if raw_path.exists():
                hidden = raw_path.with_suffix(raw_path.suffix + ".__hidden")
                shutil.move(str(raw_path), str(hidden))
                hidden_pairs.append((raw_path, hidden))
        try:
            print(f"=== {ds}: ingest_nodes ===")
            fd_ingest.ingest_nodes(dataset_folder_name=ds)
            print(f"=== {ds}: profile_valentine_dataset(threshold=0.55) ===")
            fd_disc.profile_valentine_dataset(dataset_name=ds, valentine_threshold=0.55)
        finally:
            for raw_path, hidden in hidden_pairs:
                if hidden.exists():
                    shutil.move(str(hidden), str(raw_path))


def summarise() -> None:
    drv = GraphDatabase.driver(fd_config.NEO4J_HOST, auth=fd_config.NEO4J_CREDENTIALS)
    with drv.session(database=fd_config.NEO4J_DATABASE) as s:
        n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        r = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(f"nodes={n} rels={r}")
        rows = s.run(
            "MATCH (a)-[r]->(b) "
            "WITH split(a.id,'/')[0] AS da, split(b.id,'/')[0] AS db, count(*) AS c "
            "WHERE da <> db RETURN da, db, c ORDER BY c DESC"
        )
        cross = list(rows)
        if cross:
            print("CROSS-DATASET EDGES (should be empty):")
            for row in cross:
                print(f"  {row['da']} -> {row['db']}: {row['c']}")
        else:
            print("No cross-dataset edges.")
    drv.close()


if __name__ == "__main__":
    ensure_database()
    wipe_database()
    reingest()
    summarise()
