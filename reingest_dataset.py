"""
Build (or rebuild) one dataset's join-discovery graph in Neo4j via genuine schema matching
(Valentine/Coma) -- the same mechanism used to build the reference graph behind the AutoFeat
paper's reported numbers (see autofeat_og_ingestion's reingest_per_dataset.py, which this
generalizes: no hardcoded per-dataset list, works against this repo's own data/<DATASET_TYPE>
layout via feature_discovery.config, and wipes only this dataset's nodes rather than the whole
database).

Why per-dataset ingestion (not --discover-connections-data-lake across every dataset at once):
running Valentine across all datasets in one pass produces cross-dataset RELATED edges, and BFS
join discovery then leaks features between benchmark datasets (e.g. from jannis into miniboone).
The AutoFeat paper describes per-benchmark-dataset matching, which is what ingest_nodes +
profile_valentine_dataset (this script) does.

Usage:
    uv run python reingest_dataset.py --dataset bioresponse
    uv run python reingest_dataset.py --dataset bioresponse --no-wipe   # add to an existing graph
    NEO4J_DATABASE=neo4j uv run python reingest_dataset.py --dataset credit   # Community Edition
"""
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Folder name under data/<DATASET_TYPE>, e.g. bioresponse")
    parser.add_argument("--threshold", type=float, default=0.55, help="Valentine/Coma similarity threshold (paper default: 0.55)")
    parser.add_argument("--wipe", dest="wipe", action="store_true", default=True)
    parser.add_argument("--no-wipe", dest="wipe", action="store_false", help="Add to the existing graph instead of clearing this dataset's nodes first")
    args = parser.parse_args()

    from feature_discovery.config import DATA_FOLDER
    from feature_discovery.dataset_relation_graph import ingest_data as fd_ingest
    from feature_discovery.dataset_relation_graph import dataset_discovery as fd_disc
    from feature_discovery.graph_processing import neo4j_live

    folder = DATA_FOLDER / args.dataset
    if not folder.exists():
        raise SystemExit(f"--dataset {args.dataset!r} not found under {DATA_FOLDER}")

    # Hide the raw, undecomposed full table (e.g. bioresponse.csv) during ingestion if present --
    # if ingested it Valentine-matches against table_0_0.csv (near-identical columns) and breaks
    # BFS with a spurious intra-dataset edge. No-op if the tree doesn't ship that duplicate.
    raw_duplicate = folder / f"{args.dataset}.csv"
    hidden = None
    if raw_duplicate.exists():
        hidden = raw_duplicate.with_suffix(raw_duplicate.suffix + ".__hidden")
        shutil.move(str(raw_duplicate), str(hidden))
        print(f"Hid raw duplicate {raw_duplicate.name} for ingestion")

    try:
        if args.wipe:
            print(f"Wiping existing '{args.dataset}' nodes (if any) ...")
            neo4j_live.wipe_dataset(args.dataset)

        print(f"=== {args.dataset}: ingest_nodes ===")
        fd_ingest.ingest_nodes(dataset_folder_name=args.dataset)

        print(f"=== {args.dataset}: profile_valentine_dataset(threshold={args.threshold}) ===")
        fd_disc.profile_valentine_dataset(dataset_name=args.dataset, valentine_threshold=args.threshold)
    finally:
        if hidden is not None and hidden.exists():
            shutil.move(str(hidden), str(raw_duplicate))

    rows = neo4j_live.export_dataset_connections(args.dataset)
    tables = sorted({r["from_table"] for r in rows} | {r["to_label"] for r in rows})
    print(f"\nDone. {len(rows)} discovered edges across {len(tables)} tables for '{args.dataset}'.")


if __name__ == "__main__":
    main()
