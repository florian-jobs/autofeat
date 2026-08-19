"""
Export a dataset's discovered join graph (built by reingest_dataset.py via real Valentine/Coma
matching in Neo4j) into a connections.csv-shaped CSV that ablation.py's load_join_paths can read
directly -- as an alternative to the synthetic, hand-authored ground-truth tree in
data/<DATASET_TYPE>/<dataset>/connections.csv. Written to a separate file
(connections_discovered.csv) so it never clobbers the ground-truth file.

Usage:
    uv run python reingest_dataset.py --dataset bioresponse   # build the graph first
    uv run python export_discovered_connections.py --dataset bioresponse
    uv run python src/feature_discovery/experiments/ablation.py --dataset bioresponse \\
        --connections data/benchmark/bioresponse/connections_discovered.csv
"""
import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default=None, help="Output CSV path (default: data/<DATASET_TYPE>/<dataset>/connections_discovered.csv)")
    args = parser.parse_args()

    from feature_discovery.config import DATA_FOLDER
    from feature_discovery.graph_processing import neo4j_live

    rows = neo4j_live.export_dataset_connections(args.dataset)
    if not rows:
        raise SystemExit(
            f"No discovered edges found for {args.dataset!r} -- run "
            f"`uv run python reingest_dataset.py --dataset {args.dataset}` first"
        )

    df = pd.DataFrame([{
        "fk_table": r["from_table"],
        "fk_column": r["from_column"],
        "pk_table": r["to_label"],
        "pk_column": r["to_column"],
        "weight": r["weight"],
    } for r in rows])

    out_path = args.out or str(DATA_FOLDER / args.dataset / "connections_discovered.csv")
    df.to_csv(out_path, index=False)
    tables = sorted(set(df["fk_table"]) | set(df["pk_table"]))
    print(f"Wrote {len(df)} edges across {len(tables)} tables to {out_path}")


if __name__ == "__main__":
    main()
