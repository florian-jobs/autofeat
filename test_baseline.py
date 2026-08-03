import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from baseline import AutoFeatBaseline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AutoFeatBaseline against a queries/corpus directory (local fixture or real server layout)."
    )
    parser.add_argument("--queries-dir", default="tmp/queries", help="Directory containing <base-table>/ with the base table CSV + join_paths.csv")
    parser.add_argument("--data-dir", default="tmp/corpus", help="Directory containing the lake corpus")
    parser.add_argument("--corpus", default="credit_lake", help="Corpus subfolder name under --data-dir")
    parser.add_argument("--base-table", default="credit", help="Base table id")
    parser.add_argument("--target-column-id", type=int, default=0, help="Target column index")
    parser.add_argument("--downstream-task", default="classification", choices=["classification", "regression"])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of lake tables to traverse from the base table (BFS-bounded). "
             "Use this to sample a subgraph out of a large (e.g. 60GB) corpus instead of the whole join graph.",
    )
    return parser.parse_args()


def build_limited_join_paths(join_paths_path: Path, base_table_id: str, limit: int) -> Path:
    """Write a copy of join_paths.csv restricted to a BFS-bounded subgraph of at most `limit` tables."""
    join_paths_df = pd.read_csv(join_paths_path)

    visited = {base_table_id}
    frontier = {base_table_id}
    while frontier and len(visited) < limit:
        neighbours = set(join_paths_df.loc[join_paths_df["from_id"].isin(frontier), "to_id"])
        neighbours |= set(join_paths_df.loc[join_paths_df["to_id"].isin(frontier), "from_id"])
        frontier = (neighbours - visited)
        if len(visited) + len(frontier) > limit:
            frontier = set(list(frontier)[: limit - len(visited)])
        visited |= frontier

    limited_df = join_paths_df[
        join_paths_df["from_id"].isin(visited) & join_paths_df["to_id"].isin(visited)
    ]
    limited_path = Path(tempfile.gettempdir()) / f"join_paths_limit_{limit}.csv"
    limited_df.to_csv(limited_path, index=False)
    print(f"Limited join graph to {len(visited)} tables ({len(limited_df)} of {len(join_paths_df)} edges) -> {limited_path}")
    return limited_path


def main():
    args = parse_args()

    config = SimpleNamespace(
        queries_dir=args.queries_dir,
        base_table=args.base_table,
        data_dir=args.data_dir,
        corpus=args.corpus,
        target_column_id=args.target_column_id,
        downstream_task=args.downstream_task,
    )

    if args.limit is not None:
        join_paths_path = Path(args.queries_dir) / args.base_table / "join_paths.csv"
        base_table_id = args.base_table + ".csv"
        config.connections_csv_path = str(build_limited_join_paths(join_paths_path, base_table_id, args.limit))

    baseline = AutoFeatBaseline(value_ratio=0.65, top_k=15, algorithm="LR")
    result = baseline.run(config)
    print(result)


if __name__ == "__main__":
    main()
