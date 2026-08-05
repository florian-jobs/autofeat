"""
Split a single wide CSV (e.g. an OpenML tabular classification dataset) into a snowflake-schema
benchmark dataset, following the "Benchmark Setting" construction described in the AutoFeat paper
(Section VII-A): a base table + a random tree of child tables, linked by synthetic KFK columns that
are just the shared row-index copied along each branch (so every join is lossless and 1:1, matching
the paper's data/benchmark/credit layout: table_0_0.csv -> table_1_1.csv/table_1_2.csv -> ...).

Output layout matches what ablation.py / load_join_paths() already expect:
    data/benchmark/<name>/table_<depth>_<idx>.csv   (one per node, base is table_0_0.csv)
    data/benchmark/<name>/connections.csv            (fk_table, fk_column, pk_table, pk_column)
and a row appended to data/benchmark/datasets.csv so it flows through init_datasets()/ablation.py.

Example:
    uv run python build_benchmark_dataset.py \\
        --input path/to/openml_dataset.csv --target-column class \\
        --name mydataset --num-tables 6 --max-depth 2
"""
import argparse
import random
from pathlib import Path

import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the source CSV (a single wide OpenML-style table)")
    parser.add_argument("--target-column", required=True, help="Label column; stays in the base table only")
    parser.add_argument("--name", required=True, help="Dataset name; output goes to data/benchmark/<name>/")
    parser.add_argument("--num-tables", type=int, default=6, help="Total number of tables, including the base table")
    parser.add_argument("--max-depth", type=int, default=2, help="Max tree depth below the base table")
    parser.add_argument("--min-cols-per-table", type=int, default=1, help="Minimum feature columns per non-base table")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark-dir", default="data/benchmark", help="Root benchmark directory")
    return parser.parse_args()

def build_tree(num_tables: int, max_depth: int, rng: random.Random):
    """Return a list of (table_id, depth, parent_id) for a random tree rooted at table_0_0, one entry per table."""
    nodes = [("table_0_0", 0, None)]
    per_depth_count = {0: 1}
    while len(nodes) < num_tables:
        depth = rng.randint(1, max_depth)
        candidate_parents = [n for n in nodes if n[1] == depth - 1]
        if not candidate_parents:
            continue
        parent_id, _, _ = rng.choice(candidate_parents)
        idx = per_depth_count.get(depth, 0)
        table_id = f"table_{depth}_{idx}"
        per_depth_count[depth] = idx + 1
        nodes.append((table_id, depth, parent_id))
    return nodes

def main():
    args = parse_args()
    rng = random.Random(args.seed)

    df = pd.read_csv(args.input)
    if args.target_column not in df.columns:
        raise ValueError(f"--target-column {args.target_column!r} not found (columns: {list(df.columns)})")

    feature_columns = [c for c in df.columns if c != args.target_column]
    required_columns = (args.num_tables - 1) * args.min_cols_per_table
    if required_columns > len(feature_columns):
        raise ValueError(
            f"--num-tables {args.num_tables} with --min-cols-per-table {args.min_cols_per_table} needs "
            f"{required_columns} feature columns, only {len(feature_columns)} available; "
            f"lower --num-tables or --min-cols-per-table"
        )

    nodes = build_tree(args.num_tables, args.max_depth, rng)
    node_ids = [n[0] for n in nodes]

    # Randomly partition feature columns across all tables (base included), respecting the minimum
    # per non-base table so every child table actually carries some real data.
    shuffled = feature_columns[:]
    rng.shuffle(shuffled)
    columns_per_table = {tid: [] for tid in node_ids}
    non_base = node_ids[1:]
    i = 0
    for tid in non_base:
        columns_per_table[tid] = shuffled[i:i + args.min_cols_per_table]
        i += args.min_cols_per_table
    for col in shuffled[i:]:
        columns_per_table[rng.choice(node_ids)].append(col)

    out_dir = Path(args.benchmark_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    row_id = pd.RangeIndex(len(df))
    own_key = {}  # table_id -> its own exposed key column name (only if it has children)
    children_of = {tid: [] for tid in node_ids}
    parent_of = {}
    for tid, _depth, parent_id in nodes:
        parent_of[tid] = parent_id
        if parent_id is not None:
            children_of[parent_id].append(tid)

    for tid, depth, parent_id in nodes:
        depth_suffix = tid.split("_", 1)[1]  # "<depth>_<idx>"
        table_df = pd.DataFrame({c: df[c] for c in columns_per_table[tid]})

        if parent_id is not None:
            parent_key = own_key[parent_id]
            table_df.insert(0, parent_key, row_id)

        if children_of[tid]:
            key_name = f"Key_{depth_suffix}"
            table_df.insert(0, key_name, row_id)
            own_key[tid] = key_name

        if tid == "table_0_0":
            table_df[args.target_column] = df[args.target_column]

        table_df.to_csv(out_dir / f"{tid}.csv", index=False)

    connections = []
    for tid, _depth, parent_id in nodes:
        if parent_id is None:
            continue
        key = own_key[parent_id]
        connections.append({
            "fk_table": f"{parent_id}.csv", "fk_column": key,
            "pk_table": f"{tid}.csv", "pk_column": key,
        })
    pd.DataFrame(connections).to_csv(out_dir / "connections.csv", index=False)

    datasets_csv = Path(args.benchmark_dir) / "datasets.csv"
    datasets_df = pd.read_csv(datasets_csv)
    if args.name in set(datasets_df["base_table_label"]):
        print(f"'{args.name}' already present in {datasets_csv}, leaving it as-is.")
    else:
        target_dtype = df[args.target_column].dtype
        target_nunique = df[args.target_column].nunique()
        if target_nunique == 2:
            dataset_type = "binary"
        elif pd.api.types.is_numeric_dtype(target_dtype):
            dataset_type = "regression"
        else:
            raise ValueError(
                f"--target-column {args.target_column!r} has {target_nunique} non-numeric classes; "
                f"only binary classification or numeric regression targets are supported "
                f"(the paper's own benchmark datasets are all binary classification)"
            )
        new_row = pd.DataFrame([{
            "base_table_path": args.name, "base_table_name": "table_0_0.csv",
            "base_table_label": args.name, "target_column": args.target_column,
            "dataset_type": dataset_type,
        }])
        pd.concat([datasets_df, new_row], ignore_index=True).to_csv(datasets_csv, index=False)
        print(f"Appended '{args.name}' to {datasets_csv} (dataset_type={dataset_type!r}, "
              f"target dtype was {target_dtype})")

    print(f"\n{len(nodes)} tables written to {out_dir}/")
    for tid, depth, parent_id in nodes:
        cols = columns_per_table[tid]
        print(f"  {tid}.csv  (depth={depth}, parent={parent_id}, {len(cols)} feature cols)")
    print(f"connections.csv: {len(connections)} edges")
    print(f"\nRun:\n  uv run python src/feature_discovery/experiments/ablation.py --dataset {args.name}")

if __name__ == "__main__":
    main()
