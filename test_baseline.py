import argparse
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from baseline import AUTOFEATBaseline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AUTOFEATBaseline against a queries/corpus directory (local fixture or real server layout)."
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
    parser.add_argument("--value-ratio", type=float, default=0.65, help="Pruning threshold: min ratio of non-null values a joined column must have (paper default 0.65)")
    parser.add_argument("--top-k", type=int, default=15, help="Max join-path candidates kept ranked (paper default 15, kappa in the paper)")
    parser.add_argument("--sample-size", type=int, default=3000, help="Rows sampled for BFS relevance/redundancy scoring (paper default 3000). Raise this for large real-world base tables where 3000 rows may not be representative")
    parser.add_argument("--verbose", action="store_true", help="Print BFS candidate counts, the chosen join path's rank vs. runner-ups, and phase timing -- useful for sanity-checking a run against a large/unfamiliar corpus")
    parser.add_argument("--min-rows", type=int, default=None, help="Exit non-zero if the returned DataFrame has fewer rows than this -- catches a join silently collapsing/dropping most of the base table")
    parser.add_argument("--max-null-ratio", type=float, default=None, help="Exit non-zero if any joined-in column's null ratio exceeds this (0-1) -- catches a join that 'succeeded' but only ever matched empty/NaN cells")
    return parser.parse_args()


def limit_join_paths(join_paths_path: Path, base_table_id: str, limit: int) -> Path:
    """Overwrites join_paths.csv in place, restricted to a BFS-bounded subgraph of at most `limit`
    tables, after backing up the original alongside it (join_paths.csv.bak).

    baseline.py now reads join_paths.csv from a fixed path (table_dir / "join_paths.csv"), matching
    ARDABaseline/QCRBaseline/COCOABaseline, so there's no config field left to point it at a side
    file - overwriting in place (like a real Config would need to see) is the only way to keep this
    --limit sampling feature working. Returns the backup path; restore_join_paths() undoes this.
    """
    join_paths_df = pd.read_csv(join_paths_path)

    visited = {base_table_id}
    frontier = {base_table_id}
    while frontier and len(visited) < limit:
        neighbours = set(join_paths_df.loc[join_paths_df["from_id"].isin(frontier), "to_id"])
        neighbours |= set(join_paths_df.loc[join_paths_df["to_id"].isin(frontier), "from_id"])
        frontier = (neighbours - visited)
        if len(visited) + len(frontier) > limit:
            frontier = set(sorted(frontier)[: limit - len(visited)])
        visited |= frontier

    limited_df = join_paths_df[
        join_paths_df["from_id"].isin(visited) & join_paths_df["to_id"].isin(visited)
    ]
    backup_path = join_paths_path.with_suffix(join_paths_path.suffix + ".bak")
    shutil.copyfile(join_paths_path, backup_path)
    limited_df.to_csv(join_paths_path, index=False)
    print(f"Limited join graph to {len(visited)} tables ({len(limited_df)} of {len(join_paths_df)} edges) "
          f"-> {join_paths_path} (original backed up to {backup_path})")
    return backup_path


def restore_join_paths(join_paths_path: Path, backup_path: Path) -> None:
    """Undoes limit_join_paths(): restores the original join_paths.csv from its backup."""
    shutil.move(backup_path, join_paths_path)


def check_correctness(result: pd.DataFrame, base_table: str, args) -> bool:
    """Print row/column/null-coverage stats and apply --min-rows / --max-null-ratio thresholds if given.

    Returns False (and the caller should exit non-zero) if a threshold was violated.
    """
    ok = True
    n_rows, n_cols = result.shape
    joined_columns = [c for c in result.columns if not c.startswith(f"{base_table}.csv.")]
    print(f"\n[check] {n_rows} rows, {n_cols} columns ({len(joined_columns)} joined in from lake tables)")

    if args.min_rows is not None and n_rows < args.min_rows:
        print(f"[check] FAIL: {n_rows} rows < --min-rows {args.min_rows}")
        ok = False

    if joined_columns:
        null_ratios = result[joined_columns].isna().mean().sort_values(ascending=False)
        worst = null_ratios.head(5)
        print("[check] highest-null joined columns:")
        for col, ratio in worst.items():
            print(f"[check]   {ratio:6.1%}  {col}")
        if args.max_null_ratio is not None:
            offenders = null_ratios[null_ratios > args.max_null_ratio]
            if len(offenders) > 0:
                print(f"[check] FAIL: {len(offenders)} joined column(s) exceed --max-null-ratio {args.max_null_ratio}")
                ok = False

    return ok


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

    join_paths_backup = None
    if args.limit is not None:
        join_paths_path = Path(args.queries_dir) / args.base_table / "join_paths.csv"
        base_table_id = args.base_table + ".csv"
        join_paths_backup = limit_join_paths(join_paths_path, base_table_id, args.limit)

    baseline = AUTOFEATBaseline(
        value_ratio=args.value_ratio,
        top_k=args.top_k,
        sample_size=args.sample_size,
        verbose=args.verbose,
    )
    try:
        result = baseline.run(config)
    finally:
        if join_paths_backup is not None:
            restore_join_paths(Path(args.queries_dir) / args.base_table / "join_paths.csv", join_paths_backup)

    try:
        print(result)
    except UnicodeEncodeError:
        # Windows console (cp1252) can't display some Unicode content in the augmented data --
        # the pipeline already succeeded above, this is only the diagnostic print. Not an issue on
        # the actual (Linux, UTF-8) production/beluga environment.
        enc = sys.stdout.encoding or "ascii"
        print(str(result).encode(enc, errors="replace").decode(enc))

    if args.min_rows is not None or args.max_null_ratio is not None or args.verbose:
        ok = check_correctness(result.to_pandas(), args.base_table, args)
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
