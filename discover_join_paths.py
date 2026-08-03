"""
Discover a join_paths.csv for one base table against a (possibly huge) corpus, without Neo4j and
without scanning the whole corpus: Valentine schema-matches the base table against up to --limit
candidate tables found under --corpora, keeps matches on --query_column above --threshold, and
stages everything baseline.py/test_baseline.py need (a queries/<base_table>/ dir with the base table
CSV + join_paths.csv, and a corpus/<base_table>_lake/ dir with the matched tables).

Example:
    uv run python discover_join_paths.py \\
        --corpora ~/data/corpora/open_data/joinable_tables/ --limit 5 \\
        --input ~/data/corpora/open_data/joinable_tables/nyc/nyc-finance-39g5-gbp3/table.csv \\
        --query_column agency_name --target_column total_current_budget_amount
"""
import argparse
import shutil
import warnings
from pathlib import Path

import pandas as pd
from valentine import valentine_match
from valentine.algorithms import Coma


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the base table CSV")
    parser.add_argument("--corpora", required=True, help="Root directory to search for candidate lake tables")
    parser.add_argument("--query_column", required=True, help="Base table column to find joins on")
    parser.add_argument("--target_column", required=True, help="Base table target/label column")
    parser.add_argument("--limit", type=int, default=20, help="Max number of candidate tables to schema-match against")
    parser.add_argument("--threshold", type=float, default=0.55, help="Minimum Valentine similarity to accept a match")
    parser.add_argument("--base-table-id", default=None, help="Id for the base table (default: --input's parent folder name)")
    parser.add_argument("--queries-dir", default="tmp/queries", help="Where to stage <base_table>/ (base table CSV + join_paths.csv)")
    parser.add_argument("--data-dir", default="tmp/corpus", help="Where to stage the matched lake tables")
    return parser.parse_args()


def find_candidates(corpora_dir: Path, input_path: Path, limit: int):
    candidates = sorted(
        p for p in corpora_dir.rglob("*.csv") if p.resolve() != input_path.resolve()
    )
    return candidates[:limit]


def table_id_for(csv_path: Path) -> str:
    # Real corpora tend to store one CSV per uniquely-named folder (e.g. .../nyc-finance-39g5-gbp3/table.csv);
    # the folder name is the meaningful, unique id. Falls back to the file's own stem if it's already unique.
    return csv_path.parent.name or csv_path.stem


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser()
    corpora_dir = Path(args.corpora).expanduser()
    base_table_id = args.base_table_id or table_id_for(input_path)

    base_df = pd.read_csv(input_path, encoding="utf8")
    if args.query_column not in base_df.columns:
        raise ValueError(f"--query_column {args.query_column!r} not found in {input_path} (columns: {list(base_df.columns)})")
    if args.target_column not in base_df.columns:
        raise ValueError(f"--target_column {args.target_column!r} not found in {input_path} (columns: {list(base_df.columns)})")
    target_column_id = list(base_df.columns).index(args.target_column)

    candidates = find_candidates(corpora_dir, input_path, args.limit)
    print(f"Matching {input_path} ({args.query_column!r}) against {len(candidates)} candidate tables under {corpora_dir} ...")

    matches = []  # (candidate_table_id, candidate_path, to_column, similarity)
    for candidate_path in candidates:
        candidate_id = table_id_for(candidate_path)
        try:
            candidate_df = pd.read_csv(candidate_path, encoding="utf8")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                match_result = valentine_match(base_df, candidate_df, Coma(strategy="COMA_OPT"))
        except Exception as e:
            print(f"  skip {candidate_id}: {e}")
            continue

        best = None
        for ((_, col_from), (_, col_to)), similarity in match_result.items():
            if col_from != args.query_column or similarity < args.threshold:
                continue
            if best is None or similarity > best[1]:
                best = (col_to, similarity)

        if best is not None:
            to_column, similarity = best
            print(f"  match: {candidate_id}.{to_column} (similarity={similarity:.2f})")
            matches.append((candidate_id, candidate_path, to_column, similarity))

    if not matches:
        print("No matches found above threshold; no join_paths.csv written.")
        return

    queries_table_dir = Path(args.queries_dir) / base_table_id
    corpus_dir = Path(args.data_dir) / f"{base_table_id}_lake"
    queries_table_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(input_path, queries_table_dir / input_path.name)
    for candidate_id, candidate_path, _, _ in matches:
        shutil.copy(candidate_path, corpus_dir / f"{candidate_id}.csv")

    base_table_node_id = base_table_id + ".csv"
    join_paths_df = pd.DataFrame([
        {
            "from_id": base_table_node_id,
            "from_column": args.query_column,
            "to_id": candidate_id + ".csv",
            "to_column": to_column,
            "weight": 1,
        }
        for candidate_id, _, to_column, _ in matches
    ])
    join_paths_path = queries_table_dir / "join_paths.csv"
    join_paths_df.to_csv(join_paths_path, index=False)

    print(f"\n{len(matches)} join path(s) written to {join_paths_path}")
    print(f"Base table staged at: {queries_table_dir / input_path.name}")
    print(f"Matched lake tables staged at: {corpus_dir}")
    print("\nRun:")
    print(
        f"  python test_baseline.py --queries-dir {args.queries_dir} --data-dir {args.data_dir} "
        f"--corpus {base_table_id}_lake --base-table {base_table_id} --target-column-id {target_column_id}"
    )


if __name__ == "__main__":
    main()
