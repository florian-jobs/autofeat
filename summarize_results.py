"""
Summarize results/thesis/<dataset>_<approach>.csv files into one row per dataset+algorithm:
the best (highest-accuracy) evaluated join path, with how many tables it joined. Meant as the
local half of comparing against the AutoFeat paper's numbers (docs/assets/papers/ICDE_FeatureDiscovery.pdf,
Table II for OpenML ceiling references, Figures 4-7 for AutoFeat's own reported accuracy per
dataset/algorithm).

Usage:
    uv run python summarize_results.py [--results-dir results/thesis]
"""
import argparse
import ast
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results/thesis")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    csv_files = sorted(results_dir.glob("*_AutoFeat.csv"))
    if not csv_files:
        print(f"No *_AutoFeat.csv files found in {results_dir}")
        return

    rows = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        for algorithm, group in df.groupby("algorithm"):
            best = group.loc[group["accuracy"].idxmax()]
            path = ast.literal_eval(best["data_path"]) if isinstance(best["data_path"], str) else []
            tables_joined = len({hop[3] for hop in path}) + 1 if path else 1  # +1 for the base table
            rows.append({
                "dataset": best["data_label"],
                "algorithm": algorithm,
                "best_accuracy": round(best["accuracy"], 4),
                "tables_joined": tables_joined,
                "rank": round(best["rank"], 4) if pd.notna(best["rank"]) else None,
            })

    summary = pd.DataFrame(rows).sort_values(["dataset", "algorithm"])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
