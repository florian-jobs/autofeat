"""
Run AutoFeat with different PYTHONHASHSEED values and check how much the best join path changes.

Rsel in Algorithm 1 is shared over the whole traversal (Rsel = Rsel U Rred), same as in upstream
autofeat. So the redundancy scores depend on the order tables are visited, which depends on set
iteration order, i.e. on PYTHONHASHSEED. This is how the algorithm works, not a bug, so we just
measure the variance.

Modes:
  --mode bfs   (default): only the BFS ranking, no training. Logs depth of the top candidate per seed.
  --mode full  full ablation.py run (BFS + training, best accuracy wins). Logs tables_joined and
               accuracy and keeps results/thesis/<dataset>_AutoFeat.csv for each seed. Slow.

Usage:
    uv run python sweep_hashseed.py --dataset bioresponse --seeds 0-29
    uv run python sweep_hashseed.py --dataset bioresponse --seeds 42,1000,7,123 --mode full
"""
import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_seeds(spec: str):
    seeds = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part and not part.startswith("-"):
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def run_bfs_worker(argv):
    """Runs in a subprocess with PYTHONHASHSEED set, prints a SWEEP_RESULT <json> line."""
    sys.path.insert(0, "src")
    from feature_discovery.experiments.ablation import load_join_paths
    from feature_discovery.experiments.init_datasets import init_datasets
    from feature_discovery.experiments.utils_dataset import filter_datasets
    from feature_discovery.autofeat_pipeline.autofeat import AutoFeat
    from feature_discovery.autofeat_pipeline.join_path_utils import get_path_length

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--value-ratio", type=float, default=0.65)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--sample-size", type=int, default=3000)
    args = parser.parse_args(argv)

    init_datasets()
    dataset = filter_datasets([args.dataset])[0]
    join_paths_df = load_join_paths(f"data/benchmark/{args.dataset}/connections.csv")
    lake_data_folder = f"data/benchmark/{args.dataset}"

    bfs = AutoFeat(
        join_paths_df=join_paths_df,
        lake_data_folder=lake_data_folder,
        base_table_sep=",",
        base_table_id=dataset.base_table_name,
        base_table_label=dataset.base_table_label,
        save_joins_to_disk=True,
        use_polars=True,
        target_column=dataset.target_column,
        value_ratio=args.value_ratio,
        top_k=args.top_k,
        sample_size=args.sample_size,
        task=dataset.dataset_type,
    )
    bfs.streaming_feature_selection(
        join_paths_df=join_paths_df,
        lake_data_folder=lake_data_folder,
        lake_table_sep=",",
        queue={dataset.base_table_name},
    )

    if not bfs.ranking:
        print("SWEEP_RESULT " + json.dumps({"n_candidates": 0}))
        return

    sorted_paths = sorted(bfs.ranking.items(), key=lambda r: (-float(r[1]), get_path_length(r[0]), r[0]))
    top_k_list = sorted_paths[: args.top_k]
    result = {
        "n_candidates": len(bfs.ranking),
        "best_depth": get_path_length(sorted_paths[0][0]),
        "best_rank": sorted_paths[0][1],
        "top_k_depths": [get_path_length(name) for name, _ in top_k_list],
    }
    print("SWEEP_RESULT " + json.dumps(result))


def sweep_bfs(dataset, seeds, value_ratio, top_k, sample_size):
    rows = []
    worker_argv = [
        "--dataset", dataset,
        "--value-ratio", str(value_ratio),
        "--top-k", str(top_k),
        "--sample-size", str(sample_size),
    ]
    for seed in seeds:
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        proc = subprocess.run(
            [sys.executable, __file__, "--_worker-bfs", *worker_argv],
            env=env, capture_output=True, text=True,
        )
        line = next((l for l in proc.stdout.splitlines() if l.startswith("SWEEP_RESULT ")), None)
        if line is None:
            print(f"seed={seed}: worker produced no result (exit {proc.returncode})\n{proc.stderr[-2000:]}",
                  file=sys.stderr)
            continue
        data = json.loads(line[len("SWEEP_RESULT "):])
        data["seed"] = seed
        rows.append(data)
        if data.get("n_candidates", 0) == 0:
            print(f"seed={seed:>6}  no candidates discovered")
        else:
            depths = data["top_k_depths"]
            print(f"seed={seed:>6}  best_depth={data['best_depth']:>3}  best_rank={data['best_rank']:.4f}  "
                  f"n_candidates={data['n_candidates']:>3}  top_k_depths={depths}")
    return pd.DataFrame(rows)


def sweep_full(dataset, seeds, algorithm, top_k, value_ratio, sample_size, results_dir):
    rows = []
    archive_dir = results_dir / "hashseed_sweep" / dataset
    archive_dir.mkdir(parents=True, exist_ok=True)
    results_csv = results_dir / f"{dataset}_AutoFeat.csv"

    for seed in seeds:
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        print(f"seed={seed}: running full ablation.py ...", flush=True)
        proc = subprocess.run(
            [sys.executable, "src/feature_discovery/experiments/ablation.py",
             "--dataset", dataset, "--algorithm", algorithm, "--top-k", str(top_k),
             "--value-ratio", str(value_ratio), "--sample-size", str(sample_size)],
            env=env,
        )
        if proc.returncode != 0 or not results_csv.exists():
            print(f"seed={seed}: ablation.py failed (exit {proc.returncode})", file=sys.stderr)
            continue

        df = pd.read_csv(results_csv)
        if df.empty:
            print(f"seed={seed}: empty results", file=sys.stderr)
            continue

        best = df.loc[df["accuracy"].idxmax()]
        path = ast.literal_eval(best["data_path"]) if isinstance(best["data_path"], str) else []
        tables_joined = len({hop[3] for hop in path}) + 1 if path else 1
        rows.append({
            "seed": seed,
            "best_accuracy": round(float(best["accuracy"]), 4),
            "tables_joined": tables_joined,
            "rank": round(float(best["rank"]), 4) if pd.notna(best["rank"]) else None,
        })
        print(f"seed={seed:>6}  best_accuracy={best['accuracy']:.4f}  tables_joined={tables_joined:>3}  "
              f"rank={best['rank']:.4f}")

        shutil.copy(results_csv, archive_dir / f"seed{seed}.csv")

    return pd.DataFrame(rows)


def main():
    if "--_worker-bfs" in sys.argv:
        idx = sys.argv.index("--_worker-bfs")
        return run_bfs_worker(sys.argv[idx + 1:])

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="bioresponse")
    parser.add_argument("--seeds", default="0-19", help='e.g. "0-19" or "42,1000,7"')
    parser.add_argument("--mode", choices=["bfs", "full"], default="bfs")
    parser.add_argument("--algorithm", default="LR")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--value-ratio", type=float, default=0.65)
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--results-dir", default="results/thesis")
    parser.add_argument("--out", default=None, help="CSV path to save sweep results")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    print(f"Sweeping {len(seeds)} seed(s) over dataset={args.dataset!r} mode={args.mode!r} ...\n")

    if args.mode == "bfs":
        df = sweep_bfs(args.dataset, seeds, args.value_ratio, args.top_k, args.sample_size)
        if not df.empty and "best_depth" in df.columns:
            print("\n=== best_depth summary across seeds ===")
            print(df["best_depth"].describe().to_string())
    else:
        results_dir = Path(args.results_dir)
        df = sweep_full(args.dataset, seeds, args.algorithm, args.top_k, args.value_ratio,
                        args.sample_size, results_dir)
        if not df.empty:
            print("\n=== tables_joined summary across seeds ===")
            print(df["tables_joined"].describe().to_string())
            print("\n=== best_accuracy summary across seeds ===")
            print(df["best_accuracy"].describe().to_string())

    out = args.out or f"results/hashseed_sweep_{args.dataset}_{args.mode}.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} row(s) to {out}")


if __name__ == "__main__":
    main()
