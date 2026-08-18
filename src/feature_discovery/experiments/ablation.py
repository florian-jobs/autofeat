# Thesis-reproduction entry point: runs the AutoFeat pipeline against a data/benchmark/<dataset>
# table with a known KFK join graph (connections.csv) and writes per-path results to
# results/thesis/<dataset>_<approach>.csv, for comparing reproduced accuracy against the numbers
# reported in docs/assets/papers/ICDE_FeatureDiscovery.pdf. Separate from baseline.py, which is the
# beluga integration adapter and has no CLI of its own.
import os
import sys

# Must run before numpy/pandas are imported anywhere below (they trigger OpenBLAS's thread-pool
# sizing at import time): AutoGluon's model training spawns its own worker processes internally,
# and if each of those processes' BLAS calls grabs a thread pool sized to the core count, total
# demanded threads exceeds OpenBLAS's hardcoded 128-thread-context build limit on many-core
# servers (hit on hercules). Pinning to 1 thread per process forces all parallelism to come from
# AutoGluon's/joblib's own process fan-out instead of nested BLAS thread pools. `env.copy()` below
# (for the PYTHONHASHSEED relaunch) picks these up too, so the same fix reaches the child process.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# `os.environ['PYTHONHASHSEED'] = '42'` elsewhere in this codebase (autofeat.py,
# evaluation_algorithms.py) is a no-op for the *current* process: CPython reads PYTHONHASHSEED
# exactly once, at interpreter startup, to seed str/set/dict hash randomization. Setting it from
# inside an already-running process has zero effect on that process's hashes. Since BFS traversal
# uses sets (self.discovered, queue, all_neighbours) whose iteration order depends on those
# hashes - and connections.csv gives every edge weight=1, so tie-breaking leans heavily on that
# order - every invocation was picking a randomly different traversal order, changing which join
# path got ranked/trained as "best" from run to run (confirmed empirically: ranking scores for
# credit differed between runs without this). Fix: re-launch with PYTHONHASHSEED set in the real
# environment before Python starts, which is the only place it actually takes effect. Uses
# subprocess rather than os.execvpe(..) for the re-launch: process-image replacement via execvpe
# was found to segfault under uv's Windows launcher.
#
# Default seed is 42, but an already-set PYTHONHASHSEED in the environment is honoured instead of
# being overridden -- lets you explore how BFS tie-breaking (and thus which join path/how many
# tables get chained) changes with the seed, e.g. `PYTHONHASHSEED=7 uv run python ablation.py ...`.
if "PYTHONHASHSEED" not in os.environ:
    import subprocess

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "42"
    result = subprocess.run([sys.executable] + sys.argv, env=env)
    sys.exit(result.returncode)

import logging
import time
from typing import Tuple, List

import pandas as pd

from feature_discovery.autofeat_pipeline.autofeat import AutoFeat
from feature_discovery.config import RESULTS_FOLDER
from feature_discovery.experiments.dataset_object import Dataset
from feature_discovery.experiments.evaluate_join_paths import evaluate_paths
from feature_discovery.experiments.init_datasets import init_datasets
from feature_discovery.experiments.result_object import Result
from feature_discovery.experiments.utils_dataset import filter_datasets

def load_join_paths(connections_csv_path: str) -> pd.DataFrame:
    """Read a benchmark-setting connections.csv (fk/pk columns) into the from_id/to_id shape AutoFeat expects."""
    df = pd.read_csv(connections_csv_path)
    df = df.rename(columns={
        "fk_table": "from_id",
        "fk_column": "from_column",
        "pk_table": "to_id",
        "pk_column": "to_column",
    })
    df["weight"] = 1
    return df

def autofeat(
        dataset: Dataset,
        join_paths_df: pd.DataFrame,
        lake_data_folder: str,
        base_table_sep: str,
        value_ratio: float,
        top_k: int,
        algorithm: str,
        approach: str = Result.TFD,
        pearson: bool = False,
        jmi: bool = False,
        no_relevance: bool = False,
        no_redundancy: bool = False,
        save_joins_to_disk: bool = True,
        use_polars: bool = True,
        sample_size: int = 3000,
) -> Tuple[List[Result], List[Tuple], List[str]]:
    """
    Run AutoFeat's BFS join-path discovery for `dataset`, then evaluate the top-k ranked paths with
    `algorithm` (see evaluation_algorithms.get_hyperparameters for supported values). Writes one row
    per evaluated path to results/thesis/<dataset.base_table_label>_<approach>.csv and returns the
    same results alongside the ranked path list and the union of selected features.
    """
    logging.debug(f"Running on TFD (Transitive Feature Discovery) result with AutoGluon")

    start = time.time()
    bfs_traversal = AutoFeat(
        join_paths_df=join_paths_df,
        lake_data_folder=lake_data_folder,
        base_table_sep=base_table_sep,
        base_table_id=dataset.base_table_name,
        base_table_label=dataset.base_table_label,
        save_joins_to_disk=save_joins_to_disk,
        use_polars=use_polars,
        target_column=dataset.target_column,
        value_ratio=value_ratio,
        top_k=top_k,
        sample_size=sample_size,
        task=dataset.dataset_type,
        pearson=pearson,
        jmi=jmi,
        no_redundancy=no_redundancy,
        no_relevance=no_relevance,
    )
    bfs_traversal.streaming_feature_selection(join_paths_df=join_paths_df,
                                              lake_data_folder=lake_data_folder,
                                              lake_table_sep=base_table_sep,
                                              queue={dataset.base_table_name})
    end = time.time()

    logging.debug(f"FINISHED {approach}")

    all_results, top_k_paths, selected_features = evaluate_paths(bfs_result=bfs_traversal,
                                                                 problem_type=dataset.dataset_type,
                                                                 algorithm=algorithm,
                                                                 join_paths_df=join_paths_df,
                                                                 lake_data_folder=lake_data_folder,
                                                                 lake_table_sep=base_table_sep,
                                                                 base_table_sep=base_table_sep,
                                                                 )

    for result in all_results:
        result.approach = approach
        result.feature_selection_time = end - start
        result.total_time += result.feature_selection_time
        result.top_k = top_k
        result.data_label = dataset.base_table_label
        result.cutoff_threshold = value_ratio

    logging.debug("Save results ... ")
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)  # not guaranteed to exist on a fresh clone/pull
    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / f"{dataset.base_table_label}_{approach}.csv", index=False)

    return all_results, top_k_paths, selected_features

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run AutoFeat's ablation pipeline against a data/benchmark/<dataset> table.")
    parser.add_argument("--dataset", default="credit", help="base_table_label in data/benchmark/datasets.csv")
    parser.add_argument("--value-ratio", type=float, default=0.65)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--algorithm", default="LR")
    parser.add_argument("--sample-size", type=int, default=3000,
                        help="Rows sampled for BFS relevance/redundancy scoring (paper default: 3000). "
                             "Only affects path/feature ranking, not final model training, which always "
                             "uses the full joined table. Raising this can matter for large datasets "
                             "(e.g. covertype) where 3000 rows is a very small fraction of the base table.")
    args = parser.parse_args()

    init_datasets()
    matches = filter_datasets([args.dataset])
    if not matches:
        raise SystemExit(
            f"--dataset {args.dataset!r} not found in data/benchmark/datasets.csv "
            f"(build it first with build_benchmark_dataset.py, or check the spelling)"
        )
    dataset = matches[0]
    autofeat(dataset,
             value_ratio=args.value_ratio,
             top_k=args.top_k,
             algorithm=args.algorithm,
             sample_size=args.sample_size,
             join_paths_df=load_join_paths(f"data/benchmark/{args.dataset}/connections.csv"),
             lake_data_folder=f"data/benchmark/{args.dataset}",
             base_table_sep=",")
