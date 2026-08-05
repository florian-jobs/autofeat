# Thesis-reproduction entry point: runs the AutoFeat pipeline against a data/benchmark/<dataset>
# table with a known KFK join graph (connections.csv) and writes per-path results to
# results/thesis/<dataset>_<approach>.csv, for comparing reproduced accuracy against the numbers
# reported in docs/assets/papers/ICDE_FeatureDiscovery.pdf. Separate from baseline.py, which is the
# beluga integration adapter and has no CLI of its own.
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
             join_paths_df=load_join_paths(f"data/benchmark/{args.dataset}/connections.csv"),
             lake_data_folder=f"data/benchmark/{args.dataset}",
             base_table_sep=",")
