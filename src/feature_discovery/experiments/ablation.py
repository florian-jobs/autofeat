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
    init_datasets()
    dataset = filter_datasets(["credit"])[0]
    autofeat(dataset,
             value_ratio=0.65,
             top_k=15,
             algorithm="LR",
             join_paths_df=load_join_paths("data/benchmark/credit/connections.csv"),
             lake_data_folder="data/benchmark/credit",
             base_table_sep=",")
