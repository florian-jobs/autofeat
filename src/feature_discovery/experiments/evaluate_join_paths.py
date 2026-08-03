from __future__ import annotations
from feature_discovery.autofeat_pipeline.autofeat import AutoFeat
from feature_discovery.autofeat_pipeline.join_path_utils import get_path_length
from feature_discovery.experiments.evaluation_algorithms import evaluate_all_algorithms
from feature_discovery.graph_processing.neo4j_transactions import get_df_with_prefix
from typing import Tuple, List

import logging
import os
import pickle
import random

import numpy as np
import pandas as pd
import tqdm

try:
    from sklearnex import patch_sklearn

    logging.getLogger("sklearnex").disabled = True
    patch_sklearn()
except ImportError:
    pass

def resolve_path_list(bfs_result: AutoFeat, join_name: str):
    features = list(bfs_result.partial_join_selected_features[join_name])
    features.append(bfs_result.target_column)
    features.extend(bfs_result.partial_join_selected_features[bfs_result.base_table_id])  # base features
    logging.debug(f"Feature before join_key removal:\n{features}")
    # features = list((set(features) - set(bfs_result.join_keys[join_name])).intersection(set(dataframe.columns)))
    # logging.debug(f"Feature after join_key removal:\n{features}")

    features_tables = sorted({f"{feat.split('.csv')[0]}.csv" for feat in features})

    # to_table -> (from_table, from_column, to_column, to_table), one entry per hop in the join path
    path_tables = {}
    for p in join_name.split("--"):
        aux = p.split("-")
        if len(aux) == 4:
            path_tables[aux[3]] = (aux[0], aux[1], aux[2], aux[3])

    path_list = build_hop_list(features_tables, path_tables)

    return path_list, features


def build_hop_list(tables, path_tables):
    """
    Resolve an ordered, deduplicated list of join hops (from_table, from_column, to_column, to_table)
    needed to reach every table in `tables`, starting from the base table. Hops are ordered so that
    a hop's `from_table` has always already been joined (or is the base table) by the time it's applied.
    """
    hops = []
    seen_tables = set()
    for table in tables:
        chain = []
        node = table
        while node in path_tables:
            hop = path_tables[node]
            chain.append(hop)
            node = hop[0]
        chain.reverse()  # root (nearest the base table) first
        for hop in chain:
            to_table = hop[3]
            if to_table not in seen_tables:
                hops.append(hop)
                seen_tables.add(to_table)
    return hops

def evaluate_paths(bfs_result: AutoFeat, problem_type: str, algorithm: str, join_paths_df: pd.DataFrame,
                   lake_data_folder: str, lake_table_sep: str, base_table_sep: str, top_k_paths: int = 15,
                   budget_clock=None, partial_state_path: str | None = None) -> Tuple[List, List[Tuple]]:
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'

    logging.debug(f"Evaluate top-{top_k_paths} paths ... ")
    # Sort with explicit ordering for deterministic results:
    # 1. Higher rank first (-rank)
    # 2. Shorter paths first (length, not negated)
    # 3. Alphabetically by path name for stability
    sorted_paths = sorted(
        bfs_result.ranking.items(),
        key=lambda r: (-float(r[1]), get_path_length(r[0]), r[0])
    )
    top_k_path_list = sorted_paths if len(sorted_paths) < top_k_paths else sorted_paths[:top_k_paths]
    base_features = bfs_result.partial_join_selected_features[bfs_result.base_table_id]

    all_results = []
    selected_features = set()
    print(f"Evaluating {top_k_path_list} paths...")
    for path in tqdm.tqdm(top_k_path_list):
        # Cooperative budget check at iteration boundary. Once the wall-clock
        # deadline has been reached the loop exits with whatever results were
        # accumulated so far. Catches the common case where the SIGALRM hard
        # cap is delayed by a long C-extension call inside the previous
        # iteration's ``evaluate_all_algorithms``.
        if budget_clock is not None and getattr(budget_clock, 'expired', False):
            print(f"[evaluate_paths] budget exhausted; stopping after {len(all_results)} paths", flush=True)
            break
        join_name, rank = path
        if join_name == bfs_result.base_table_id:
            continue

        path_list, features = resolve_path_list(bfs_result, join_name)

        # dataframe = join_from_path(path_list, bfs_result.target_column, bfs_result.base_table_id)
        dataframe = join_from_path(path_list, join_paths_df, lake_data_folder, lake_table_sep, base_table_sep,
                                   bfs_result.base_table_id, bfs_result.target_column)

        # Skip if join failed to produce a valid dataframe
        if dataframe is None:
            continue

        dataframe_columns = set(dataframe.columns)
        features = list(dict.fromkeys(f for f in features if f in dataframe_columns))

        if len(features) < 2:
            features = list(bfs_result.partial_join_selected_features[bfs_result.base_table_id])
            features.append(bfs_result.target_column)

        results, _ = evaluate_all_algorithms(dataframe=dataframe[features],
                                             target_column=bfs_result.target_column,
                                             problem_type=problem_type,
                                             algorithm=algorithm)
        for result in results:
            result.rank = rank
            result.data_path = path_list
            for f in result.join_path_features:
                selected_features.add(f)
        print('Results for path:', results)
        all_results.extend(results)

        # Persist after each iteration so an external killer (SIGALRM, SIGTERM,
        # OOM, or a hard-deadline watchdog) leaves usable partial state behind.
        if partial_state_path is not None:
            try:
                with open(partial_state_path, 'wb') as _f:
                    pickle.dump({
                        'all_results': all_results,
                        'selected_features': list(selected_features),
                        'top_k_path_list': top_k_path_list,
                    }, _f)
            except Exception:
                pass

        dataframe = None

    selected_features = list(selected_features)

    return all_results, top_k_path_list, selected_features

def join_from_path(path: list[tuple], join_paths_df, lake_data_folder: str, lake_table_sep: str, base_table_sep: str,
                   base_node: str, target: str = None):
    """
    `path` is an ordered list of (from_table, from_column, to_column, to_table) hops, as produced by
    `build_hop_list`. Each hop's `from_table` is either `base_node` or a table joined by an earlier hop.
    """
    joined_df, _ = get_df_with_prefix(
        join_paths_df,
        lake_data_folder,
        base_node,
        base_table_sep,
        target
    )

    for from_table, from_column, to_column, to_table in path:
        try:
            right_table, _ = get_df_with_prefix(join_paths_df, lake_data_folder, to_table, lake_table_sep)

            # Resolve placeholder column names (col_0, col_1, etc.) to actual column names
            if to_column.startswith('col_'):
                try:
                    col_index = int(to_column.split('_')[1])
                    actual_columns = [col.replace(f"{to_table}.", "", 1) for col in right_table.columns if
                                      col.startswith(f"{to_table}.")]
                    if col_index < len(actual_columns):
                        to_column = actual_columns[col_index]
                except (ValueError, IndexError):
                    pass

            if from_column.startswith('col_'):
                try:
                    col_index = int(from_column.split('_')[1])
                    actual_columns = [col.replace(f"{from_table}.", "", 1) for col in joined_df.columns if
                                      col.startswith(f"{from_table}.")]
                    if col_index < len(actual_columns):
                        from_column = actual_columns[col_index]
                except (ValueError, IndexError):
                    pass

            right_table = right_table.groupby(f'{to_table}.{to_column}').sample(n=1, random_state=42)

            joined_df = pd.merge(
                joined_df,
                right_table,
                how="left",
                left_on=f'{from_table}.{from_column}',
                right_on=f'{to_table}.{to_column}'
            )
        except Exception:
            continue

    return joined_df
