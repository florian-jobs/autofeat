from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import polars as pl

from feature_discovery.autofeat_pipeline.autofeat import AutoFeat
from feature_discovery.autofeat_pipeline.join_path_utils import get_path_length
from feature_discovery.experiments.dataset_object import CLASSIFICATION, REGRESSION
from feature_discovery.experiments.evaluate_join_paths import join_from_path, resolve_path_list
from feature_discovery.graph_processing.neo4j_transactions import clear_df_cache

class AutoFeatBaseline:
    def __init__(
            self,
            value_ratio: float = 0.65,
            top_k: int = 15,
            algorithm: str = "LR",
            sample_size: int = 3000,
    ) -> None:
        self.value_ratio = value_ratio
        self.top_k = top_k
        self.algorithm = algorithm
        self.sample_size = sample_size

    def _read_base_table(self, config, table_dir, base_table_sep):
        try:
            from beluga.online.base_table import read_base_table
        except ImportError:
            read_base_table = None

        if read_base_table is not None:
            polars_df = read_base_table(config.base_table, table_dir, config)
            return polars_df.to_pandas()

        explicit_name = getattr(config, "base_table_filename", None)
        if explicit_name:
            base_table_path = table_dir / explicit_name
        else:
            csv_files = sorted(
                p for p in Path(table_dir).glob("*.csv") if p.name != "join_paths.csv"
            )
            if len(csv_files) != 1:
                raise ValueError(
                    f"Cannot resolve base table file in {table_dir}: found {len(csv_files)} csv files"
                )
            base_table_path = csv_files[0]

        return pd.read_csv(
            base_table_path, header=0, engine="python", encoding="utf8", sep=base_table_sep,
            quotechar=chr(34), escapechar=chr(92),
        )

    def run(self, config=None):
        if config is None:
            raise ValueError("A Config instance is required")

        if not config.base_table:
            raise ValueError("config.base_table must be set")

        if config.queries_dir is None:
            raise ValueError("config.queries_dir must be set (external base table location)")

        if config.data_dir is None:
            raise ValueError("config.data_dir must be set (external lake corpus location)")

        table_dir = Path(config.queries_dir) / config.base_table
        lake_data_folder = Path(config.data_dir) / config.corpus
        base_table_sep = getattr(config, "base_table_sep", ",")

        base_table_df = self._read_base_table(config, table_dir, base_table_sep)

        target_column_id = getattr(config, "target_column_id", None)
        if target_column_id is not None:
            target_column = base_table_df.columns[target_column_id]
        else:
            target_column = base_table_df.columns[-1]

        downstream_task = getattr(config, "downstream_task", "classification")
        dataset_type = REGRESSION if downstream_task == "regression" else CLASSIFICATION

        # The rest of this pipeline (resolve_path_list in particular) assumes
        # every node id ends in ".csv" when reconstructing table names from
        # prefixed feature names, so the materialised base table node id must
        # follow that convention too, even though config.base_table itself may
        # not end in .csv.
        base_table_id = config.base_table + ".csv"
        clear_df_cache()
        lake_data_folder.mkdir(parents=True, exist_ok=True)
        base_table_df.to_csv(lake_data_folder / base_table_id, index=False, sep=base_table_sep)

        join_paths_df_path = getattr(config, "connections_csv_path", None) or str(table_dir / "join_paths.csv")
        join_paths_df = pd.read_csv(join_paths_df_path)

        bfs_traversal = AutoFeat(
            join_paths_df=join_paths_df,
            lake_data_folder=str(lake_data_folder),
            base_table_sep=base_table_sep,
            base_table_id=base_table_id,
            base_table_label=config.base_table,
            save_joins_to_disk=True,
            use_polars=True,
            target_column=target_column,
            value_ratio=self.value_ratio,
            top_k=self.top_k,
            sample_size=self.sample_size,
            task=dataset_type,
            pearson=False,
            jmi=False,
            no_redundancy=False,
            no_relevance=False,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bfs_traversal.streaming_feature_selection(
                join_paths_df=join_paths_df,
                lake_data_folder=str(lake_data_folder),
                lake_table_sep=base_table_sep,
                queue={base_table_id},
            )

            sorted_paths = sorted(
                bfs_traversal.ranking.items(),
                key=lambda r: (-float(r[1]), get_path_length(r[0]), r[0]),
            )
            join_name, _rank = sorted_paths[0]

            if join_name == bfs_traversal.base_table_id:
                return pl.from_pandas(base_table_df)

            path_list, features = resolve_path_list(bfs_traversal, join_name)
            dataframe = join_from_path(
                path_list,
                join_paths_df,
                str(lake_data_folder),
                base_table_sep,
                base_table_sep,
                bfs_traversal.base_table_id,
                bfs_traversal.target_column,
            )

        if dataframe is None:
            raise ValueError(f"Join failed for path: {join_name}")

        features = list(set(features).intersection(set(dataframe.columns)))
        if len(features) < 2:
            features = bfs_traversal.partial_join_selected_features[bfs_traversal.base_table_id]
            features.append(bfs_traversal.target_column)

        return pl.from_pandas(dataframe[features])
