"""
Beluga entry point for the AutoFeat baseline: wires the Neo4j-decoupled AutoFeat pipeline
(streaming_feature_selection over join_paths.csv, then evaluate_join_paths) behind beluga's Config
object, so beluga can run AutoFeat without knowing anything about its internals. Shaped to match the
sibling QCRBaseline/ARDABaseline contract (duck-typed Config in, polars.DataFrame out), with the
deviations noted inline where AutoFeat's needs differ - e.g. it validates AND uses target_column_id,
where the siblings validate it but always use the last column instead.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from importlib import resources

import pandas as pd
import polars as pl

from feature_discovery.autofeat_pipeline.autofeat import AutoFeat
from feature_discovery.autofeat_pipeline.join_path_utils import get_path_length
from feature_discovery.experiments.dataset_object import CLASSIFICATION, REGRESSION
from feature_discovery.experiments.evaluate_join_paths import join_from_path, resolve_path_list
from feature_discovery.graph_processing.neo4j_transactions import clear_df_cache

class AutoFeatBaseline:
    """Runs join-path discovery + feature selection for one beluga Config and returns the augmented base table."""

    def __init__(
            self,
            value_ratio: float = 0.65,
            top_k: int = 15,
            algorithm: str = "LR",
            sample_size: int = 3000,
            verbose: bool = False,
    ) -> None:
        self.value_ratio = value_ratio
        self.top_k = top_k
        self.algorithm = algorithm
        self.sample_size = sample_size
        # Opt-in diagnostics (candidate counts, chosen path, timing) printed to stdout.
        # Off by default so beluga's own calls (which never pass this) are unaffected.
        self.verbose = verbose

    def _read_base_table(self, config, table_dir, base_table_sep):
        """Prefer beluga's own reader; fall back to a plain CSV read when beluga isn't importable (dev/testing)."""
        try:
            from beluga.online.base_table import read_base_table
        except ImportError:
            read_base_table = None

        # read base table with beluga if possible
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
        """Duck-typed run(config) -> polars.DataFrame; raises ValueError on any invalid/unresolvable input."""
        if config is None:  # Mirrors: config = Config() if config is None else config
            raise ValueError("A Config instance is required")

        if not config.base_table:
            raise ValueError("config.base_table must be set")

        if config.target_column_id is None:
            raise ValueError("Value for target_column_id not specified in the configuration file")

        # extract relevant config parameters from beluga config
        if config.queries_dir is not None:
            table_dir = Path(config.queries_dir) / config.base_table
        else:
            table_dir = resources.files("beluga.data").joinpath(
                "queries/beers")  # to update with a new default base table

        if config.data_dir is not None:
            lake_data_folder = Path(config.data_dir) / config.corpus
        else:
            lake_data_folder = resources.files("beluga.data").joinpath("corpora/toy")

        base_table_sep = getattr(config, "base_table_sep", ",")  # Needs checking. Not found in qcr/arda baselines.

        # Mirrors: read_base_table(config.base_table, table_dir, config)
        base_table_df = self._read_base_table(config, table_dir, base_table_sep)

        # Unlike ARDA/QCR (which validate target_column_id is set but then always use the last
        # column anyway), AutoFeat actually respects the configured value - a caller-specified
        # target column is deliberately supported here (see README's regression example).
        target_column_id = config.target_column_id
        target_column = base_table_df.columns[target_column_id]

        downstream_task = getattr(config, "downstream_task", "classification")
        if downstream_task not in ("classification", "regression"):
            raise ValueError(
                f"downstream_task {downstream_task!r} not supported by AutoFeat: use 'classification' or 'regression'"
            )
        dataset_type = REGRESSION if downstream_task == "regression" else CLASSIFICATION

        if downstream_task == "regression" and not pd.api.types.is_numeric_dtype(base_table_df[target_column]):
            raise ValueError(f"Target column ({target_column!r}) not numeric")

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

        if self.verbose:
            print(f"[AutoFeatBaseline] join graph: {len(join_paths_df)} edges, "
                  f"{len(set(join_paths_df['from_id']) | set(join_paths_df['to_id']))} tables reachable")
            bfs_start = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bfs_traversal.streaming_feature_selection(
                join_paths_df=join_paths_df,
                lake_data_folder=str(lake_data_folder),
                lake_table_sep=",",
                queue={base_table_id},
            )

            sorted_paths = sorted(
                bfs_traversal.ranking.items(),
                key=lambda r: (-float(r[1]), get_path_length(r[0]), r[0]),
            )
            join_name, rank = sorted_paths[0]

            if self.verbose:
                print(f"[AutoFeatBaseline] BFS discovered {len(sorted_paths)} candidate join paths "
                      f"in {time.time() - bfs_start:.1f}s")
                print(f"[AutoFeatBaseline] chosen path: rank={rank:.4f}, "
                      f"tables_joined={get_path_length(join_name) + 1} (#1 of {len(sorted_paths)} by rank)")
                for name, score in sorted_paths[1:6]:
                    print(f"[AutoFeatBaseline]   runner-up: rank={score:.4f}, "
                          f"tables_joined={get_path_length(name) + 1}")

            if join_name == bfs_traversal.base_table_id:
                if self.verbose:
                    print("[AutoFeatBaseline] best-ranked candidate is the base table itself; no join performed")
                return pl.from_pandas(base_table_df)

            join_start = time.time()
            path_list, features = resolve_path_list(bfs_traversal, join_name)
            dataframe = join_from_path(
                path_list,
                join_paths_df,
                str(lake_data_folder),
                ",",
                base_table_sep,
                bfs_traversal.base_table_id,
                bfs_traversal.target_column,
            )

        if dataframe is None:
            raise ValueError(f"Join failed for path: {join_name}")

        dataframe_columns = set(dataframe.columns)
        features = list(dict.fromkeys(f for f in features if f in dataframe_columns))
        if len(features) < 2:
            features = list(bfs_traversal.partial_join_selected_features[bfs_traversal.base_table_id])
            features.append(bfs_traversal.target_column)

        if self.verbose:
            print(f"[AutoFeatBaseline] materialised final join in {time.time() - join_start:.1f}s: "
                  f"{dataframe.shape[0]} rows, {len(features)} selected columns "
                  f"(base table had {base_table_df.shape[0]} rows, {base_table_df.shape[1]} columns)")

        return pl.from_pandas(dataframe[features])
