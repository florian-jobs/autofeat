"""
Beluga entry point for the AutoFeat baseline: wires the Neo4j-decoupled AutoFeat pipeline
(streaming_feature_selection over join_paths.csv, then evaluate_join_paths) behind beluga's Config
object, so beluga can run AutoFeat without knowing anything about its internals. Shaped to match the
sibling ARDABaseline/QCRBaseline/COCOABaseline contract (same config extraction, same join-first/
target-last convention, same corpus_dir/table_dir fallback), with the deviations noted inline where
AutoFeat's needs differ from theirs.

Unlike the siblings, beluga.config.schema/beluga.online.base_table are imported lazily (inside run(),
not at module level) and only used if available: this package's own pyproject.toml caps it at
`requires-python = ">=3.8,<3.10"` (autogluon==0.7.0 has no newer wheels), while beluga's own Config
uses `X | None` annotations that require Python >=3.10 to even import - the two literally cannot be
imported in the same interpreter. So this file (and the feature_discovery pipeline it wraps) is meant
to run in AutoFeat's own venv, in a separate process from the rest of beluga - see scripts/
augment_and_test.py's autofeat branch, which launches it that way rather than importing it directly.
config just needs to duck-type target_column_id/downstream_task/data_dir/corpus/queries_dir/base_table
(a real beluga Config satisfies this too, for the rare case both happen to be importable together).
"""

import time
import warnings
from importlib import resources
from pathlib import Path

import pandas as pd
import polars as pl

from feature_discovery.autofeat_pipeline.autofeat import AutoFeat
from feature_discovery.autofeat_pipeline.join_path_utils import get_path_length
from feature_discovery.experiments.dataset_object import CLASSIFICATION, REGRESSION
from feature_discovery.experiments.evaluate_join_paths import join_from_path, resolve_path_list
from feature_discovery.graph_processing.neo4j_transactions import clear_df_cache

_BASE_TABLE_SEP = ","  # matches ARDABaseline's hardcoded sep_lake=","; AutoFeat's own pipeline requires one


class AUTOFEATBaseline:

    def __init__(
        self,
        value_ratio: float = 0.65,
        top_k: int = 15,
        sample_size: int = 3_000,  # default value used in the AutoFeat implementation
        verbose: bool = False,  # opt-in diagnostics (candidate counts, chosen path, timing) to stdout
    ) -> None:

        self.value_ratio = value_ratio
        self.top_k = top_k
        self.sample_size = sample_size
        self.verbose = verbose

    @staticmethod
    def _toy_default(relative_path: str) -> Path:
        """beluga's bundled toy corpus/query table, for when config.data_dir/queries_dir isn't set. Only
        reachable if beluga happens to be importable here (see module docstring) - real usage always
        sets data_dir/queries_dir explicitly, so this is a rarely-hit convenience fallback, not the
        happy path."""
        return resources.files("beluga.data").joinpath(relative_path)

    @staticmethod
    def _read_base_table(config, table_dir: Path):
        """Returns (base_table_df: pandas.DataFrame, target_column_id: int).

        Prefers beluga's own read_base_table (consolidates headers, casts dtypes, drops duplicates -
        same as ARDABaseline/QCRBaseline/COCOABaseline) when beluga is importable; the fallback plain
        CSV read otherwise means AutoFeat still runs correctly in its own venv (see module docstring),
        just without that consolidation/casting.
        """
        try:
            from beluga.online.base_table import read_base_table
        except ImportError:
            base_table_path = table_dir / "table.csv" if (table_dir / "table.csv").exists() else next(
                p for p in Path(table_dir).glob("*.csv") if p.name != "join_paths.csv"
            )
            base_table_df = pd.read_csv(base_table_path, encoding="utf8")
            return base_table_df, config.target_column_id

        base_table_pl = read_base_table(config.base_table, table_dir, config)
        base_table_df = base_table_pl.to_pandas()
        # read_base_table already reordered columns so target is last (join-first/target-last
        # convention) - reflect that here rather than trusting config.target_column_id's raw value.
        return base_table_df, len(base_table_df.columns) - 1

    def run(
        self,
        config=None  # duck-typed: real beluga Config, or anything exposing the same attributes - see module docstring
    ) -> pl.DataFrame:

        if config is None:
            raise ValueError("A config is required (target_column_id, downstream_task, data_dir, corpus, queries_dir, base_table)")

        if not config.target_column_id:
            raise ValueError("Value for target_column_id not specified in the configuration file")

        if config.downstream_task == "construction":
            raise ValueError("Construction task not supported by AutoFeat: use 'classification' or 'regression'")

        if config.data_dir is not None:
            lake_data_folder = Path(config.data_dir) / config.corpus
        else:
            lake_data_folder = self._toy_default("corpora/toy")

        if config.queries_dir is not None:
            table_dir = Path(config.queries_dir) / config.base_table
        else:
            table_dir = self._toy_default("queries/beers")  # to update with a new default base table

        base_table_df, target_column_id = self._read_base_table(config, table_dir)
        target_column = base_table_df.columns[target_column_id]
        # Move the target column last, matching ARDABaseline/QCRBaseline/COCOABaseline's convention -
        # base_table_pd below (what actually gets written to the lake and joined against) needs this,
        # even though AutoFeat itself looks target_column up by name, not position.
        base_table_df = base_table_df[[c for c in base_table_df.columns if c != target_column] + [target_column]]

        if config.downstream_task == "regression" and not pd.api.types.is_numeric_dtype(base_table_df[target_column]):
            raise ValueError(f"Target column ({target_column!r}) not numeric")

        dataset_type = REGRESSION if config.downstream_task == "regression" else CLASSIFICATION

        # The rest of this pipeline (resolve_path_list in particular) assumes every node id ends in
        # ".csv" when reconstructing table names from prefixed feature names, so the materialised
        # base table node id must follow that convention too, even though config.base_table itself
        # may not end in .csv.
        base_table_id = config.base_table + ".csv"
        base_table_pd = base_table_df
        clear_df_cache()
        lake_data_folder = Path(lake_data_folder)
        lake_data_folder.mkdir(parents=True, exist_ok=True)
        base_table_pd.to_csv(lake_data_folder / base_table_id, index=False, sep=_BASE_TABLE_SEP)

        # No leaky_features.json handling here (unlike ARDA/QCR/COCOA): AutoFeat's underlying
        # streaming_feature_selection has no column-level exclusion hook to filter candidates
        # through, so wiring this up would need changes to the vendored feature_discovery pipeline
        # itself, not just this file.
        join_paths_df_path = str(table_dir / "join_paths.csv")
        join_paths_df = pd.read_csv(join_paths_df_path)
        if "from_table" in join_paths_df.columns and "from_id" not in join_paths_df.columns:
            # beluga's own get_join_paths.py (scripts/get_join_paths.py) writes from_table/to_table,
            # not from_id/to_id -- AutoFeat/neo4j_transactions.py expect the latter everywhere
            # (ablation.py's ground-truth connections.csv loader and setup_baseline_test_fixture.py's
            # fixture both rename the same way). Without this, join_paths_df['from_id'] a few lines
            # down raises KeyError on any real beluga-generated join_paths.csv -- the local test
            # fixture never caught this because it already writes from_id/to_id directly.
            join_paths_df = join_paths_df.rename(columns={"from_table": "from_id", "to_table": "to_id"})

        bfs_traversal = AutoFeat(
            join_paths_df=join_paths_df,
            lake_data_folder=str(lake_data_folder),
            base_table_sep=_BASE_TABLE_SEP,
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
            print(f"[AUTOFEATBaseline] join graph: {len(join_paths_df)} edges, "
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
                print(f"[AUTOFEATBaseline] BFS discovered {len(sorted_paths)} candidate join paths "
                      f"in {time.time() - bfs_start:.1f}s")
                print(f"[AUTOFEATBaseline] chosen path: rank={rank:.4f}, "
                      f"tables_joined={get_path_length(join_name) + 1} (#1 of {len(sorted_paths)} by rank)")
                for name, score in sorted_paths[1:6]:
                    print(f"[AUTOFEATBaseline]   runner-up: rank={score:.4f}, "
                          f"tables_joined={get_path_length(name) + 1}")

            if join_name == bfs_traversal.base_table_id:
                if self.verbose:
                    print("[AUTOFEATBaseline] best-ranked candidate is the base table itself; no join performed")
                return pl.from_pandas(base_table_df)

            join_start = time.time()
            path_list, features = resolve_path_list(bfs_traversal, join_name)
            dataframe = join_from_path(
                path_list,
                join_paths_df,
                str(lake_data_folder),
                ",",
                _BASE_TABLE_SEP,
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
            print(f"[AUTOFEATBaseline] materialised final join in {time.time() - join_start:.1f}s: "
                  f"{dataframe.shape[0]} rows, {len(features)} selected columns "
                  f"(base table had {base_table_pd.shape[0]} rows, {base_table_pd.shape[1]} columns)")

        return pl.from_pandas(dataframe[features])


"""
from beluga.config.loader import load_config

config = load_config("config.yaml")

autofeat = AUTOFEATBaseline()

print(autofeat.run(config))
"""
