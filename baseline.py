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

import json
import shutil
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

# Name of the corpus-wide, multi-hop join-path file (see scripts/get_corpus_join_paths.py): table-to-
# table edges across the whole corpus, sitting next to the per-base-table join_paths.csv files (which
# only ever record edges from a base/query table to the corpus, not between corpus tables).
_CORPUS_JOIN_PATHS_FILENAME = "join_paths.csv"


def _materialize_lake_table(source: Path, dest: Path) -> None:
    """
    Makes `source` (a nested corpus table, <table_id>/table.csv) available at `dest` (a flat
    <table_id>.csv - the layout AutoFeat's own get_df_with_prefix() expects inside lake_data_folder,
    confirmed against its own discover_join_paths.py tool, which stages candidates the same way).
    Symlinks when possible (cheap, and shared across every base table run against this corpus);
    falls back to copying when symlinks aren't available (e.g. no privilege on Windows).
    """
    if dest.exists() or dest.is_symlink():
        return
    try:
        dest.symlink_to(source.resolve())
    except OSError:
        try:
            shutil.copyfile(source, dest)
        except FileExistsError:
            pass  # another concurrent run materialized it first


def _resolve_column_ref(ref, csv_path: Path, header_cache: dict) -> str:
    """
    BlendIndex.get_top_joins() (queried by scripts/get_join_paths.py and
    get_corpus_join_paths.py) returns column_id as a positional index, not a name - most corpus
    tables don't have a metadata.json-declared header for beluga to resolve one from. AutoFeat's own
    join logic (step_join et al.) needs the literal column name as it appears in the physical csv
    though, so a purely-numeric ref gets resolved against that csv's own header row here; anything
    else (e.g. the query-table side, which get_join_paths.py already writes as a real name) is
    passed through unchanged. header_cache avoids re-reading the same csv's header for every edge
    that references it.
    """
    ref_str = str(ref)
    if not ref_str.lstrip("-").isdigit():
        return ref_str
    if csv_path not in header_cache:
        try:
            header_cache[csv_path] = pd.read_csv(csv_path, nrows=0).columns.tolist()
        except Exception:
            header_cache[csv_path] = []
    header = header_cache[csv_path]
    idx = int(ref_str)
    return header[idx] if idx < len(header) else ref_str


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
            corpus_dir = Path(config.data_dir) / config.corpus
        else:
            corpus_dir = Path(self._toy_default("corpora/toy"))

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

        # lake_data_folder is a flat staging dir (<table_id>.csv per table), which is what AutoFeat's
        # own get_df_with_prefix() expects - confirmed against discover_join_paths.py, AutoFeat's own
        # tool, which stages candidates the same way. The real corpus (corpus_dir) instead nests each
        # table as <table_id>/table.csv, so referenced corpus tables get symlinked in below rather
        # than read from corpus_dir directly. Shared across every base table run against this corpus
        # (not per-base-table), so candidate tables only need materializing once.
        lake_data_folder = corpus_dir.parent / f"{config.corpus}_autofeat_lake"
        lake_data_folder.mkdir(parents=True, exist_ok=True)
        base_table_pd.to_csv(lake_data_folder / base_table_id, index=False, sep=_BASE_TABLE_SEP)

        leaky_features_path = table_dir / "leaky_features.json"
        if leaky_features_path.exists():
            with open(leaky_features_path, "r", encoding="utf-8", errors="replace") as file:
                leaky_features = json.load(file)
        else:
            leaky_features = dict()

        # join_paths.csv (per base table): direct edges from this base table to the corpus, as
        # discovered by scripts/get_join_paths.py. join_paths.csv (corpus-wide, if present next to
        # the corpus itself): table-to-table edges across the whole corpus, from
        # scripts/get_corpus_join_paths.py - without these, AutoFeat can only ever join one hop out
        # from the base table, even though its own BFS supports traversing further.
        join_paths_dfs = [pd.read_csv(table_dir / "join_paths.csv")]
        corpus_join_paths_path = corpus_dir / _CORPUS_JOIN_PATHS_FILENAME
        if corpus_join_paths_path.exists():
            join_paths_dfs.append(pd.read_csv(corpus_join_paths_path))
        join_paths_df = pd.concat(join_paths_dfs, ignore_index=True)

        if "from_table" in join_paths_df.columns and "from_id" not in join_paths_df.columns:
            # beluga's own get_join_paths.py/get_corpus_join_paths.py write from_table/to_table, not
            # from_id/to_id -- AutoFeat/neo4j_transactions.py expect the latter everywhere (ablation.py's
            # ground-truth connections.csv loader and setup_baseline_test_fixture.py's fixture both
            # rename the same way). Without this, join_paths_df['from_id'] a few lines down raises
            # KeyError on any real beluga-generated join_paths.csv -- the local test fixture never
            # caught this because it already writes from_id/to_id directly.
            join_paths_df = join_paths_df.rename(columns={"from_table": "from_id", "to_table": "to_id"})

        # beluga's own from_id/to_id values are bare table ids (no ".csv") - arda/qcr consume them
        # that way directly, so get_join_paths.py/get_corpus_join_paths.py can't change that shared
        # format. AutoFeat needs the ".csv" suffix (see base_table_id above), so it's added here,
        # only to this in-memory copy, not to the files on disk.
        for col in ("from_id", "to_id"):
            join_paths_df[col] = join_paths_df[col].apply(lambda x: x if str(x).endswith(".csv") else f"{x}.csv")

        # Materialize every referenced corpus table (other than the base table, already written
        # above) as a flat <table_id>.csv, symlinked from the real nested corpus layout.
        referenced_ids = set(join_paths_df["from_id"]) | set(join_paths_df["to_id"])
        for node_id in referenced_ids - {base_table_id}:
            table_id = node_id[:-4] if node_id.endswith(".csv") else node_id
            source = corpus_dir / table_id / "table.csv"
            if source.exists():
                _materialize_lake_table(source, lake_data_folder / node_id)

        # Resolve positional column refs (see _resolve_column_ref) against each row's own table -
        # must happen after materializing above, so the flat lake csv's are there to read headers from.
        header_cache = {}
        for id_col, name_col in (("from_id", "from_column"), ("to_id", "to_column")):
            join_paths_df[name_col] = [
                _resolve_column_ref(ref, lake_data_folder / node_id, header_cache)
                for node_id, ref in zip(join_paths_df[id_col], join_paths_df[name_col])
            ]

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
            leaky_features=leaky_features,
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
