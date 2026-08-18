import glob
import itertools
import os
from typing import List

# Parallel(n_jobs=-1) below spawns one worker *process* per CPU core. Each worker's numpy/pandas
# calls then independently try to open a BLAS thread pool also sized to the core count, so total
# demanded threads scales ~quadratically with core count. On many-core servers (e.g. hercules)
# that exceeds OpenBLAS's hardcoded 128-thread-context build limit and it aborts. Pin each
# worker's BLAS libs to a single thread so all the parallelism comes from joblib's process fan-out
# instead of nested thread pools -- must be set before numpy is imported anywhere in the process.
for _blas_env_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                      "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_blas_env_var, "1")

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from valentine import valentine_match
from valentine.algorithms import Coma

from feature_discovery.config import DATA_FOLDER, CONNECTIONS
from feature_discovery.graph_processing.neo4j_transactions import merge_nodes_relation_tables


def profile_valentine_all(valentine_threshold: float = 0.55):
    files = glob.glob(f"{DATA_FOLDER}/**/*.csv", recursive=True)
    files = [f for f in files if CONNECTIONS not in f]

    profile_valentine_logic(files, valentine_threshold)


def profile_valentine_dataset(dataset_name: str, valentine_threshold: float = 0.55):
    files = glob.glob(f"{DATA_FOLDER / dataset_name}/**/*.csv", recursive=True)
    files = [f for f in files if CONNECTIONS not in f]

    profile_valentine_logic(files, valentine_threshold)


def profile_valentine_logic(files: List[str], valentine_threshold: float = 0.55):
    def profile(table_pair):
        (tab1, tab2) = table_pair

        a_table_path = tab1.partition(f"{DATA_FOLDER}/")[2]
        b_table_path = tab2.partition(f"{DATA_FOLDER}/")[2]

        a_table_name = a_table_path.split("/")[-1]
        b_table_name = b_table_path.split("/")[-1]

        print(f"Processing the match between:\n\t{a_table_path}\n\t{b_table_path}")
        df1 = pd.read_csv(tab1, encoding="utf8")
        df2 = pd.read_csv(tab2, encoding="utf8")
        matches = valentine_match(df1, df2, Coma(strategy="COMA_OPT"))

        for item in matches.items():
            ((_, col_from), (_, col_to)), similarity = item
            if similarity > valentine_threshold:
                print(f"Similarity {similarity} between:\n\t{a_table_path} -- {col_from}\n\t{b_table_path} -- {col_to}")

                merge_nodes_relation_tables(a_table_name=a_table_name,
                                            b_table_name=b_table_name,
                                            a_table_path=a_table_path,
                                            b_table_path=b_table_path,
                                            a_col=col_from,
                                            b_col=col_to,
                                            weight=similarity)

                merge_nodes_relation_tables(a_table_name=b_table_name,
                                            b_table_name=a_table_name,
                                            a_table_path=b_table_path,
                                            b_table_path=a_table_path,
                                            a_col=col_to,
                                            b_col=col_from,
                                            weight=similarity)

    Parallel(n_jobs=-1)(delayed(profile)(table_pair) for table_pair in tqdm(itertools.combinations(files, r=2)))
