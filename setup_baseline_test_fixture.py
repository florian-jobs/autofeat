import shutil
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE = PROJECT_ROOT / "data" / "benchmark" / "credit"

QUERIES_DIR = PROJECT_ROOT / "tmp" / "queries" / "credit"
CORPUS_DIR = PROJECT_ROOT / "tmp" / "corpus" / "credit_lake"

BASE_TABLE_FILE = "table_0_0.csv"
BASE_TABLE_NODE_ID = "credit.csv"


def main():
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy(SOURCE / BASE_TABLE_FILE, QUERIES_DIR / BASE_TABLE_FILE)

    for csv_file in sorted(SOURCE.glob("*.csv")):
        if csv_file.name in (BASE_TABLE_FILE, "connections.csv", "datasets.csv"):
            continue
        shutil.copy(csv_file, CORPUS_DIR / csv_file.name)

    connections = pd.read_csv(SOURCE / "connections.csv")
    join_paths = connections.rename(columns={
        "fk_table": "from_id",
        "fk_column": "from_column",
        "pk_table": "to_id",
        "pk_column": "to_column",
    })
    join_paths["weight"] = 1
    join_paths["from_id"] = join_paths["from_id"].replace(BASE_TABLE_FILE, BASE_TABLE_NODE_ID)
    join_paths["to_id"] = join_paths["to_id"].replace(BASE_TABLE_FILE, BASE_TABLE_NODE_ID)

    join_paths.to_csv(QUERIES_DIR / "join_paths.csv", index=False)

    print("Base table copied to:", QUERIES_DIR / BASE_TABLE_FILE)
    print("Join paths written to:", QUERIES_DIR / "join_paths.csv")
    print("Lake corpus copied to:", CORPUS_DIR)
    print(join_paths)


if __name__ == "__main__":
    main()
