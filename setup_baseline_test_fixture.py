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

    # from_table/to_table (not from_id/to_id) to match what beluga's own scripts/get_join_paths.py
    # actually writes -- baseline.py normalizes this on load, so testing against the real column
    # names here is what actually exercises that path instead of coincidentally bypassing it.
    connections = pd.read_csv(SOURCE / "connections.csv")
    join_paths = connections.rename(columns={
        "fk_table": "from_table",
        "fk_column": "from_column",
        "pk_table": "to_table",
        "pk_column": "to_column",
    })
    join_paths["weight"] = 1
    join_paths["from_table"] = join_paths["from_table"].replace(BASE_TABLE_FILE, BASE_TABLE_NODE_ID)
    join_paths["to_table"] = join_paths["to_table"].replace(BASE_TABLE_FILE, BASE_TABLE_NODE_ID)

    join_paths.to_csv(QUERIES_DIR / "join_paths.csv", index=False)

    print("Base table copied to:", QUERIES_DIR / BASE_TABLE_FILE)
    print("Joi5059n paths written to:", QUERIES_DIR / "join_paths.csv")
    print("Lake corpus copied to:", CORPUS_DIR)
    print(join_paths)


if __name__ == "__main__":
    main()
