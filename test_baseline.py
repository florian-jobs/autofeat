from types import SimpleNamespace

from baseline import AutoFeatBaseline

config = SimpleNamespace(
    queries_dir="tmp/queries",
    base_table="credit",
    data_dir="tmp/corpus",
    corpus="credit_lake",
    target_column_id=0,
    join_column_id=4,
    downstream_task="classification",
)

baseline = AutoFeatBaseline(value_ratio=0.65, top_k=15, algorithm="LR")
result = baseline.run(config)
print(result)
