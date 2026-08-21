import numpy as np

np.random.seed(42)

import typer
from typing_extensions import Annotated

from feature_discovery.dataset_relation_graph.dataset_discovery import profile_valentine_dataset, profile_valentine_all
from feature_discovery.dataset_relation_graph.ingest_data import ingest_nodes
from feature_discovery.experiments.init_datasets import ALL_DATASETS

app = typer.Typer()

@app.command()
def ingest_data(
    data_discovery_threshold: Annotated[
        float,
        typer.Option(
            help="Run dataset discovery to find more connections within the entire data lake with given"
            " accuracy rate threshold"
        ),
    ] = None,
    discover_connections_data_lake: Annotated[
        bool, typer.Option(help="Run dataset discovery to find more connections within the entire data lake")
    ] = False,
):
    """
    Ingest all dataset from specified "data" folder.
    """
    ingest_nodes()

    if data_discovery_threshold and discover_connections_data_lake:
        profile_valentine_all(valentine_threshold=data_discovery_threshold)
        return

    if data_discovery_threshold and not discover_connections_data_lake:
        for dataset in ALL_DATASETS:
            profile_valentine_dataset(dataset.base_table_label, valentine_threshold=data_discovery_threshold)


if __name__ == "__main__":
    app()
