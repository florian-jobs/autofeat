"""
Driver-backed Neo4j client for the offline ingestion pipeline (dataset_relation_graph/ingest_data.py,
dataset_relation_graph/dataset_discovery.py): builds the join-discovery graph via genuine schema
matching (Valentine/Coma) against a live Neo4j instance -- the same mechanism used to build the
reference graph behind the AutoFeat paper's reported numbers (see autofeat_og_ingestion).

Kept separate from neo4j_transactions.py, which simulates the same read API purely in-memory over a
join_paths_df (no live Neo4j) -- that's what AutoFeat's own BFS traversal (autofeat_pipeline/autofeat.py)
and the beluga integration (baseline.py) depend on, and it must keep working without a Neo4j server.
"""
from typing import List

from neo4j import GraphDatabase

from feature_discovery.config import NEO4J_HOST, NEO4J_CREDENTIALS, NEO4J_DATABASE
from feature_discovery.graph_processing.neo4j_queries import (
    _merge_nodes_relation_tables,
    _get_relation_properties,
    _get_node_by_id,
    _get_pk_fk_nodes,
    _get_adjacent_nodes,
    _get_relation_properties_node_name,
    _export_all_connections,
    _export_dataset_connections,
    _create_node,
)

driver = GraphDatabase.driver(NEO4J_HOST, auth=NEO4J_CREDENTIALS)


def merge_nodes_relation_tables(a_table_name, b_table_name, a_table_path, b_table_path, a_col, b_col, weight=1):
    with driver.session(database=NEO4J_DATABASE) as session:
        return session.write_transaction(
            _merge_nodes_relation_tables,
            a_table_name, b_table_name, a_table_path, b_table_path, a_col, b_col, weight,
        )


def get_relation_properties(from_id, to_id):
    with driver.session(database=NEO4J_DATABASE) as session:
        return session.write_transaction(_get_relation_properties, from_id, to_id)


def get_relation_properties_node_name(from_id, to_id):
    with driver.session(database=NEO4J_DATABASE) as session:
        return session.write_transaction(_get_relation_properties_node_name, from_id, to_id)


def get_node_by_id(node_id):
    with driver.session(database=NEO4J_DATABASE) as session:
        return session.write_transaction(_get_node_by_id, node_id)


def get_pk_fk_nodes(source_path):
    with driver.session(database=NEO4J_DATABASE) as session:
        return session.write_transaction(_get_pk_fk_nodes, source_path)


def get_adjacent_nodes(node_id) -> list:
    """Return a list of node IDs adjacent to node_id."""
    with driver.session(database=NEO4J_DATABASE) as session:
        return session.write_transaction(_get_adjacent_nodes, node_id)


def export_all_connections() -> List[dict]:
    with driver.session(database=NEO4J_DATABASE) as session:
        return session.write_transaction(_export_all_connections)


def export_dataset_connections(dataset_label: str) -> List[dict]:
    with driver.session(database=NEO4J_DATABASE) as session:
        return session.write_transaction(_export_dataset_connections, dataset_label)


def create_node(node_id, node_label):
    with driver.session(database=NEO4J_DATABASE) as session:
        return session.write_transaction(_create_node, node_id, node_label)


def wipe_database() -> None:
    """Delete every node in the target database. Only safe on a disposable/dedicated instance --
    prefer wipe_dataset() when the database may hold other datasets' graphs too."""
    with driver.session(database=NEO4J_DATABASE) as session:
        session.run("MATCH (n) DETACH DELETE n").consume()


def wipe_dataset(dataset_label: str) -> None:
    """Delete only nodes whose id is under dataset_label/ (and their relationships), leaving any
    other datasets' graphs in the same database untouched."""
    with driver.session(database=NEO4J_DATABASE) as session:
        session.run(
            "MATCH (n) WHERE n.id STARTS WITH $prefix DETACH DELETE n",
            prefix=f"{dataset_label}/",
        ).consume()
