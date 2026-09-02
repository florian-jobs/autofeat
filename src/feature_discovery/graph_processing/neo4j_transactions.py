import pandas as pd
import polars as pl

# Module-level cache for raw (unprefixed) dataframes, keyed by (lake_data_folder, node_id)
_df_cache: dict[tuple, pd.DataFrame] = {}


def get_pk_fk_nodes(join_paths_df: pd.DataFrame, source_path: str):
    """Return pairs of nodes (n, m) connected by a relationship with weight == 1."""
    matches = join_paths_df[
        ((join_paths_df["from_id"] == source_path) | (join_paths_df["to_id"] == source_path))
        & (join_paths_df.get("weight", 0) == 1)
    ]

    results = []
    for _, row in matches.iterrows():
        n = {"id": row["from_id"]}
        m = {"id": row["to_id"]}
        results.append([n, m])
    return results


def get_adjacent_nodes(join_paths_df: pd.DataFrame, base_node_id: str):
    """Simulate Neo4j: return all node IDs adjacent to base_node_id."""
    adjacent = set()
    for _, row in join_paths_df.iterrows():
        if row["from_id"] == base_node_id:
            adjacent.add(row["to_id"])
        elif row["to_id"] == base_node_id:
            adjacent.add(row["from_id"])
    return list(adjacent)


def get_relation_properties_node_name(join_paths_df: pd.DataFrame, from_id: str, to_id: str):
    """Simulate Neo4j: return relationship properties between two nodes, oriented so from_label/
    from_column always correspond to the caller's `from_id` (the already-joined side), regardless
    of which direction the row happens to be stored in -- a join_paths_df that stores both
    directions of a match (independently-set from/to columns) would otherwise point the caller at
    a column that only exists in the not-yet-joined table, crashing the join a few steps
    downstream. A join_paths_df that only ever stores one direction per edge (parent -> child) is
    a no-op here. Sorted by weight descending, stably: the caller (autofeat.py) picks its
    highest_ranked_join_keys by taking a prefix of this list and stopping at the first weight that
    differs from the first entry's, assuming descending order -- the real Neo4j backend's Cypher
    enforces this explicitly (`ORDER BY r.weight DESC`) but this in-memory version didn't. kind=
    "stable" so tied weights (e.g. a uniform weight=1 graph) keep their original relative order
    instead of being reshuffled by pandas' default, non-stable quicksort."""
    matches = join_paths_df[
        ((join_paths_df["from_id"] == from_id) & (join_paths_df["to_id"] == to_id))
        | ((join_paths_df["from_id"] == to_id) & (join_paths_df["to_id"] == from_id))
    ]
    if "weight" in matches.columns:
        matches = matches.sort_values("weight", ascending=False, kind="stable")

    results = []
    for _, row in matches.iterrows():
        if row["from_id"] == from_id:
            props = {
                "from_label": row["from_id"],
                "to_label": row["to_id"],
                "from_column": row["from_column"],
                "to_column": row["to_column"],
            }
            pair = (row["from_id"], row["to_id"])
        else:
            props = {
                "from_label": row["to_id"],
                "to_label": row["from_id"],
                "from_column": row["to_column"],
                "to_column": row["from_column"],
            }
            pair = (row["to_id"], row["from_id"])
        if "weight" in row:
            props["weight"] = row["weight"]

        results.append([props, pair[0], pair[1]])
    return results

def get_node_by_id(join_paths_df: pd.DataFrame, node_id: str):
    """Simulate Neo4j _get_node_by_id() using own join paths CSV."""
    matches = join_paths_df[
        (join_paths_df["from_id"] == node_id) | (join_paths_df["to_id"] == node_id)
    ]

    if matches.empty:
        return None

    columns = set()
    for _, row in matches.iterrows():
        if row["from_id"] == node_id and "from_column" in row:
            columns.add(row["from_column"])
        if row["to_id"] == node_id and "to_column" in row:
            columns.add(row["to_column"])

    return {
        "id": node_id,
        "columns": sorted(list(columns)),
        "relationships": len(matches),  # optional metadata
    }

def clear_df_cache():
    """Clear the module-level dataframe cache."""
    _df_cache.clear()


def get_df_with_prefix(
    join_paths_df: pd.DataFrame,
    lake_data_folder: str,
    node_id: str,
    table_sep: str,
    target_column=None,
    key_column=None,
    use_polars: bool = False,
) -> tuple:
    """
    Get the node from the database, read the file identified by node_id and prefix the column names with the node label.
    Raw (unprefixed) dataframes are cached by default to avoid re-reading large CSVs.

    :param node_id: ID of the node - used to retrieve the corresponding node from the database
    :param target_column: Optional parameter. The name of the label/target column containing the classes,
            only needed when the dataset to read contains the class.
    :param key_column: Optional parameter. The name of the base table's own join/key column - like
            target_column, kept unprefixed so callers downstream (e.g. beluga's test(), which looks
            up the key/target by their original, unprefixed names) can still find it. Only meaningful
            for the base node; right-hand join candidates need their columns fully prefixed.
    :return: 0: A pandas dataframe whose columns are prefixed with the node label, 1: the node label
    """
    node = get_node_by_id(join_paths_df, node_id)
    if node is None:
        print(f"Node with id {node_id} not found in join paths dataframe.")
        raise ValueError(f"Node with id {node_id} not found in join paths dataframe.")
    
    node_label = node.get("id")

    # Check cache for raw dataframe
    cache_key = (lake_data_folder, node_id)
    if cache_key in _df_cache:
        raw_df = _df_cache[cache_key].copy()
    else:
        if use_polars:
            raw_df = pl.read_csv(f'{lake_data_folder}/{node_id}', encoding="utf8-lossy", quote_char='"', separator=table_sep, ignore_errors=True, truncate_ragged_lines=True).to_pandas()
        else:
            raw_df = pd.read_csv(
                f'{lake_data_folder}/{node_id}', header=0, engine="python", encoding="utf8", sep=table_sep,
                on_bad_lines='skip'
            )
        # Some real corpus files (e.g. Statistics Canada exports) start with a UTF-8 BOM and/or
        # quote their header cells; left in place, the column name ends up as e.g. 'ï»¿"REF_DATE"'
        # instead of 'REF_DATE', which then mismatches the clean name join_paths.csv expects and
        # raises a KeyError in step_join(). Strip both before anything else uses these columns.
        bom = chr(0xFEFF)
        raw_df.columns = [str(col).lstrip(bom).replace('ï»¿', '').strip('"').strip() for col in raw_df.columns]
        _df_cache[cache_key] = raw_df.copy()

    # Apply prefixing - target_column and key_column (if given) are excluded, so they keep their
    # original, unprefixed names for downstream code that looks them up by that name.
    exclude = [c for c in (target_column, key_column) if c]
    if exclude:
        dataframe = raw_df.set_index(exclude).add_prefix(f"{node_label}.").reset_index()
    else:
        dataframe = raw_df.add_prefix(f"{node_label}.")

    return dataframe, node_label