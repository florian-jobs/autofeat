# Join-path names are built by chaining hops together. Table/column ids from real corpora routinely
# contain dashes (e.g. Socrata-style ids like "nyc-finance-39g5-gbp3"), so plain "-"/"--" can't be used
# as delimiters without ambiguity. These are non-printable control characters that can't collide with
# any real table or column name.
HOP_SEP = "\x1e"    # separates one hop from the next
FIELD_SEP = "\x1f"  # separates the 4 fields (from_table, from_column, to_column, to_table) within a hop


def get_path_length(path: str) -> int:
    # Path looks like table_source/table_name/key<HOP_SEP>table_source...
    path_tokens = path.split(HOP_SEP)
    # Length = 1 means that we have 2 tables
    return len(path_tokens) - 1


def compute_join_name(join_key_property: tuple, partial_join_name: str) -> str:
    """
    Compute the name of the partial join, given the properties of the new join and the previous join name.

    :param join_key_property: (neo4j relation property, outbound label, inbound label)
    :param partial_join_name: Name of the partial join.
    :return: The name of the next partial join
    """
    join_prop, from_table, to_table = join_key_property
    hop = FIELD_SEP.join([from_table, join_prop['from_column'], join_prop['to_column'], to_table])
    return f"{partial_join_name}{HOP_SEP}{hop}"
