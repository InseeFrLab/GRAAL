def get_code_information(graph, code: str):
    query = """
    MATCH (node {CODE: $code})
    OPTIONAL MATCH (node)<-[:HAS_CHILD]-(parent)
    OPTIONAL MATCH (node)-[:HAS_CHILD]->(child)
    WITH node, parent, collect({code: child.CODE, name: child.NAME}) as children
    RETURN node.CODE as code,
    node.LEVEL as level,
    node.NAME as name,
    node.text as description,
    node.Includes as includes,
    node.IncludesAlso as includes_also,
    node.Excludes as excludes,
    node.Implementation_rule as implementation_rule,
    parent.CODE as parent_code,
    children,
    size(children) as children_count
    """
    result = graph.graph.query(query, params={"code": code})

    if not result:
        print("No result in get_code_information")
        return ()

    return result[0]


def NAF_to_NACE(code):
    return code[0:2] + '.' + code[2:4]
