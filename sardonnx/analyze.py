"""
Explore existing ONNX models.
"""

from onnx import GraphProto, NodeProto


class MalformedGraphError(ValueError):
    """Raised when an ONNX graph is structurally invalid or inconsistent."""


def last_op_nodes(
    graph: GraphProto, op_type: str | set[str], num: int | None = 1
) -> tuple[list[NodeProto], list[int]]:
    """
    Conducts a breadth first search from the graph outpts to identify the last
    `n` nodes of a particular op type.

    NOTE: More than `num` nodes may be returned as the final frontier will be
    fully explored to allow nodes at the same depth to both be identified.

    :param graph: The ONNX graph to search in.
    :param op_type: The op type to search for. A full list can be found
        [here](https://onnx.ai/onnx/operators/).
    :param num: The target number of nodes to return. If `None`, all nodes of the
        op type will be returned.

    :return: A list of the last nodes of the given op type in the graph in the
        order of discovery and a list of the corresponding depths.
    """
    if isinstance(op_type, str):
        op_type = {op_type}

    backmap: dict[str, NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            if out in backmap:
                raise MalformedGraphError(f"Duplicate output: {out}")
            backmap[out] = node

    explored: set[int] = set()
    curr_frontier = [out.name for out in graph.output]
    next_frontier: list[str] = []
    results: list[NodeProto] = []
    depths: list[int] = []
    depth = 0
    while curr_frontier and (num is None or len(results) < num):
        for name in curr_frontier:
            node = backmap[name]
            if id(node) in explored:
                continue
            explored.add(id(node))
            if node.op_type in op_type:
                results.append(node)
                depths.append(depth)
            next_frontier.extend(inp for inp in node.input if inp in backmap)
        curr_frontier = next_frontier
        next_frontier = []
        depth += 1
    return results, depths
