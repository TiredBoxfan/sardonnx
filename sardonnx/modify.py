"""
Modifications to existing ONNX models.
"""

import itertools

from onnx import GraphProto


def set_batch(graph: GraphProto, value: int | str) -> None:
    """
    Sets the first dimension of the graph, persumed to be the batch dimension,
    in place.

    :param graph: The ONNX GraphProto object to modify.
    :param value: The value to set the first dimension to.
    """
    for val in itertools.chain(graph.input, graph.output, graph.value_info):
        if not val.HasField("tensor_type"):
            continue
        shape = val.type.tensor_type.shape
        if len(shape.dim) < 1:
            continue  # No batch dimension to set.

        dim = shape.dim[0]
        dim.ClearField("dim_value")
        dim.ClearField("dim_param")
        if isinstance(value, int):
            dim.dim_value = value
        else:
            dim.dim_param = value
