"""
Modifications to existing ONNX models.
"""

import itertools
from io import BytesIO

import onnx
from onnx import GraphProto, ModelProto


def clone_model(model: ModelProto) -> ModelProto:
    """
    Clones an ONNX model by serializing and deserializing it.

    :param model: The ONNX model to clone.

    :return: A deep copy of the ONNX ModelProto object.
    """
    buffer = BytesIO()
    onnx.save_model(model, buffer)
    buffer.seek(0)
    return onnx.load_model(buffer)


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


def rename_components(graph: GraphProto, mapping: dict[str, str]) -> None:
    """
    Renames the components of an ONNX graph in place for all proto types.
    No checks are done for collisions.

    :param graph: The ONNX GraphProto object to modify.
    :param mapping: A mapping of old names to new names. Components with names
        not in the mapping will not be modified.
    """
    for init in graph.initializer:
        init.name = mapping.get(init.name, init.name)

    for inp in graph.input:
        inp.name = mapping.get(inp.name, inp.name)

    for node in graph.node:
        node.name = mapping.get(node.name, node.name)
        for i, inp in enumerate(node.input):
            node.input[i] = mapping.get(inp, inp)
        for i, out in enumerate(node.output):
            node.output[i] = mapping.get(out, out)

    for out in graph.output:
        out.name = mapping.get(out.name, out.name)

    for val in graph.value_info:
        val.name = mapping.get(val.name, val.name)
