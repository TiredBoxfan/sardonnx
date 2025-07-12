"""
Modifications to existing ONNX models.
"""

import copy
import itertools
from io import BytesIO
from tempfile import NamedTemporaryFile

import onnx
from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer,
    RepeatedScalarFieldContainer,
)
from onnx import GraphProto, ModelProto, TensorProto, ValueInfoProto


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


def get_submodel(
    model: ModelProto,
    input_names: list[str] | None,
    output_names: list[str] | None,
) -> ModelProto:
    """
    Extracts a submodel from an ONNX model.

    :param model: The ONNX model to extract from.
    :param input_names: The names of the values to use as inputs.
        If `None`, the original inputs will be used.
    :param output_names: The names of the values to use as outputs.
        If `None`, the original outputs will be used.

    :return: The extracted submodel ModelProto.
    """
    if input_names is None:
        input_names = [inp.name for inp in model.graph.input]
    if output_names is None:
        output_names = [out.name for out in model.graph.output]
    with NamedTemporaryFile() as f_in, NamedTemporaryFile() as f_out:
        onnx.save_model(model, f_in.name)
        f_in.flush()
        onnx.utils.extract_model(f_in.name, f_out.name, input_names, output_names)
        return onnx.load_model(f_out.name)


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


def _namespace_prefix(
    name: str,
    prefix: str,
) -> str:
    """
    Prefixes a name with a given prefix using a content-appropriate separator.
    Names that start with '/' are treated as paths, while other names will use
    dot notation. Empty names are returned unchanged.

    :param name: The name to prefix.
    :param prefix: The prefix to apply.

    :return: The prefixed name.
    """
    if not name:
        return ""
    return f"/{prefix}{name}" if name.startswith("/") else f"{prefix}.{name}"


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


def attach_secondary(primary: GraphProto, secondary: GraphProto, prefix: str) -> None:
    """
    Attaches the secondary graph onto the primary graph in place, prefixing
    the components of the secondary graph to avoid collisions. It is assumed
    that the inputs to the secondary graph are already present in the supplied
    primary graph. No validation of the model is done.

    :param primary: The primary graph to attach to.
    :param secondary: The secondary graph to attach.
    :param prefix: The prefix to apply to the secondary graph.
    """
    prefix = prefix.strip("/. ")
    exclude = {inp.name for inp in secondary.input}

    def _add_group(
        target: RepeatedCompositeFieldContainer[ValueInfoProto | TensorProto],
        source: RepeatedCompositeFieldContainer[ValueInfoProto | TensorProto],
    ) -> None:
        """
        Adds prefixed deepcopies of the source protos to the target.

        :param target: The target to add to.
        :param source: The source to add from.
        """
        for val in source:
            if val.name in exclude:
                continue
            clone = copy.deepcopy(val)
            clone.name = _namespace_prefix(clone.name, prefix)
            target.append(clone)

    _add_group(primary.initializer, secondary.initializer)
    _add_group(primary.output, secondary.output)
    _add_group(primary.value_info, secondary.value_info)

    def _node_io_prefix(seq: RepeatedScalarFieldContainer[str]) -> None:
        """
        Prefixes a sequence of names with the given prefix in place.

        :param seq: The sequence of names to prefix.
        """
        for i, item in enumerate(seq):
            if item not in exclude:
                seq[i] = _namespace_prefix(item, prefix)

    for node in secondary.node:
        clone = copy.deepcopy(node)
        # Prefix node names consistently with / to simulate namespaces.
        clone.name = f"/{prefix}/{clone.name.lstrip('/')}"
        _node_io_prefix(clone.input)
        _node_io_prefix(clone.output)
        primary.node.append(clone)
