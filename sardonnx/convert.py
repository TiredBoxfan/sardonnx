"""
Conversion utilities for converting between supported frameworks and ONNX.
"""

import inspect
from io import BytesIO
from typing import Iterable, Union

import numpy as np
import onnx
from onnx import ModelProto
from typing_extensions import TypeAlias

from sardonnx.modify import set_batch

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import tf2onnx
except ImportError:
    tf2onnx = None


TensorType: TypeAlias = Union[np.ndarray, "torch.Tensor", "tf.Tensor"]
ModelType: TypeAlias = Union[ModelProto, "torch.nn.Module", "tf.keras.Model"]


def convert_tensor(tensor: TensorType, target_type: type) -> TensorType:
    """
    Converts a tensor from one framework to another.

    :param tensor: The tensor to convert.
    :param target_type: The type to convert to.

    :return: The converted tensor.

    :raises TypeError: If the input tensor type is not supported.
    :raises ValueError: If the target type is not supported.
    """
    # Early return if already desired type.
    if isinstance(tensor, target_type):
        return tensor

    # Convert to NumPy.
    if isinstance(tensor, np.ndarray):
        array = tensor
    elif torch and isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
    elif tf and isinstance(tensor, tf.Tensor):
        array = tensor.numpy()
    else:
        raise TypeError(f"Unsupported input tensor type: {type(tensor)}")

    # Return target type.
    if target_type is np.ndarray:
        return array
    if torch and target_type is torch.Tensor:
        return torch.from_numpy(array)
    if tf and target_type is tf.Tensor:
        return tf.convert_to_tensor(array)
    raise ValueError(f"Unsupported target type: {target_type}")


def torch_to_onnx(
    model: "torch.nn.Module",
    inputs: Iterable[TensorType],
    opset: int,
    dynamic_batch: bool = True,
) -> ModelProto:
    """
    Converts a PyTorch model to an ONNX model.

    :param model: The PyTorch model to convert.
    :param inputs: Sample inputs to the model matching the model's forward
        signature as an iterable sequence, even if only one input is provided.
    :param opset: The ONNX opset version to use.
    :param dynamic_batch: Whether to interpret the first dimension of the inputs
        as a dynamic batch size.

    :return: The converted ONNX ModelProto object.
    """
    # Get the device of the model.
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = "cpu"

    # Preprocess inputs.
    args = tuple(convert_tensor(inp, torch.Tensor).to(device) for inp in inputs)
    input_names = list(inspect.signature(model.forward).parameters.keys())

    # Convert to ONNX.
    buffer = BytesIO()
    torch.onnx.export(
        model,
        args,
        buffer,
        opset_version=opset,
        input_names=input_names,
        dynamic_axes=(
            {name: {0: "batch_size"} for name in input_names} if dynamic_batch else None
        ),
    )
    buffer.seek(0)
    return onnx.load_model(buffer)


def _keras_sequential_to_functional(model: "tf.keras.Sequential") -> "tf.keras.Model":
    """
    Converts a Keras Sequential model to an equivalent Functional model.

    :param model: The Keras Sequential model to convert.

    :return: The converted Keras Functional model.
    """
    inputs = tf.keras.Input(shape=model.input_shape[1:])
    outputs = inputs
    for layer in model.layers:
        outputs = layer(outputs)
    return tf.keras.Model(
        inputs=inputs, outputs=outputs, name=model.name + "_functional"
    )


def keras_to_onnx(
    model: "tf.keras.Model",
    opset: int,
    dynamic_batch: bool = True,
) -> ModelProto:
    """
    Converts a Keras model to an ONNX model.

    :param model: The Keras model to convert.
    :param opset: The ONNX opset version to use.
    :param dynamic_batch: Whether to interpret the first dimension of the inputs
        as a dynamic batch size.

    :return: The converted ONNX ModelProto object.

    :raises ValueError: If the model's input is not defined.
    """
    # Model must use the functional API.
    if isinstance(model, tf.keras.Sequential):
        model = _keras_sequential_to_functional(model)

    if not model.inputs:
        raise ValueError("Model has no inputs to infer the signature from.")

    signature = [
        tf.TensorSpec(
            shape=[None if dynamic_batch else inp.shape[0] or 1, *inp.shape[1:]],
            dtype=inp.dtype,
            name=inp.name,
        )
        for inp in model.inputs
    ]
    onnx_model, _ = tf2onnx.convert.from_keras(model, signature, opset)
    return onnx_model


def to_onnx(
    model: ModelType,
    inputs: Iterable[TensorType],
    opset: int,
    dynamic_batch: bool = True,
) -> ModelProto:
    """
    Converts a model from any supported framework to an ONNX model.
    If the model is already an ONNX model, it will be cloned then updated
    accordingly.

    :param model: The model to convert.
    :param inputs: Sample inputs to the model matching the model's forward
        signature as an iterable sequence, even if only one input is provided.
    :param opset: The ONNX opset version to use.
    :param dynamic_batch: Whether to interpret the first dimension of the inputs
        as a dynamic batch size.
    """
    if isinstance(model, ModelProto):
        # `convert_version` does NOT modify in place.
        model = onnx.version_converter.convert_version(model, opset)
        if dynamic_batch:
            set_batch(model.graph, "batch_size")
        return model
    if torch and isinstance(model, torch.nn.Module):
        return torch_to_onnx(model, inputs, opset, dynamic_batch)
    if tf and isinstance(model, tf.keras.Model):
        return keras_to_onnx(model, opset, dynamic_batch)
    raise ValueError(f"Unsupported model type: {type(model)}")
