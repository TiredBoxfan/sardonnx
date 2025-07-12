"""
Conversion utilities for converting between supported frameworks and ONNX.
"""

from typing import Union

import numpy as np
from onnx import ModelProto
from typing_extensions import TypeAlias

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

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
