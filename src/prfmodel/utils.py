"""Utility functions."""

import functools
import math
import re
import warnings
from collections.abc import Callable
import numpy as np
import pandas as pd
from keras import ops
from keras.config import floatx
from .stimulus import PRFStimulus
from .typing import Tensor

_EXPECTED_NDIM = 2

DTYPES = {"bfloat16", "float16", "float32", "float64"}
"""
Accepted dtypes for `prfmodel.typing.Tensor` objects.

Accepted dtypes are: `"bfloat16"`, `"float16"`, `"float32"`, and `"float64"`.

"""


class UndefinedResponseWarning(UserWarning):
    """Warning for when a response is undefined and contains NaNs."""


def convert_parameters_to_tensor(parameters: pd.DataFrame, dtype: str) -> Tensor:
    """Convert model parameters in a dataframe into a tensor.

    Parameters
    ----------
    parameters : pandas.DataFrame
        Dataframe with columns containing different model parameters and rows containing
        parameter values for different voxels.

    Returns
    -------
    Tensor
        Tensor with the first axis corresponding to voxels and the second axis corresponding to different parameters.

    Examples
    --------
    Single parameters:

    >>> import pandas as pd
    >>> params = pd.DataFrame({
    >>>     "param_1": [0.0, 1.0, 2.0],
    >>> })
    >>> x = convert_parameters_to_tensor(params)
    >>> print(x.shape)
    (3, 1)

    Multiple parameters:

    >>> params = pd.DataFrame({
    >>>     "param_1": [0.0, 1.0, 2.0],
    >>>     "param_2": [0.0, -1.0, -2.0],
    >>> })
    >>> x = covert_parameters_to_tensor(params)
    >>> print(x.shape)
    (3, 2)

    """
    return ops.convert_to_tensor(parameters, dtype=dtype)


def get_dtype(dtype: str | None) -> str:
    """Get the (default) dtype.

    Utility function to pass through a dtype or get the default dtype set by `keras.config.floatx()`.

    Parameters
    ----------
    dtype : str or None
        The dtype to pass through. If `None`, returns `keras.config.floatx()`.

    Returns
    -------
    str
        The dtype.

    Raises
    ------
    ValueError
        When `dtype` is not of the values defined in `DTYPES`.

    """
    if dtype is not None and dtype not in DTYPES:
        msg = f"Argument 'dtype' must be one of {DTYPES}"
        raise ValueError(msg)
    return dtype or floatx()


def batched(fn: Callable) -> Callable:
    """Decorate a model prediction function to make batched predictions.

    Splits the `parameters` argument (a :class:`pandas.DataFrame`) along the row (voxel) dimension into
    chunks of size `batch_size`, calls `fn` for each chunk, and concatenates the results along the first axis.

    The wrapped function gains a ``batch_size`` keyword argument. When ``batch_size`` is ``None`` (the default),
    all voxels are processed in a single call.

    Parameters
    ----------
    fn : callable
        A model prediction function with signature ``fn(stimulus, parameters, **kwargs)``.

    Returns
    -------
    callable
        Wrapped function with signature ``fn(stimulus, parameters, *, batch_size=None, **kwargs)``.

    Examples
    --------
    >>> from prfmodel.utils import batched
    >>> batched_predict = batched(model)
    >>> result = batched_predict(stimulus, parameters, batch_size=128)

    As a decorator:

    >>> @batched
    ... def predict(stimulus, parameters, *, dtype=None):
    ...     ...
    >>> result = predict(stimulus, parameters, batch_size=64)

    """

    @functools.wraps(fn)
    def wrapper(
        stimulus: PRFStimulus,
        parameters: pd.DataFrame,
        batch_size: int | None = None,
        **kwargs,
    ) -> Tensor:
        if batch_size is None:
            return fn(stimulus, parameters, **kwargs)

        num_voxels = len(parameters)
        num_batches = math.ceil(num_voxels / batch_size)

        results = []
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, num_voxels)
            batch_parameters = parameters.iloc[start:end]
            results.append(fn(stimulus, batch_parameters, **kwargs))

        return ops.concatenate(results, axis=0)

    return wrapper


def _get_common_shape(data: dict) -> tuple[int, ...]:
    shapes = [ops.convert_to_tensor(val).shape for val in data.values()]
    try:
        return np.broadcast_shapes(*shapes)
    except ValueError as exc:
        # Replace argnums with dict keys in error message
        def _replace_arg(match: re.Match) -> str:
            num = match.group(1)
            return f"arg {list(data.keys())[num]}"

        msg = re.sub(r"arg \d+", _replace_arg, repr(exc))
        raise ValueError(msg) from exc


def _get_norm_fun(norm: str) -> Callable:
    norm_dict = {
        "sum": ops.sum,
        "max": ops.max,
        "mean": ops.mean,
        "norm": ops.norm,
    }

    if norm not in norm_dict:
        msg = f"Argument 'norm' must be in {list(norm_dict.keys())}"
        raise ValueError(msg)

    return norm_dict[norm]


def normalize_response(response: Tensor, norm: str | None = "sum") -> Tensor:
    """
    Normalize a response.

    Divides a response by a normalization (e.g., its sum) computed over the second dimension.

    Parameters
    ----------
    response : Tensor
        Response with shape (num_voxels, num_frames).
    norm : str, optional, default="sum"
        Normalization to apply.

    Returns
    -------
    Tensor
        The normalized response with shape (num_voxels, num_frames).

    Notes
    -----
    A warning is raised when the normalization is zero which leads to an undefined normalized response.

    """
    response = ops.convert_to_tensor(response)
    response_ndim = ops.ndim(response)

    if response_ndim != _EXPECTED_NDIM:
        msg = f"Response must have two dimensions but {response_ndim} dimensions were found"
        raise ValueError(msg)

    if norm is None:
        return response

    norm_fun = _get_norm_fun(norm)

    response_norm = norm_fun(response, axis=1, keepdims=True)

    norm_is_zero = response_norm == ops.cast(0, response.dtype)

    if ops.any(norm_is_zero):
        msg = "Response norm is zero leading to undefined normalized responses"
        warnings.warn(message=msg, category=UndefinedResponseWarning)

    return response / response_norm


class ParamsDict:
    """
    A dictionary-like object that supports dataframe-style column selection but returns Keras tensors.

    Serves as an adapter during fitting to supply parameters to models while avoiding converting tensors into
    actual dataframes.

    Parameters
    ----------
    data : dict
        Dictionary of parameter tensors to perform column style selection on.
    dtype : str, optional
        The dtype that parameter tensors are converted to. If `None` (the default), uses the dtype from
        :func:`prfmodel.utils.get_dtype`.

    """

    def __init__(self, data: dict, dtype: str | None = None):
        dtype = get_dtype(dtype)

        for key, val in data.items():
            val_tensor = ops.convert_to_tensor(val, dtype=dtype)

            val_tensor = self._reshape_item(key, val_tensor)

            data[key] = val_tensor

        item_shape = _get_common_shape(data)

        for key, val in data.items():
            data[key] = ops.broadcast_to(val, item_shape)

        self._data = data
        self._item_shape = item_shape
        self._dtype = dtype

    @staticmethod
    def _reshape_item(key: str, value: Tensor) -> Tensor:
        new_value = ops.squeeze(value)
        # We cannot use tensor.ndim() because keras.Variable does not have this attribute
        value_ndim = len(new_value.shape)

        if value_ndim == 0:
            new_value = ops.expand_dims(new_value, 0)
        elif value_ndim > 1:
            msg = f"Data element {key} must be broadcastable to a single dimension but has shape {new_value.shape}"
            raise ValueError(msg)

        return new_value

    def __getitem__(self, key: str | list[str]) -> Tensor:
        if isinstance(key, str):
            return ops.convert_to_tensor(self._data[key], self._dtype)

        return ops.stack([self._data[key] for key in key], axis=1)

    def __setitem__(self, key: str | list[str], value: Tensor) -> None:
        value = ops.convert_to_tensor(value, dtype=self._dtype)

        value_ndim = len(value.shape)

        if isinstance(key, str) and (
            value_ndim < _EXPECTED_NDIM or (value_ndim == _EXPECTED_NDIM and value.shape[1] == 1)
        ):
            value = self._reshape_item(key, value)
            self._data[key] = ops.broadcast_to(value, self._item_shape)

        elif isinstance(key, list) and all(isinstance(k, str) for k in key) and value_ndim == _EXPECTED_NDIM:
            value = ops.transpose(ops.broadcast_to(value, (self._item_shape[0], len(key))))

            for _key, _val in zip(key, value, strict=True):
                self._data[_key] = _val

        else:
            msg = f"Value shape {value.shape} did not match the expected shape {self.shape}"
            raise ValueError(msg)

    @property
    def columns(self) -> list[str]:
        """Names of parameter columns."""
        return list(self._data.keys())

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the parameters (rows, columns)."""
        return (self._item_shape[0], len(self.columns))

    @property
    def dtype(self) -> str:
        """
        The dtype of the parameters.

        If `None`, uses `keras.config.floatx()` which defaults
        to `float32`.
        """
        return self._dtype

    def copy(self) -> "ParamsDict":
        """Create a copy of the object."""
        return ParamsDict(dict(self._data.items()))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the object into a dataframe."""
        return pd.DataFrame(self._data)
