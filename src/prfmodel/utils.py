"""Utility functions."""

import functools
import math
import re
import warnings
from abc import abstractmethod
from collections.abc import Callable
from typing import ClassVar
from typing import Protocol
from typing import runtime_checkable
import numpy as np
import pandas as pd
from keras import ops
from keras.config import floatx
from ._docstring import doc
from .stimuli import Stimulus
from .typing import Tensor

_EXPECTED_NDIM = 2

DTYPES = {"bfloat16", "float16", "float32", "float64"}
"""
Accepted dtypes for `prfmodel.typing.Tensor` objects.

Accepted dtypes are: `"bfloat16"`, `"float16"`, `"float32"`, and `"float64"`.

"""


class UndefinedResponseWarning(UserWarning):
    """Warning for when a response is undefined and contains NaNs."""


@runtime_checkable
class ModelProtocol(Protocol):
    """
    Protocol for model classes.

    Cannot be instantiated on its own.
    This protocol is intended to serve as the parent class for custom submodels within
    :class:`~prfmodel.models.base.BaseComposite`. Subclasses must override the abstract
    :attr:`parameter_names` property.

    Attributes
    ----------
    parameter_names : list of str
        Names of parameters used by the model class.

    Examples
    --------
    Create a custom object class that inherits from the base class:

    >>> class CustomModel(ModelProtocol):
    ...     @property
    ...     def parameter_names(self):
    ...         return ["a", "b"]
    >>> model = CustomModel()
    >>> print(model.parameter_names)
    ['a', 'b']

    """

    @property
    @abstractmethod
    def parameter_names(self) -> list[str]:
        """A list with names of parameters that are used by the model."""

    def _check_parameters(self, parameters: pd.DataFrame) -> None:
        missing_params = [param for param in self.parameter_names if param not in parameters.columns]
        if missing_params:
            msg = f"Missing required parameter names: {missing_params}"
            raise ValueError(msg)

    _positive_parameter_names: ClassVar[tuple[str, ...]] = ()
    """Parameters that must be strictly positive for the model to be defined."""

    def _check_parameter_values(self, parameters: pd.DataFrame) -> None:
        """Check that the parameter values lie inside the domain the model is defined on.

        Called from the public facades, which are never traced, so this may read a value back to a Python
        `bool` -- unlike anything reachable from `call`. A model that holds submodels overrides this to
        forward the call to them, because a fitter reaches a submodel through `call` and would otherwise
        bypass the check entirely.

        Subclasses declare the parameters this applies to with :attr:`_positive_parameter_names`. Names
        that are absent from `parameters` are skipped, so a parameter supplied by a default is only
        checked once the defaults have been merged in.

        """
        for name in self._positive_parameter_names:
            if name in parameters.columns and not (parameters[name].to_numpy() > 0.0).all():
                msg = f"Parameter '{name}' must be > 0"
                raise ValueError(msg)


@doc
def convert_parameters_to_tensor(parameters: pd.DataFrame, dtype: str) -> Tensor:
    """Convert model parameters in a dataframe into a tensor.

    Parameters
    ----------
    %(parameters)s

    Returns
    -------
    Tensor
        Tensor with the first axis corresponding to units and the second axis corresponding to different parameters.

    Examples
    --------
    Single parameters:

    >>> import pandas as pd
    >>> params = pd.DataFrame({
    ...     "param_1": [0.0, 1.0, 2.0],
    ... })
    >>> x = convert_parameters_to_tensor(params, dtype="float32")
    >>> print(x.shape)
    (3, 1)

    Multiple parameters:

    >>> params = pd.DataFrame({
    ...     "param_1": [0.0, 1.0, 2.0],
    ...     "param_2": [0.0, -1.0, -2.0],
    ... })
    >>> x = convert_parameters_to_tensor(params, dtype="float32")
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

    Splits the `parameters` argument (a :class:`pandas.DataFrame`) along the row (unit) dimension into
    chunks of size `batch_size`, calls `fn` for each chunk, and concatenates the results along the first axis.

    The wrapped function gains a ``batch_size`` keyword argument. When ``batch_size`` is ``None`` (the default),
    all units are processed in a single call.

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
    >>> batched_predict = batched(model)  # doctest: +SKIP
    >>> result = batched_predict(stimulus, parameters, batch_size=128)  # doctest: +SKIP

    As a decorator:

    >>> @batched
    ... def predict(stimulus, parameters, *, dtype=None):
    ...     ...
    >>> result = predict(stimulus, parameters, batch_size=64)  # doctest: +SKIP

    """

    @functools.wraps(fn)
    def wrapper(
        stimulus: Stimulus,
        parameters: pd.DataFrame,
        batch_size: int | None = None,
        **kwargs,
    ) -> Tensor:
        if batch_size is None:
            return fn(stimulus, parameters, **kwargs)

        num_units = len(parameters)
        num_batches = math.ceil(num_units / batch_size)

        results = []
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, num_units)
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
        Response with shape (num_units, num_frames).
    norm : str, optional, default="sum"
        Normalization to apply.

    Returns
    -------
    Tensor
        The normalized response with shape (num_units, num_frames).

    Notes
    -----
    A warning is raised when the normalization is zero which leads to an undefined normalized response.

    Examples
    --------
    >>> import numpy as np
    >>> from prfmodel.utils import normalize_response
    >>> response = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> normed = normalize_response(response, norm="sum")
    >>> print(normed.shape)
    (2, 3)
    >>> from keras import ops
    >>> print(round(float(ops.sum(normed[0])), 6))
    1.0

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

    This is the tensor-holding counterpart of the :class:`pandas.DataFrame` that models accept in their
    :meth:`call` implementation. Use :func:`prfmodel.utils.as_params` to build one from a data frame.

    Parameters
    ----------
    data : dict
        Dictionary of parameter tensors to perform column style selection on.
    %(dtype)s

    """

    def __init__(self, data: dict, dtype: str | None = None):
        dtype = get_dtype(dtype)

        # Build a new mapping rather than writing back into 'data'. Callers keep owning the dictionary they
        # pass -- the fitters hand in the same parameter mapping on every optimization step -- so mutating
        # the argument would leak tensors reshaped against an earlier call back to the caller.
        reshaped = {key: self._reshape_item(key, ops.convert_to_tensor(val, dtype=dtype)) for key, val in data.items()}

        item_shape = _get_common_shape(reshaped)

        self._data = {key: ops.broadcast_to(val, item_shape) for key, val in reshaped.items()}
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

    def __setitem__(self, key: str | list[str], value: Tensor | float) -> None:
        # A scalar is accepted so that a caller can fill a whole column with one value, which is how
        # 'BaseImpulse' merges its default parameters.
        item = ops.convert_to_tensor(value, dtype=self._dtype)

        value_ndim = len(item.shape)

        if isinstance(key, str) and (
            value_ndim < _EXPECTED_NDIM or (value_ndim == _EXPECTED_NDIM and item.shape[1] == 1)
        ):
            self._data[key] = ops.broadcast_to(self._reshape_item(key, item), self._item_shape)

        elif isinstance(key, list) and all(isinstance(k, str) for k in key) and value_ndim == _EXPECTED_NDIM:
            transposed = ops.transpose(ops.broadcast_to(item, (self._item_shape[0], len(key))))

            for _key, _val in zip(key, transposed, strict=True):
                self._data[_key] = _val

        else:
            msg = f"Value shape {item.shape} did not match the expected shape {self.shape}"
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
        return ParamsDict(self.to_dict(), dtype=self._dtype)

    def to_dict(self) -> dict:
        """Return the parameter tensors as a plain dictionary keyed by parameter name."""
        return dict(self._data)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the object into a dataframe."""
        return pd.DataFrame(self._data)


@doc
def as_params(parameters: "pd.DataFrame | ParamsDict", dtype: str) -> ParamsDict:
    """Convert user-supplied parameters into the tensor-holding representation.

    This converts the parameters from a model's user-facing
    :meth:`__call__` to its tensor-only :meth:`call` implementation.

    Parameters
    ----------
    parameters : pandas.DataFrame or ParamsDict
        Parameters to convert. A :class:`~prfmodel.utils.ParamsDict` is returned unchanged when it already
        carries `dtype`, and rebuilt with `dtype` otherwise.
    dtype : str
        The dtype that parameter tensors are converted to.

    Returns
    -------
    %(parameters_tensors)s

    Examples
    --------
    >>> import pandas as pd
    >>> from prfmodel.utils import as_params
    >>> params = as_params(pd.DataFrame({"sigma": [1.0, 1.5]}), dtype="float32")
    >>> print(params.shape)
    (2, 1)
    >>> as_params(params, dtype="float32") is params
    True

    """
    if isinstance(parameters, ParamsDict):
        if parameters.dtype == dtype:
            return parameters

        return ParamsDict(parameters.to_dict(), dtype=dtype)

    return ParamsDict(parameters.to_dict(orient="list"), dtype=dtype)
