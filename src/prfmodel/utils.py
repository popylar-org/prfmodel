"""Utility functions."""

from collections.abc import Sequence
import pandas as pd
from keras import ops
from keras.config import floatx
from .typing import Tensor

DTYPES = {"bfloat16", "float16", "float32", "float64"}
"""
Accepted dtypes for `prfmodel.typing.Tensor` objects.

Accepted dtypes are: `"bfloat16"`, `"float16"`, `"float32"`, and `"float64"`.

"""


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


class ParamsDict:
    """
    A dictionary-like object that supports dataframe-style column selection but returns Keras tensors.

    Serves as an adapter during fitting to supply parameters to models while avoiding converting tensors into
    actual dataframes.

    Parameters
    ----------
    data: dict
        Dictionary of parameter tensors to perform column style selection on.

    """

    def __init__(self, data: dict):
        self._data = data

    def __getitem__(self, keys: str | Sequence[str]) -> Tensor:
        if isinstance(keys, str):
            return ops.convert_to_tensor(self._data[keys])

        return ops.transpose(ops.convert_to_tensor([self._data[key] for key in keys]))

    def __setitem__(self, keys: str | Sequence[str], values: Tensor | Sequence[Tensor]) -> None:
        if isinstance(keys, str):
            keys = [keys]

        if not isinstance(values, Sequence):
            values = [values]

        for key, val in zip(keys, values, strict=True):
            self._data[key] = val

    def copy(self) -> "ParamsDict":
        """Create a copy of the object."""
        return ParamsDict(dict(self._data.items()))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the object into a dataframe."""
        return pd.DataFrame(self._data)
