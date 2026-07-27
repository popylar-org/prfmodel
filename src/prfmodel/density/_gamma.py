"""Gamma density functions."""

from collections.abc import Callable
from keras import ops
from prfmodel._backend import gammaln
from prfmodel.exceptions import ShapeMismatchError
from prfmodel.typing import Tensor

_ARG_DIM = 2


def _check_parameter_shape(param: Tensor, name: str) -> None:
    if (param.shape != () and len(param.shape) != _ARG_DIM) or (len(param.shape) == _ARG_DIM and param.shape[1] != 1):
        msg = f"{name} parameter must have shape () or (n, 1) but has shape {param.shape}"
        raise ValueError(msg)


def _check_gamma_density_input(
    value: Tensor,
    shape: Tensor,
    scale: Tensor,
    shift: Tensor | None = None,
) -> None:
    _check_parameter_shape(shape, "Shape")
    _check_parameter_shape(scale, "Scale")

    if shift is not None:
        shift = ops.convert_to_tensor(shift)
        _check_parameter_shape(shift, "Shift")

        if shape.shape != scale.shape:
            raise ShapeMismatchError("shape", shape.shape, "scale", scale.shape)  # noqa: EM101 (exception literal)
        if shape.shape != shift.shape:
            raise ShapeMismatchError("shape", shape.shape, "shift", shift.shape)  # noqa: EM101 (exception literal)
    else:
        if shape.shape != scale.shape:
            raise ShapeMismatchError("shape", shape.shape, "scale", scale.shape)  # noqa: EM101 (exception literal)

        if (value.shape != () and len(value.shape) != _ARG_DIM) or (
            len(value.shape) == _ARG_DIM and value.shape[0] != 1
        ):
            msg = f"Value must have shape () or (1, m) but has shape {value.shape}"
            raise ValueError(msg)

    if not ops.all(value > 0.0):
        msg = "Value must be > 0"
        raise ValueError(msg)

    if not ops.all(shape > 0.0):
        msg = "Shape parameter must be > 0"
        raise ValueError(msg)

    if not ops.all(scale > 0.0):
        msg = "Scale parameter must be > 0"
        raise ValueError(msg)


def _gamma_density(value: Tensor, shape: Tensor, scale: Tensor, norm: bool = True) -> Tensor:
    # Calculate log density and then exponentiate
    dens = (shape - 1) * ops.log(value) - value / scale

    if norm:
        # Normalize
        return ops.exp(dens - shape * ops.log(scale) - gammaln(shape))

    return ops.exp(dens)


def gamma_density(value: Tensor, shape: Tensor, scale: Tensor, norm: bool = True) -> Tensor:
    r"""
    Calculate the density of a gamma distribution.

    The distribution uses a shape and scale parameterization.
    Raises an error when evaluated at negative values.

    Parameters
    ----------
    value : :data:`prfmodel.typing.Tensor`
        The values at which to evaluate the gamma distribution. Must be > 0 and scalar or with shape (1, m).
    shape : :data:`prfmodel.typing.Tensor`
        The shape parameter. Must be > 0 with shape () and scalar or with shape (n, 1).
    scale : :data:`prfmodel.typing.Tensor`
        The scale parameter. Must be > 0 and scalar or with shape (n, 1).
    norm : bool, default=True
        Whether to compute the normalized density.

    Returns
    -------
    :data:`prfmodel.typing.Tensor`
        The density of the gamma distribution at `value` as a scalar or with shape (n, m).

    Notes
    -----
    The unnormalized density of the gamma distribution
    with `shape` :math:`\alpha` and `scale` :math:`\theta` is given by:

    .. math::

        f(x) = x^{\mathtt{\alpha} - 1} e^{-x / \mathtt{\theta}}.

    When `norm=True`, the density is multiplied with a normalizing constant:

    .. math::

        f_{norm} = \frac{1}{\mathtt{\theta}^{\mathtt{\alpha}} \Gamma(\mathtt{\alpha})} * f(x).

    Examples
    --------
    >>> import numpy as np
    >>> from prfmodel.density import gamma_density
    >>> t = np.array([[1.0, 2.0, 3.0]])   # shape (1, 3)
    >>> shape = np.array([[2.0], [4.0]])   # shape (2, 1)
    >>> scale = np.array([[1.0], [1.0]])    # shape (2, 1)
    >>> dens = gamma_density(t, shape, scale)
    >>> print(dens.shape)
    (2, 3)

    """
    value = ops.convert_to_tensor(value)
    shape = ops.convert_to_tensor(shape)
    scale = ops.convert_to_tensor(scale)

    _check_gamma_density_input(value, shape, scale)

    return _gamma_density(value, shape, scale, norm)


def _shift_density(
    fun: Callable,
    value: Tensor,
    shift: Tensor,
    **kwargs,
) -> Tensor:
    value_shifted = value - shift
    value_shifted_is_positive = value_shifted > 0.0
    # Replace values <= 0 with ones and replace their density later with zeros
    value_shifted_valid = ops.where(value_shifted_is_positive, value_shifted, 1.0)

    return ops.where(value_shifted_is_positive, fun(value_shifted_valid, **kwargs), 0.0)


def shifted_gamma_density(
    value: Tensor,
    shape: Tensor,
    scale: Tensor,
    shift: Tensor,
    norm: bool = True,
) -> Tensor:
    """
    Calculate the density of a shifted gamma distribution.

    The gamma distribution is shifted by `shift` and padded with zeros if necessary.

    Parameters
    ----------
    value : :data:`prfmodel.typing.Tensor`
        The values at which to evaluate the shifted gamma distribution. Must be scalar or with shape (1, m).
    shape : :data:`prfmodel.typing.Tensor`
        The shape parameter. Must be > 0 and scalar or with shape (n, 1).
    scale : :data:`prfmodel.typing.Tensor`
        The scale parameter. Must be > 0 and scalar or with shape (n, 1).
    shift : :data:`prfmodel.typing.Tensor`
        The shift parameter. When > 0, shifts the distribution to the right.
    norm : bool, default=True
        Whether to compute the normalized density.

    Returns
    -------
    :data:`prfmodel.typing.Tensor`
        The density of the shifted gamma distribution at `value` as a scalar or with shape (n, m).
        The density for shifted values that are zero or lower is zero.

    See Also
    --------
    gamma_density : The (unshifted) gamma distribution density.

    Examples
    --------
    >>> import numpy as np
    >>> from prfmodel.density import shifted_gamma_density
    >>> t = np.array([[1.0, 2.0, 3.0]])   # shape (1, 3)
    >>> shape = np.array([[2.0], [4.0]])   # shape (2, 1)
    >>> scale = np.array([[1.0], [1.0]])    # shape (2, 1)
    >>> shift = np.array([[0.5], [0.0]])   # shape (2, 1)
    >>> dens = shifted_gamma_density(t, shape, scale, shift)
    >>> print(dens.shape)
    (2, 3)

    """
    value = ops.convert_to_tensor(value)
    shape = ops.convert_to_tensor(shape)
    scale = ops.convert_to_tensor(scale)
    shift = ops.convert_to_tensor(shift)

    _check_gamma_density_input(value, shape, scale, shift)

    return _shift_density(_gamma_density, value, shift, shape=shape, scale=scale, norm=norm)


def _derivative_gamma_density(value: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    dens = _gamma_density(value, shape, scale)

    # We express the derivative in terms of the pdf
    term_deriv = (shape - 1) / value - 1 / scale

    return dens * term_deriv


def derivative_gamma_density(value: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    r"""
    Calculate the derivative density of a gamma distribution.

    The distribution uses a shape and scale parameterization.
    Raises an error when evaluated at negative values.

    Parameters
    ----------
    value : :data:`prfmodel.typing.Tensor`
        The values at which to evaluate the derivative gamma distribution. Must be > 0 and scalar or with shape (1, m).
    shape : :data:`prfmodel.typing.Tensor`
        The shape parameter. Must be > 0 and scalar or with shape (n, m).
    scale : :data:`prfmodel.typing.Tensor`
        The scale parameter. Must be > 0 and scalar or with shape (n, m).

    Returns
    -------
    :data:`prfmodel.typing.Tensor`
        The derivative density of the gamma distribution at `value` as a scalar or with shape (n, m).

    Notes
    -----
    The density of the gamma distribution
    with `shape` :math:`\alpha` and `scale` :math:`\theta` is given by:

    .. math::

        f(x) =  \frac{1}{\mathtt{\theta}^{\mathtt{\alpha}} \Gamma(\mathtt{\alpha})}
        x^{\mathtt{\alpha} - 1} e^{-x / \mathtt{\theta}}.

    The derivative of the density with respect to :math:`x` can be defined as a function of the original density
    :math:`f(x)`:

    .. math::

        f(x)' = f(x) \frac{(\alpha - 1)}{t} - \frac{1}{\theta}

    See Also
    --------
    gamma_density : The gamma distribution density.

    Examples
    --------
    >>> import numpy as np
    >>> from prfmodel.density import derivative_gamma_density
    >>> t = np.array([[1.0, 2.0, 3.0]])   # shape (1, 3)
    >>> shape = np.array([[2.0], [4.0]])   # shape (2, 1)
    >>> scale = np.array([[1.0], [1.0]])    # shape (2, 1)
    >>> dens = derivative_gamma_density(t, shape, scale)
    >>> print(dens.shape)
    (2, 3)

    """
    value = ops.convert_to_tensor(value)
    shape = ops.convert_to_tensor(shape)
    scale = ops.convert_to_tensor(scale)

    _check_gamma_density_input(value, shape, scale)

    return _derivative_gamma_density(value, shape, scale)
