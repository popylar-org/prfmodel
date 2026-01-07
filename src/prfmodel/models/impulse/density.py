"""Density functions."""

from collections.abc import Callable
from keras import ops
from prfmodel.backend import gammaln
from prfmodel.models.base import BatchDimensionError
from prfmodel.typing import Tensor

_ARG_DIM = 2


def _check_parameter_shape(param: Tensor, name: str) -> None:
    if (param.shape != () and len(param.shape) != _ARG_DIM) or (len(param.shape) == _ARG_DIM and param.shape[1] != 1):
        msg = f"{name} parameter must have shape () or (n, 1) but has shape {param.shape}"
        raise ValueError(msg)


def _check_gamma_density_input(
    value: Tensor,
    shape: Tensor,
    rate: Tensor,
    shift: Tensor | None = None,
) -> None:
    _check_parameter_shape(shape, "Shape")
    _check_parameter_shape(rate, "Rate")

    if shift is not None:
        shift = ops.convert_to_tensor(shift)
        _check_parameter_shape(shift, "Shift")

        if shape.shape != rate.shape or shape.shape != shift.shape:
            raise BatchDimensionError(
                ["shape", "rate", "shift"],
                [shape.shape, rate.shape, shift.shape],
            )
    else:
        if shape.shape != rate.shape:
            raise BatchDimensionError(
                ["shape", "rate"],
                [shape.shape, rate.shape],
            )

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

    if not ops.all(rate > 0.0):
        msg = "Rate parameter must be > 0"
        raise ValueError(msg)


def _gamma_density(value: Tensor, shape: Tensor, rate: Tensor, norm: bool = True) -> Tensor:
    # Calculate log density and then exponentiate
    dens = (shape - 1) * ops.log(value) - rate * value

    if norm:
        # Normalize
        return ops.exp(shape * ops.log(rate) + dens - gammaln(shape))

    return ops.exp(dens)


def gamma_density(value: Tensor, shape: Tensor, rate: Tensor, norm: bool = True) -> Tensor:
    r"""
    Calculate the density of a gamma distribution.

    The distribution uses a shape and rate parameterization.
    Raises an error when evaluated at negative values.

    Parameters
    ----------
    value : Tensor
        The values at which to evaluate the gamma distribution. Must be > 0 and scalar or with shape (1, m).
    shape : Tensor
        The shape parameter. Must be > 0 with shape () and scalar or with shape (n, 1).
    rate : Tensor
        The rate parameter. Must be > 0 and scalar or with shape (n, 1).
    norm : bool, default=True
        Whether to compute the normalized density.

    Returns
    -------
    Tensor
        The density of the gamma distribution at `value` as a scalar or with shape (n, m).

    Notes
    -----
    The unnormalized density of the gamma distribution
    with `shape` :math:`\alpha` and `rate` :math:`\lambda` is given by:

    .. math::

        f(x) = x^{\mathtt{\alpha} - 1} e^{-\mathtt{\lambda} x}.

    When `norm=True`, the density is multiplied with a normalizing constant:

    .. math::

        f_{norm} = \frac{\mathtt{\lambda}^{\mathtt{\alpha}}}{\Gamma(\mathtt{\alpha})} * f(x).

    """
    value = ops.convert_to_tensor(value)
    shape = ops.convert_to_tensor(shape)
    rate = ops.convert_to_tensor(rate)

    _check_gamma_density_input(value, shape, rate)

    return _gamma_density(value, shape, rate, norm)


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
    rate: Tensor,
    shift: Tensor,
    norm: bool = True,
) -> Tensor:
    """
    Calculate the density of a shifted gamma distribution.

    The gamma distribution is shifted by `shift` and padded with zeros if necessary.

    Parameters
    ----------
    value : Tensor
        The values at which to evaluate the shifted gamma distribution. Must be scalar or with shape (1, m).
    shape : Tensor
        The shape parameter. Must be > 0 and scalar or with shape (n, 1).
    rate : Tensor
        The rate parameter. Must be > 0 and scalar or with shape (n, 1).
    shift : Tensor
        The shift parameter. When > 0, shifts the distribution to the right.
    norm : bool, default=True
        Whether to compute the normalized density.

    Returns
    -------
    Tensor
        The density of the shifted gamma distribution at `value` as a scalar or with shape (n, m).
        The density for shifted values that are zero or lower is zero.

    See Also
    --------
    gamma_density : The (unshifted) gamma distribution density.

    """
    value = ops.convert_to_tensor(value)
    shape = ops.convert_to_tensor(shape)
    rate = ops.convert_to_tensor(rate)
    shift = ops.convert_to_tensor(shift)

    _check_gamma_density_input(value, shape, rate, shift)

    return _shift_density(_gamma_density, value, shift, shape=shape, rate=rate, norm=norm)


def _derivative_gamma_density(value: Tensor, shape: Tensor, rate: Tensor) -> Tensor:
    dens = _gamma_density(value, shape, rate)

    # We express the derivative in terms of the pdf
    term_deriv = (shape - 1) / value - rate

    return dens * term_deriv


def derivative_gamma_density(value: Tensor, shape: Tensor, rate: Tensor) -> Tensor:
    r"""
    Calculate the derivative density of a gamma distribution.

    The distribution uses a shape and rate parameterization.
    Raises an error when evaluated at negative values.

    Parameters
    ----------
    value : Tensor
        The values at which to evaluate the derivative gamma distribution. Must be > 0 and scalar or with shape (1, m).
    shape : Tensor
        The shape parameter. Must be > 0 and scalar or with shape (n, m).
    rate : Tensor
        The rate parameter. Must be > 0 and scalar or with shape (n, m).

    Returns
    -------
    Tensor
        The derivative density of the gamma distribution at `value` as a scalar or with shape (n, m).

    Notes
    -----
    The density of the gamma distribution
    with `shape` :math:`\alpha` and `rate` :math:`\lambda` is given by:

    .. math::

        f(x) =  \frac{\mathtt{\lambda}^{\mathtt{\alpha}}}{\Gamma(\mathtt{\alpha})}
        x^{\mathtt{\alpha} - 1} e^{\mathtt{\lambda} x}.

    The derivative of the density with respect to :math:`x` can be defined as a function of the original density
    :math:`f(x)`:

    .. math::

        f(x)' = f(x) \frac{(\alpha - 1)}{t} - \lambda

    See Also
    --------
    gamma_density : The gamma distribution density.

    """
    value = ops.convert_to_tensor(value)
    shape = ops.convert_to_tensor(shape)
    rate = ops.convert_to_tensor(rate)

    _check_gamma_density_input(value, shape, rate)

    return _derivative_gamma_density(value, shape, rate)


def shifted_derivative_gamma_density(
    value: Tensor,
    shape: Tensor,
    rate: Tensor,
    shift: Tensor,
) -> Tensor:
    """
    Calculate the density of a shifted derivative gamma distribution.

    The derivative of the gamma distribution is shifted by `shift` and padded with zeros if necessary.

    Parameters
    ----------
    value : Tensor
        The values at which to evaluate the derivative shifted gamma distribution. Must be scalar or with shape (1, m).
    shape : Tensor
        The shape parameter. Must be > 0 and scalar or with shape (n, 1).
    rate : Tensor
        The rate parameter. Must be > 0 and scalar or with shape (n, 1).
    shift : Tensor
        The shift parameter. Must be scalar or with shape (n, 1). When > 0, shifts the distribution to the right.

    Returns
    -------
    Tensor
        The density of the shifted derivative gamma distribution at `value` as a scalar or with shape (n, m).
        The density for shifted values that are zero or lower
        is zero.

    See Also
    --------
    derivative_gamma_density : The derivative gamma distribution density.
    shifted_gamma_density : The shifted gamma distribution density.

    """
    value = ops.convert_to_tensor(value)
    shape = ops.convert_to_tensor(shape)
    rate = ops.convert_to_tensor(rate)
    shift = ops.convert_to_tensor(shift)

    _check_gamma_density_input(value, shape, rate, shift)

    return _shift_density(_derivative_gamma_density, value, shift, shape=shape, rate=rate)
