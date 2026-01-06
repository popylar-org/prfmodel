"""Density functions."""

from collections.abc import Callable
from keras import ops
from prfmodel.backend import gammaln
from prfmodel.typing import Tensor


def gamma_density(value: Tensor, shape: Tensor, rate: Tensor, norm: bool = True) -> Tensor:
    r"""
    Calculate the density of a gamma distribution.

    The distribution uses a shape and rate parameterization.
    Raises an error when evaluated at negative values.

    Parameters
    ----------
    value : Tensor
        The values at which to evaluate the gamma distribution. Must be > 0.
    shape : Tensor
        The shape parameter. Must be > 0.
    rate : Tensor
        The rate parameter. Must be > 0.
    norm : bool, default=True
        Whether to compute the normalized density.

    Returns
    -------
    Tensor
        The density of the gamma distribution at `value`.

    Notes
    -----
    The unnormalized density of the gamma distribution
    with `shape` :math:`\alpha` and `rate` :math:`\lambda` is given by:

    .. math::

        f(x) = x^{\mathtt{\alpha} - 1} e^{-\mathtt{\lambda} x}.

    When `norm=True`, the density is multiplied with a normalizing constant:

    .. math::

        f_{norm} = \frac{\mathtt{\lambda}^{\mathtt{\alpha}}}{\Gamma(\mathtt{\alpha})} * f(x).

    Raises
    ------
    ValueError
        If `values`, `shape`, or `rate` are zero or negative.

    """
    value = ops.convert_to_tensor(value)
    shape = ops.convert_to_tensor(shape)
    rate = ops.convert_to_tensor(rate)

    if not ops.all(value > 0.0):
        msg = "Values must be > 0"
        raise ValueError(msg)

    if not ops.all(shape > 0.0):
        msg = "Shape parameters must be > 0"
        raise ValueError(msg)

    if not ops.all(rate > 0.0):
        msg = "Rate parameters must be > 0"
        raise ValueError(msg)

    # Calculate log density and then exponentiate
    dens = (shape - 1) * ops.log(value) - rate * value

    if norm:
        # Normalize
        return ops.exp(shape * ops.log(rate) + dens - gammaln(shape))

    return ops.exp(dens)


def _shift_density(
    fun: Callable,
    value: Tensor,
    shift: Tensor,
    **kwargs,
) -> Tensor:
    value = ops.convert_to_tensor(value)
    shift = ops.convert_to_tensor(shift)

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
        The values at which to evaluate the shifted gamma distribution.
    shape : Tensor
        The shape parameter. Must be > 0.
    rate : Tensor
        The rate parameter. Must be > 0.
    shift : Tensor
        The shift parameter. When > 0, shifts the distribution to the right.
    norm : bool, default=True
        Whether to compute the normalized density.

    Returns
    -------
    Tensor
        The density of the shifted gamma distribution at `value`. The density for shifted values that are zero or lower
        is zero.

    See Also
    --------
    gamma_density : The (unshifted) gamma distribution density.

    """
    return _shift_density(gamma_density, value, shift, shape=shape, rate=rate, norm=norm)


def derivative_gamma_density(value: Tensor, shape: Tensor, rate: Tensor) -> Tensor:
    r"""
    Calculate the derivative density of a gamma distribution.

    The distribution uses a shape and rate parameterization.
    Raises an error when evaluated at negative values.

    Parameters
    ----------
    value : Tensor
        The values at which to evaluate the derivative gamma distribution. Must be > 0.
    shape : Tensor
        The shape parameter. Must be > 0.
    rate : Tensor
        The rate parameter. Must be > 0.

    Returns
    -------
    Tensor
        The derivative density of the gamma distribution at `value`.

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

    Raises
    ------
    ValueError
        If `values`, `shape`, or `rate` are zero or negative.

    See Also
    --------
    gamma_density : The gamma distribution density.

    """
    value = ops.convert_to_tensor(value)
    shape = ops.convert_to_tensor(shape)
    rate = ops.convert_to_tensor(rate)

    dens = gamma_density(value, shape, rate)

    # We express the derivative in terms of the pdf
    term_deriv = (shape - 1) / value - rate

    return dens * term_deriv


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
        The values at which to evaluate the derivative shifted gamma distribution.
    shape : Tensor
        The shape parameter. Must be > 0.
    rate : Tensor
        The rate parameter. Must be > 0.
    shift : Tensor
        The shift parameter. When > 0, shifts the distribution to the right.

    Returns
    -------
    Tensor
        The density of the shifted derivative gamma distribution at `value`.
        The density for shifted values that are zero or lower
        is zero.

    See Also
    --------
    derivative_gamma_density : The derivative gamma distribution density.
    shifted_gamma_density : The shifted gamma distribution density.

    """
    return _shift_density(derivative_gamma_density, value, shift, shape=shape, rate=rate)
