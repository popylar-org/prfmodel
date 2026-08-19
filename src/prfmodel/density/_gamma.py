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


def _check_gamma_density_shapes(
    value: Tensor,
    shape: Tensor,
    scale: Tensor,
    shift: Tensor | None = None,
) -> None:
    """Check the static shapes of the gamma density arguments.

    Reads `.shape` only, which is known while a graph is being built, so this is safe to call from a
    compiled function and runs once per trace rather than once per step.

    """
    _check_parameter_shape(shape, "Shape")
    _check_parameter_shape(scale, "Scale")

    if shape.shape != scale.shape:
        raise ShapeMismatchError("shape", shape.shape, "scale", scale.shape)  # noqa: EM101 (exception literal)

    if shift is not None:
        shift = ops.convert_to_tensor(shift)
        _check_parameter_shape(shift, "Shift")

        if shape.shape != shift.shape:
            raise ShapeMismatchError("shape", shape.shape, "shift", shift.shape)  # noqa: EM101 (exception literal)

    if (value.shape != () and len(value.shape) != _ARG_DIM) or (len(value.shape) == _ARG_DIM and value.shape[0] != 1):
        msg = f"Value must have shape () or (1, m) but has shape {value.shape}"
        raise ValueError(msg)


def _mask_nonpositive(fun: Callable, value: Tensor, **kwargs) -> Tensor:
    """Evaluate `fun` where `value` is positive and return zero everywhere else.

    A gamma distribution is supported on the positive reals, and its density is zero below that. The
    naive expression instead takes the logarithm of a non-positive number and yields `NaN`, so the
    non-positive entries are replaced by ones, evaluated along with the rest, and then discarded.
    Substituting rather than skipping matters for the gradient too: `ops.where` propagates the gradient
    of *both* branches, so a `NaN` produced in the discarded branch would still poison it.

    """
    is_positive = value > 0.0
    value_valid = ops.where(is_positive, value, 1.0)

    return ops.where(is_positive, fun(value_valid, **kwargs), 0.0)


def _gamma_density_on_support(value: Tensor, shape: Tensor, scale: Tensor, norm: bool = True) -> Tensor:
    """Calculate the gamma density for `value` > 0, which callers reach through `_gamma_density`."""
    # Calculate log density and then exponentiate
    dens = (shape - 1) * ops.log(value) - value / scale

    if norm:
        # Normalize
        return ops.exp(dens - shape * ops.log(scale) - gammaln(shape))

    return ops.exp(dens)


def _gamma_density(value: Tensor, shape: Tensor, scale: Tensor, norm: bool = True) -> Tensor:
    return _mask_nonpositive(_gamma_density_on_support, value, shape=shape, scale=scale, norm=norm)


def gamma_density(value: Tensor, shape: Tensor, scale: Tensor, norm: bool = True) -> Tensor:
    r"""
    Calculate the density of a gamma distribution.

    The distribution uses a shape and scale parameterization.

    Parameters
    ----------
    value : :data:`prfmodel.typing.Tensor`
        The values at which to evaluate the gamma distribution. Must be scalar or with shape ``(1, m)``.
        Values at or below zero are allowed and give a density of zero.
    shape : :data:`prfmodel.typing.Tensor`
        The shape parameter. Must be scalar or with shape ``(n, 1)``.
    scale : :data:`prfmodel.typing.Tensor`
        The scale parameter. Must be scalar or with shape ``(n, 1)``.
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

    The density is zero at and below values of zero, matching `scipy.stats.gamma.pdf`. Negative `shape` returns a
    finite but meaningless number whereas negative `scale`returns NaN instead of raising an error.

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

    _check_gamma_density_shapes(value, shape, scale)

    return _gamma_density(value, shape, scale, norm)


def _shift_density(
    fun: Callable,
    value: Tensor,
    shift: Tensor,
    **kwargs,
) -> Tensor:
    # Shifting moves the support with the distribution, so the same mask applies to the shifted value.
    # `fun` is the density on its support, so masking happens exactly once.
    return _mask_nonpositive(fun, value - shift, **kwargs)


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
        Values at or below `shift` are allowed and give a density of zero.
    shape : :data:`prfmodel.typing.Tensor`
        The shape parameter. Must be scalar or with shape (n, 1).
    scale : :data:`prfmodel.typing.Tensor`
        The scale parameter. Must be scalar or with shape (n, 1).
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

    Notes
    -----
    The density is zero for values at and below `shift`, matching `scipy.stats.gamma.pdf`. Negative `shape` returns
    a finite but meaningless number whereas negative `scale`returns NaN instead of raising an error.

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

    _check_gamma_density_shapes(value, shape, scale, shift)

    return _shift_density(_gamma_density_on_support, value, shift, shape=shape, scale=scale, norm=norm)


def _derivative_gamma_density_on_support(value: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    """Calculate the derivative gamma density for `value` > 0, reached through `_derivative_gamma_density`."""
    dens = _gamma_density_on_support(value, shape, scale)

    # We express the derivative in terms of the pdf
    term_deriv = (shape - 1) / value - 1 / scale

    return dens * term_deriv


def _derivative_gamma_density(value: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    # Masking the composed expression rather than reusing '_gamma_density' is what keeps 'value' = 0
    # finite: the density is zero there and 'term_deriv' is infinite, and their product is a 'NaN'.
    return _mask_nonpositive(_derivative_gamma_density_on_support, value, shape=shape, scale=scale)


def derivative_gamma_density(value: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    r"""
    Calculate the derivative density of a gamma distribution.

    The distribution uses a shape and scale parameterization.

    Parameters
    ----------
    value : :data:`prfmodel.typing.Tensor`
        The values at which to evaluate the derivative gamma distribution. Must be scalar or with shape
        (1, m). Values at or below zero are allowed and give a derivative density of zero.
    shape : :data:`prfmodel.typing.Tensor`
        The shape parameter. Must be scalar or with shape (n, m).
    scale : :data:`prfmodel.typing.Tensor`
        The scale parameter. Must be scalar or with shape (n, m).

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

    The derivative density is zero for values at and below zero, including at
    exactly zero where the density is zero and the derivative term is infinite. Negative `shape` returns a
    finite but meaningless number whereas negative `scale`returns NaN instead of raising an error.

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

    _check_gamma_density_shapes(value, shape, scale)

    return _derivative_gamma_density(value, shape, scale)
