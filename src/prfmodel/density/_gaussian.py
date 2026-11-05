"""Gaussian density functions."""

import math
from keras import ops
from prfmodel.typing import Tensor


def normal_density(value: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    r"""
    Calculate the density of an isotropic multivariate normal distribution.

    The multivariate normal distribution has a diagonal covariance matrix with :math:`\mathtt{sigma}^2` on the
    diagonal (i.e., all dimensions have the same standard deviation `sigma`).

    Parameters
    ----------
    value : :data:`prfmodel.typing.Tensor`
        Values at which to evaluate the normal distribution. The last axis indexes the dimensions of the
        distribution.
    mu : :data:`prfmodel.typing.Tensor`
        Mean of the normal distribution. Must be broadcastable to the shape of `value`.
    sigma : :data:`prfmodel.typing.Tensor`
        Standard deviation of the normal distribution. Because the distribution is isotropic, `sigma` does not
        have a dimension axis and must be broadcastable to the shape of `value` without its last axis.

    Returns
    -------
    :data:`prfmodel.typing.Tensor`
        The normal density at `value` with the shape of `value` without its last axis.

    Notes
    -----
    The density of the isotropic multivariate normal distribution with mean :math:`\mu` and standard deviation
    :math:`\sigma` in :math:`k` dimensions is given by:

    .. math::

        f(x) = \frac{1}{(2 \pi \sigma^2)^{k / 2}} e^{-\frac{\lVert x - \mu \rVert^2}{2 \sigma^2}}.

    Examples
    --------
    >>> import numpy as np
    >>> from prfmodel.density import normal_density
    >>> value = np.zeros((4, 3, 2))   # 4 x 3 points in 2 dimensions
    >>> mu = np.array([0.0, 1.0])   # shape (2,)
    >>> sigma = np.array([[1.0]])   # shape (1, 1)
    >>> dens = normal_density(value, mu, sigma)
    >>> print(dens.shape)
    (4, 3)

    """
    value = ops.convert_to_tensor(value)
    mu = ops.convert_to_tensor(mu)
    sigma = ops.convert_to_tensor(sigma)

    num_dims = value.shape[-1]

    sigma_squared = ops.square(sigma)

    # Squared distance to the mean, summed over the dimensions of the distribution
    dist = ops.sum(ops.square(value - mu), axis=-1)
    dist /= 2 * sigma_squared

    # Divide by volume to normalize
    volume = (2 * math.pi * sigma_squared) ** (num_dims / 2)

    return ops.exp(-dist) / volume
