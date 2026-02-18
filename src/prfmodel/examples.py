"""Example stimuli and data."""

from importlib.resources import files
import numpy as np
from prfmodel.stimuli.prf import PRFStimulus


def load_2d_prf_bar_stimulus() -> PRFStimulus:
    """
    Load a two-dimensional population receptive field bar stimulus.

    Loads an example stimulus that moves in eight directions (vertical, horizontal, diagonal) through a two-dimensional
    space.

    Returns
    -------
    PRFStimulus
        A stimulus object with a `design` with shape (200, 101, 101) and a `grid` with shape (101, 101, 2).

    Notes
    -----
    The stimulus was created with the validation framework developed by Lerma-Usabiaga et al (2020)[1]_.
    It has 200 time frames, with 101 pixels in the x- and y-dimension, spanning a visual field of 20 degrees
    vertically and horizontally.

    See Also
    --------
    prfmodel.stimuli.prf.animate_2d_prf_stimulus : Create an animation for a two-dimensional stimulus.

    References
    ----------
    .. [1] Lerma-Usabiaga, G., Benson, N., Winawer, J., & Wandell, B. A. (2020). A validation framework for
        neuroimaging software: The case of population receptive fields. *PLOS Computational Biology, 16*(6),
        e1007924. https://doi.org/10.1371/journal.pcbi.1007924

    """
    path = files("prfmodel.data.stimuli").joinpath("2d_bar_stimulus.npz")

    archive = np.load(str(path))

    return PRFStimulus(
        design=archive["design"],
        grid=archive["grid"],
        dimension_labels=["y", "x"],
    )
