"""Bundled example stimuli."""

from importlib.resources import files
import numpy as np
from prfmodel.stimuli import PRFStimulus


def load_2d_prf_bar_stimulus(return_test: bool = False) -> PRFStimulus | tuple[PRFStimulus, PRFStimulus]:
    """
    Load a two-dimensional population receptive field bar stimulus.

    Loads an example stimulus in which a bar moves in four directions (top-bottom, left-right, bottom-top, right-left)
    through a two-dimensional space.

    Parameters
    ----------
    return_test : bool, default=True
        Whether to return a test stimulus. If ``False``, returns a single stimulus with the first half of the design
        matrix. If ``True``, returns a train and test stimulus with the first and second half of the design matrix
        (split in the time axis).

    Returns
    -------
    PRFStimulus
        If ``return_test=False`` (the default), a single stimulus object with a ``design`` with shape
        ``(170, 128, 128)`` and a ``grid`` with shape ``(128, 128, 2)``. If ``return_test=True``, two stimulus objects
        with a ``design`` with shape ``(170, 128, 128)`` and a ``grid`` with shape ``(128, 128, 2)``.

    Notes
    -----
    The full stimulus has 340 time frames, with 128 pixels in the x- and y-dimension, spanning a visual field of
    approximately -4 to 4 degrees of visual angle vertically and horizontally. The stimulus coordinate grid has been
    transformed from a rectangular into a quadratic shape. The moving bar patterns is mirrored in the time axis
    which means that it can be split in two almost identical halfs.

    The ``design`` is stored in screen pixel order while the ``grid`` holds visual field coordinates, which are
    related by a horizontal flip because the participant viewed the screen through a mirror. The horizontal
    coordinate therefore decreases across the design columns, so a bar in the leftmost columns falls in the right
    visual hemifield (positive ``mu_x``). The vertical coordinate increases down the rows.
    :meth:`~prfmodel.stimuli.PRFStimulus.create_2d_bar_stimulus` uses the same convention, so ``mu_x`` estimates are
    directly comparable between the two.

    See Also
    --------
    prfmodel.plotting.animate_2d_prf_stimulus : Create an animation for a two-dimensional stimulus.
    prfmodel.stimuli.PRFStimulus.create_2d_bar_stimulus : Create a two-dimensional bar stimulus.

    Examples
    --------
    Load a single stimulus with the first half of the design matrix.

    >>> from prfmodel.examples import load_2d_prf_bar_stimulus
    >>> stimulus = load_2d_prf_bar_stimulus()
    >>> print(stimulus)
    PRFStimulus(design=array[170, 128, 128], grid=array[128, 128, 2], dimension_labels=['y', 'x'])

    Load a train and test stimulus with the first and second half of the design matrix.

    >>> stimulus_train, stimulus_test = load_2d_prf_bar_stimulus(return_test=True)
    >>> print(stimulus_train)
    PRFStimulus(design=array[170, 128, 128], grid=array[128, 128, 2], dimension_labels=['y', 'x'])

    """
    path = files("prfmodel.data.stimuli").joinpath("2d_bar_stimulus.npz")

    archive = np.load(str(path))

    design = archive["design"]
    grid = archive["grid"]
    dimension_labels = ["y", "x"]

    num_split = design.shape[0] // 2
    design_train = design[:num_split]
    design_test = design[num_split:]

    stimulus_train = PRFStimulus(
        design=design_train,
        grid=grid,
        dimension_labels=dimension_labels,
    )

    if not return_test:
        return stimulus_train

    stimulus_test = PRFStimulus(
        design=design_test,
        grid=np.copy(grid),
        dimension_labels=dimension_labels.copy(),
    )

    return stimulus_train, stimulus_test


def load_1d_prf_lognumerosity_stimulus() -> PRFStimulus:
    """Load a one-dimensional population receptive field log numerosity stimulus.

    Loads an example stimulus that includes a one-dimensional sequence of log integers (numerosities) from a numerosity
    experiment [1]_.

    Returns
    -------
    PRFStimulus
        A stimulus object with a ``design`` with shape ``(182, 8)`` that one-hot encodes which numerosity is shown at
        each time frame and a ``grid`` with shape ``(8, 1)`` that contains the unique log integers.

    Notes
    -----
    The stimulus contains four cycles of the same sequence of unique numerosities 1, 2, 3, 4, 5, 6, 7, and 20.
    Each numerosity is presented for two consecutive frames. The numerosities 1 to 7 first ascend in order followed by
    numerosity 20 presented for eight frames and then descend in order followed by 20 for eight frames.
    The first six frames also contain numerosity 20 and serve as a prescan interval in the experiment.

    References
    ----------
    .. [1] Hendrikx, E., Paul, J. M., van Ackooij, M., van der Stoep, N., & Harvey, B. M. (2024). Cortical quantity
        representations of visual numerosity and timing overlap increasingly into superior cortices but remain
        distinct. *NeuroImage*, 286, 120515. https://doi.org/10.1016/j.neuroimage.2024.120515

    Examples
    --------
    >>> stimulus = load_1d_prf_lognumerosity_stimulus()
    >>> print(stimulus)
    PRFStimulus(design=array[182, 8], grid=array[8, 1], dimension_labels=['log_numerosity'])

    """
    path = files("prfmodel.data.stimuli").joinpath("1d_lognumerosity_stimulus.npz")

    archive = np.load(str(path))

    design = archive["design"]
    grid = archive["grid"]

    dimension_labels = ["log_numerosity"]

    return PRFStimulus(
        design=design,
        grid=grid,
        dimension_labels=dimension_labels,
    )
