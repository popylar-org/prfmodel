"""Container for contrast sensitivity function stimulus."""

from dataclasses import dataclass
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from .base import Stimulus


class CSFStimulusShapeError(Exception):
    """
    Exception raised when the shapes of the sf and contrast arrays do not match.

    Parameters
    ----------
    sf_shape : tuple of int
        Shape of the sf array.
    contrast_shape : tuple of int
        Shape of the contrast array.

    """

    def __init__(self, sf_shape: tuple[int, ...], contrast_shape: tuple[int, ...]):
        super().__init__(f"Shapes of 'sf' {sf_shape} and 'contrast' {contrast_shape} do not match")


class CSFStimulusDimensionError(Exception):
    """
    Exception raised when sf or contrast is not a one-dimensional array.

    Parameters
    ----------
    arg_name : str
        Name of the argument with wrong dimensions.
    arg_shape : tuple of int
        Shape of the argument.

    """

    def __init__(self, arg_name: str, arg_shape: tuple[int, ...]):
        super().__init__(f"'{arg_name}' must be one-dimensional but has shape {arg_shape}")


@dataclass(frozen=True, eq=False)
class CSFStimulus(Stimulus):
    """
    Container for a contrast sensitivity function stimulus.

    Parameters
    ----------
    sf : numpy.ndarray
        Spatial frequency at each time frame, with shape ``(num_frames,)``.
    contrast : numpy.ndarray
        Contrast at each time frame, with shape ``(num_frames,)``.

    Raises
    ------
    CSFStimulusDimensionError
        If ``sf`` or ``contrast`` is not one-dimensional.
    CSFStimulusShapeError
        If ``sf`` and ``contrast`` do not have the same shape.

    Examples
    --------
    Create a CSF stimulus with four frames:

    >>> import numpy as np
    >>> sf = np.array([1.0, 3.0, 6.0, 12.0])
    >>> contrast = np.array([0.1, 0.2, 0.4, 0.8])
    >>> stimulus = CSFStimulus(sf=sf, contrast=contrast)
    >>> print(stimulus)
    CSFStimulus(sf=array[4], contrast=array[4])

    """

    sf: np.ndarray
    contrast: np.ndarray

    def __post_init__(self):
        self._check_dimensions()
        self._check_shapes()

    def _check_dimensions(self) -> None:
        if self.sf.ndim != 1:
            arg = "sf"
            raise CSFStimulusDimensionError(arg, self.sf.shape)
        if self.contrast.ndim != 1:
            arg = "contrast"
            raise CSFStimulusDimensionError(arg, self.contrast.shape)

    def _check_shapes(self) -> None:
        if self.sf.shape != self.contrast.shape:
            raise CSFStimulusShapeError(self.sf.shape, self.contrast.shape)


def plot_csf_stimulus_design(stimulus: CSFStimulus, **kwargs) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Plot the design of a contrast sensitivity function (CSF) stimulus.

    Plots the stimulus contrast over time with color indicating different spatial frequencies.

    Parameters
    ----------
    stimulus : CSFStimulus
        Stimulus object with contrast and spatial frequencies.
    **kwargs:
        Keyword arguments passed to :func:`~matplotlib.pyplot.subplots`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.

    See Also
    --------
    plot_csf_stimulus_curve : Contrast sensitivity curve on top of the stimulus design.

    """
    sf_levels = np.unique(stimulus.sf)

    fig, ax = plt.subplots(**kwargs)

    for freq in sf_levels:
        is_freq = stimulus.sf == freq
        con = stimulus.contrast[is_freq]
        frames = np.argmax(is_freq) + np.arange(con.shape[0])
        ax.plot(frames, con, label=freq)

    ax.set_xlabel("Time frame")
    ax.set_ylabel("Contrast")

    fig.legend(title="Spatial\nfrequency", bbox_to_anchor=(1.05, 1))

    return fig, ax


def plot_csf_stimulus_curve(
    stimulus: CSFStimulus,
    csf: np.ndarray,
    **kwargs,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Plot a contrast sensitivity curve over a stimulus design.

    Plots spatial frequency over contrast sensitivity (100/contrast) and draws a contrast sentitivity curve on top.

    Parameters
    ----------
    stimulus : CSFStimulus
        Stimulus object with contrast and spatial frequencies.
    csf : numpy.ndarray
        Contrast sensitivity curve. Must match the number of frames in ``stimulus``.
    **kwargs:
        Keyword arguments passed to :func:`~matplotlib.pyplot.subplots`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.

    See Also
    --------
    plot_csf_stimulus_design : Stimulus contrast over time.

    """
    csf = np.atleast_2d(csf)

    if csf.shape[1] != stimulus.contrast.shape[0]:
        msg = f"'csf' must have the same number of frames as 'stimulus' but has only {csf.shape[1]} frames"
        raise ValueError(msg)

    sf_levels = np.unique(stimulus.sf)
    fig, ax = plt.subplots(**kwargs)

    for freq in sf_levels:
        is_freq = stimulus.sf == freq
        con = stimulus.contrast[is_freq]
        ax.scatter(stimulus.sf[is_freq], 100.0 / con, label=freq)

    ax.loglog(stimulus.sf, np.transpose(csf), color="black")

    ax.set_xlabel("Spatial frequency")
    ax.set_ylabel("Contrast sensitivity")

    return fig, ax
