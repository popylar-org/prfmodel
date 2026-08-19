"""Container for contrast sensitivity function stimulus."""

from dataclasses import dataclass
import numpy as np
from keras import ops
from prfmodel.typing import Tensor
from .base import Stimulus
from .base import StimulusTensors


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
class CSFStimulusTensors(StimulusTensors):
    """Tensor-holding counterpart of a :class:`~prfmodel.stimuli.CSFStimulus`.

    Holds the contrast sensitivity function stimulus arrays as backend tensors. Should be created with
    :meth:`CSFStimulus.to_tensors`.

    Parameters
    ----------
    sf : :data:`prfmodel.typing.Tensor`
        The :attr:`CSFStimulus.sf` array as a tensor.
    contrast : :data:`prfmodel.typing.Tensor`
        The :attr:`CSFStimulus.contrast` array as a tensor.

    """

    sf: Tensor
    contrast: Tensor


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

    def to_tensors(self, dtype: str | None = None) -> CSFStimulusTensors:
        """Convert the stimulus arrays into backend tensors.

        Parameters
        ----------
        dtype : str, optional
            The dtype to convert the stimulus arrays to. If `None` (the default), uses the dtype from
            :func:`prfmodel.utils.get_dtype`.

        Returns
        -------
        CSFStimulusTensors
            The stimulus arrays as tensors.

        Examples
        --------
        >>> import numpy as np
        >>> stimulus = CSFStimulus(sf=np.array([1.0, 3.0]), contrast=np.array([0.1, 0.2]))
        >>> tensors = stimulus.to_tensors("float32")
        >>> print(tuple(tensors.sf.shape))
        (2,)

        """
        return CSFStimulusTensors(
            sf=ops.convert_to_tensor(self.sf, dtype=dtype),
            contrast=ops.convert_to_tensor(self.contrast, dtype=dtype),
        )
