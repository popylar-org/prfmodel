"""Container for a loaded example dataset."""

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import numpy as np
from nilearn.surface import PolyMesh
from prfmodel.stimuli import Stimulus


@dataclass(frozen=True, eq=False)
class Dataset:
    """
    An example dataset and the files it was loaded from.

    Datasets differ in what they carry, and fields that a dataset does not provide are `None`. Print the
    object to see which fields are populated, or call :func:`~prfmodel.examples.describe_dataset` for a
    description of a dataset before loading it.

    Parameters
    ----------
    name : str
        Name of the dataset, as passed to :func:`~prfmodel.examples.load_dataset`.
    response : numpy.ndarray, optional
        Neural timecourses with shape `(num_units, num_frames)`, where units are surface vertices or
        region of interest gray nodes depending on the dataset. When both hemispheres are requested, the rows
        of the left hemisphere come first.
    stimulus : prfmodel.stimuli.Stimulus, optional
        The stimulus that belongs to the response, when the dataset provides a ready-made one. Datasets whose
        stimulus requires preprocessing leave this `None` and expose the raw design through `files` instead.
    roi_index : numpy.ndarray, optional
        Integer region of interest index for each row of `response`, with shape `(num_units,)`.
    roi_mapping : dict of int to str, optional
        Mapping from the values in `roi_index` to region of interest labels.
    mesh : nilearn.surface.PolyMesh, optional
        The cortical surface mesh that the response can be projected onto.
    atlas : dict of str to numpy.ndarray, optional
        Per-hemisphere parcellation labels, keyed by `"left"` and `"right"`.
    hemisphere : str, optional
        The hemisphere selection that the dataset was loaded with.
    split : str, optional
        The data split that `response` was loaded from.
    files : collections.abc.Mapping of str to pathlib.Path
        The files that were fetched, keyed by a short logical name such as `"design"` or `"wm_lh"`.

    Notes
    -----
    The object is immutable but its arrays are not copied, so mutating them in place mutates the dataset.

    See Also
    --------
    prfmodel.examples.load_dataset : Load an example dataset.
    prfmodel.examples.describe_dataset : Describe an example dataset without loading it.

    """

    name: str
    response: np.ndarray | None = None
    stimulus: Stimulus | None = None
    roi_index: np.ndarray | None = None
    roi_mapping: dict[int, str] | None = None
    mesh: PolyMesh | None = None
    atlas: dict[str, np.ndarray] | None = None
    hemisphere: str | None = None
    split: str | None = None
    files: Mapping[str, Path] = field(default_factory=dict)

    # Contains numpy arrays as attributes which are not hashable
    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        """Create a round-trippable string representation of the dataset object."""
        arg_list = []

        for key, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                arg_list.append(f"{key}={np.array_repr(val)}")
            else:
                arg_list.append(f"{key}={val!r}")

        return f"{self.__class__.__name__}({', '.join(arg_list)})"

    def __str__(self) -> str:
        """Create a human-readable string representation of the dataset object.

        Fields that the dataset does not provide are left out.
        """
        str_list = []

        for key, val in self.__dict__.items():
            if val is None:
                continue
            if isinstance(val, np.ndarray):
                arr_shape = ", ".join([str(s) for s in val.shape])
                str_list.append(f"{key}=array[{arr_shape}]")
            elif key == "files":
                str_list.append(f"{key}=[{', '.join(val)}]")
            elif isinstance(val, dict):
                str_list.append(f"{key}=dict[{len(val)}]")
            else:
                str_list.append(f"{key}={val}")

        return f"{self.__class__.__name__}({', '.join(str_list)})"
