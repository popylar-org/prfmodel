"""Example stimuli and data."""

import io
import os
import urllib.request
import zipfile
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
import numpy as np
from nilearn.surface import PolyMesh
from nilearn.surface import load_surf_data
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

    See Also
    --------
    prfmodel.stimuli.prf.animate_2d_prf_stimulus : Create an animation for a two-dimensional stimulus.

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
    dimesion_labels = ["y", "x"]

    num_split = design.shape[0] // 2
    design_train = design[:num_split]
    design_test = design[num_split:]

    stimulus_train = PRFStimulus(
        design=design_train,
        grid=grid,
        dimension_labels=dimesion_labels,
    )

    if not return_test:
        return stimulus_train

    stimulus_test = PRFStimulus(
        design=design_test,
        grid=np.copy(grid),
        dimension_labels=dimesion_labels.copy(),
    )

    return stimulus_train, stimulus_test


def load_surface_mesh(
    dest_dir: str | os.PathLike,
    surface_type: str = "flat",
) -> PolyMesh:
    """
    Download and load a standardized cortical surface mesh.

    Downloads a surface mesh based on the Human Connectome Project [1]_ from Fig Share and extracts the surface
    meshes for the left and right hemispheres.

    Parameters
    ----------
    dest_dir : str or os.PathLike
        Directory where the surface files should be stored.
    type : str, default="flat"
        The surface type. Must be either `"flat"` (the default), `"inflated"`, `"pial"`, or `"wm"`
        (for white matter).

    Returns
    -------
    nilearn.surface.PolyMesh
        The surface mesh.

    References
    ----------
    .. [1] Van Essen, D. C., Ugurbil, K., Auerbach, E., Barch, D., Behrens, T. E. J., Bucholz, R., Chang, A., Chen, L.,
        Corbetta, M., Curtiss, S. W., Della Penna, S., Feinberg, D., Glasser, M. F., Harel, N., Heath, A. C.,
        Larson-Prior, L., Marcus, D., Michalareas, G., Moeller, S., … WU-Minn HCP Consortium. (2012).
        The Human Connectome Project: A data acquisition perspective. *NeuroImage*, 62(4), 2222-2231.
        https://doi.org/10.1016/j.neuroimage.2012.02.018

    """
    surface_type = "pia" if surface_type == "pial" else surface_type

    valid_types = ("flat", "inflated", "pia", "wm")

    if surface_type not in valid_types:
        msg = f"Surface type must be one of {valid_types}"
        raise ValueError(msg)

    subject = "hcp_999999"

    dest_path = Path(dest_dir)
    surface_dir = dest_path / subject / "surfaces"

    if not surface_dir.exists():
        dest_path.mkdir(parents=True, exist_ok=True)
        url = "https://ndownloader.figshare.com/files/25768841"

        if not url.startswith(("http:", "https:")):
            msg = "File URL must start with http: or https:"
            raise ValueError(msg)

        with urllib.request.urlopen(url) as response:  # noqa: S310 (Audit URL open for permitted schemes)
            zip_bytes = response.read()
            with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as archive:
                for file in archive.namelist():
                    if file.startswith(subject):
                        archive.extract(file, dest_path)

    return PolyMesh(
        surface_dir / f"{surface_type}_lh.gii",
        surface_dir / f"{surface_type}_rh.gii",
    )


def _get_hemisphere(hemisphere: str) -> Sequence[str]:
    hemisphere_dict = {
        "both": ("L", "R"),
        "left": ("L"),
        "right": ("R"),
    }

    if hemisphere not in hemisphere_dict:
        msg = f"Argument 'hemisphere' must be 'left', 'right', or 'both' but was '{hemisphere}'"
        raise ValueError(msg)

    return hemisphere_dict[hemisphere]


def _download_archive(file_url: str, dest_path: str | os.PathLike) -> None:
    Path(dest_path).mkdir(parents=True, exist_ok=True)

    if not file_url.startswith(("http:", "https:")):
        msg = "File URL must start with http: or https:"
        raise ValueError(msg)

    # We need to ignore ruff's flag in the next line; we already audited the URL but it still flags and issue
    with urllib.request.urlopen(file_url) as response:  # noqa: S310 (Audit URL open for permitted schemes)
        zip_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as archive:
            archive.extractall(dest_path)


def load_single_subject_fmri_data(
    dest_dir: str | os.PathLike,
    hemisphere: str = "both",
) -> np.ndarray:
    """
    Load example functional magnetic resonance imaging (fMRI) data for a single subject.

    Loads the blood oxygenation level-dependent (BOLD) response and the visual stimulus used in the experiment.
    Downloads the data from Fig Share if it does not yet exist.

    Parameters
    ----------
    dest_dir : str or os.PathLike
        Directory where the data should be stored. If the directory does not contain all required files the data is
        downloaded from Fig Share.
    hemisphere : str, default="both"
        Hemisphere(s) for which the BOLD response data should be loaded. Must be either 'left', 'right', or 'both'.
        For 'both', the data of both hemispheres are concatenated (left is first).

    Returns
    -------
    response : numpy.ndarray
        The BOLD response in the requested unit.

    Notes
    -----
    The data have been obtained by averaging 43 runs across 4 sessions from a single subject in a 7T fMRI scanner
    using a repitition time of 1.5 (see this
    `link <https://figshare.com/articles/dataset/fMRI_Teaching_Materials/14096209>`_ for further details).

    """
    _hemisphere = _get_hemisphere(hemisphere)

    file_url = "https://ndownloader.figshare.com/files/26577941"

    _download_archive(file_url, dest_dir)

    response_hemishperes = [
        np.asarray(
            load_surf_data(Path(dest_dir) / f"sub-02_task-prf_space-59k_hemi-{hemi}_run-median_desc-bold.func.gii"),
            dtype=np.float64,
        )
        for hemi in _hemisphere
    ]

    return np.concatenate(response_hemishperes)
