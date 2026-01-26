"""Example stimuli and data."""

import io
import os
import urllib
import zipfile
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
import numpy as np
from nilearn.surface import load_surf_data
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from prfmodel.stimulus import Stimulus


def load_2d_bar_stimulus() -> Stimulus:
    """
    Load a two-dimensional bar stimulus.

    Loads an example stimulus that moves in eight directions (vertical, horizontal, diagonal) through a two-dimensional
    space.

    Returns
    -------
    Stimulus
        A stimulus object with a `design` with shape (200, 101, 101) and a `grid` with shape (101, 101, 2).

    Notes
    -----
    The stimulus was created with the validation framework developed by Lerma-Usabiaga et al (2020)[1]_.
    It has 200 time frames, with 101 pixels in the x- and y-dimension, spanning a visual field of 20 degrees
    vertically and horizontally.

    See Also
    --------
    prfmodel.stimulus.animate_2d_stimulus : Create an animation for a two-dimensional stimulus.

    References
    ----------
    .. [1] Lerma-Usabiaga, G., Benson, N., Winawer, J., & Wandell, B. A. (2020). A validation framework for
        neuroimaging software: The case of population receptive fields. *PLOS Computational Biology, 16*(6),
        e1007924. https://doi.org/10.1371/journal.pcbi.1007924

    """
    path = files("prfmodel.data.stimuli").joinpath("2d_bar_stimulus.npz")

    archive = np.load(str(path))

    return Stimulus(
        design=archive["design"],
        grid=archive["grid"],
        dimension_labels=["y", "x"],
    )


def download_surface(subject: str = "hcp_999999") -> None:
    """
    Download a standardized cortical surface and store it in the local pycortex database.

    Currently only the surface for the subject identifier `"hcp_999999"` can be downloaded. This surface is based on
    the one used in the Human Connectome Project (see this `link <https://doi.org/10.6084/m9.figshare.13372958>`_).

    Parameters
    ----------
    subject : str, default="hcp_999999"
        Identifier of the subject for which to download and store the surface.

    """
    import cortex as cx  # noqa: PLC0415 (import should be at top level)

    if subject == "hcp_999999":
        if subject not in cx.db.subjects:
            with urllib.request.urlopen("https://ndownloader.figshare.com/files/25768841") as response:
                zip_bytes = response.read()
                with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as archive:
                    for file in archive.namelist():
                        if file.startswith(subject):
                            archive.extract(file, cx.database.default_filestore)

            cx.db.reload_subjects()
    else:
        msg = f"Surface for subject '{subject}' not found"
        raise ValueError(msg)


def _load_single_subject_fmri_stimulus(dest_path: str | os.PathLike) -> Stimulus:
    design_path = Path(dest_path) / "vis_design.mat"
    design = np.transpose(loadmat(design_path)["stim"])

    pixel_offset = int((design.shape[1] - design.shape[2]) / 2)
    new_design = np.zeros((design.shape[0], design.shape[1], design.shape[1]))

    for frame in range(design.shape[0]):
        square_screen = np.zeros_like(new_design[frame])
        square_screen[:, pixel_offset : (pixel_offset + design.shape[2])] = (design[frame] != 0).astype(float)
        new_design[frame, :, :] = np.transpose(gaussian_filter(square_screen, 10))

    new_design = new_design[:, ::5, ::5]

    num_grid_points = 100
    x_min, y_min, x_max, y_max = -10.0, -10.0, 10.0, 10.0

    coordinates = np.stack(
        np.meshgrid(
            np.linspace(x_min, x_max, num_grid_points),
            np.linspace(y_min, y_max, num_grid_points),
        ),
        axis=-1,
    )

    return Stimulus(new_design, coordinates, ["y", "x"])


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
    unit: str = "psc",
) -> tuple[np.ndarray, Stimulus]:
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
    unit : str, default="psc"
        Unit to which the BOLD response data should be converted. Must be either 'raw', 'psc' (percent signal change),
        or 'z_score' (subtracts mean from signal and divides by standard deviation).

    Returns
    -------
    response : numpy.ndarray
        The BOLD response in the requested unit.
    stimulus : prfmodel.stimulus.Stimulus
        The experimental stimulus.

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

    response_combined = np.concatenate(response_hemishperes)

    match unit:
        case "psc":
            response_combined = ((response_combined.T / response_combined.mean(axis=1)).T - 1.0) * 100.0
        case "z_score":
            response_combined = (
                (response_combined.T - response_combined.mean(axis=1)) / response_combined.std(axis=1)
            ).T
        case "raw":
            pass
        case other:
            msg = f"Argument 'unit' must be 'psc', 'z_score', or 'raw' but was '{other}'"
            raise ValueError(msg)

    stimulus = _load_single_subject_fmri_stimulus(dest_dir)

    return response_combined, stimulus


def load_brain_atlas(dest_path: str | os.PathLike, hemisphere: str = "both") -> np.ndarray:
    """Load a standardized brain atlas.

    Loads a brain atlas based on multimodal parcellation from the Human Connectome Project
    (see Glasser et al., 2016)[1]_. The atlas contains indices for different regions of interest. It is subsampled to
    match the cortical surface of subject `'hcp_999999'` that can be downloaded with
    :func:`prfmodel.examples.download_surface`.

    Parameters
    ----------
    dest_dir : str or os.PathLike
        Directory where the data should be stored. If the directory does not contain all required files the data is
        downloaded from Fig Share.
    hemisphere : str, default="both"
        Hemisphere(s) for which the atlas should be loaded. Must be either 'left', 'right', or 'both'.
        For 'both', the data of both hemispheres are concatenated (left is first).

    Returns
    -------
    numpy.ndarray
        The region of interest index of each vertex in the subsampled surface.

    Notes
    -----
    For details, see this `link <https://figshare.com/articles/dataset/fMRI_Teaching_Materials/14096209>`_.

    References
    ----------
    .. [1] Glasser, M. F., Coalson, T. S., Robinson, E. C., Hacker, C. D., Harwell, J., Yacoub, E., Ugurbil, K.,
    Andersson, J., Beckmann, C. F., Jenkinson, M., Smith, S. M., & Van Essen, D. C. (2016). A multi-modal parcellation
    of human cerebral cortex. *Nature, 536*(7615), 171-178. https://doi.org/10.1038/nature18933

    """
    _hemisphere = _get_hemisphere(hemisphere)

    file_url = "https://ndownloader.figshare.com/files/26587088"

    _download_archive(file_url, dest_path)

    atlas_hemishperes = [
        np.asarray(
            load_surf_data(
                Path(dest_path)
                / "atlas"
                / f"Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.59k_fs_LR.dlabel.{hemi}.gii",
            ),
            dtype=np.float64,
        )
        for hemi in _hemisphere
    ]

    atlas_combined = np.concatenate(atlas_hemishperes)

    if hemisphere == "both":
        return np.mod(atlas_combined, 180)

    return atlas_combined
