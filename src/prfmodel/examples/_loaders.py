"""Load example datasets."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from nibabel.gifti import GiftiImage
from nilearn.surface import PolyMesh
from nilearn.surface import load_surf_data
from scipy.io import loadmat
from prfmodel.examples._dataset import Dataset
from prfmodel.examples._fetch import FileFetcher
from prfmodel.examples._fetch import get_data_dir
from prfmodel.examples._matlab import read_matlab_tables
from prfmodel.examples._options import validate_options
from prfmodel.examples._registry import _get_spec
from prfmodel.examples._registry import load_checksums
from prfmodel.examples._stimuli import load_1d_prf_lognumerosity_stimulus
from prfmodel.stimuli import PRFStimulus

if TYPE_CHECKING:
    import os
    from prfmodel.examples._options import Options

# The bar apertures were resampled to 100 by 100 pixels spanning 16.6 degrees of visual angle before the
# source study fit its pRF models, so one pixel is 0.166 degrees wide.
_ECOG_STIMULUS_DEGREES = 16.6

# The visual areas of the Wang maximum probability atlas, grouped into the regions that the source study
# reports. Ventral and dorsal quarterfields are merged into one area, as are the two subdivisions of VO, PHC
# and TO, and the six intraparietal areas.
_ECOG_ROI_NAMES = ("V1", "V2", "V3", "hV4", "VO", "PHC", "V3a", "V3b", "LO1", "LO2", "TO", "IPS", "SPL1", "FEF", "none")
_ECOG_ROI_OF_WANG_AREA = {
    "V1v": "V1", "V1d": "V1",
    "V2v": "V2", "V2d": "V2",
    "V3v": "V3", "V3d": "V3",
    "hV4": "hV4",
    "VO1": "VO", "VO2": "VO",
    "PHC1": "PHC", "PHC2": "PHC",
    "V3a": "V3a",
    "V3b": "V3b",
    "LO1": "LO1", "LO2": "LO2",
    "TO1": "TO", "TO2": "TO",
    "IPS0": "IPS", "IPS1": "IPS", "IPS2": "IPS", "IPS3": "IPS", "IPS4": "IPS", "IPS5": "IPS",
    "SPL1": "SPL1",
    "FEF": "FEF",
    "none": "none",
}  # fmt: skip


def load_retbar_visual(fetch: FileFetcher, options: Options) -> Dataset:
    """Load the response of a single subject to a moving bar stimulus."""
    responses = [
        np.asarray(load_surf_data(fetch(f"response_{code}")), dtype=np.float64) for code in options.hemisphere_codes
    ]

    # The raw design needs preprocessing before it can be turned into a stimulus, so we only expose its path
    fetch("design")

    return Dataset(
        name="7t-retbar-visual",
        response=np.concatenate(responses),
        hemisphere=options.hemisphere,
        files=fetch.files,
    )


def load_numerosity_timing(fetch: FileFetcher, options: Options) -> Dataset:
    """Load the response of a single subject to a sequence of visual numerosities."""
    responses = []
    roi_indices = []

    for code in options.hemisphere_codes:
        responses.append(np.asarray(load_surf_data(fetch(f"response_{options.split}_{code}")), dtype=np.float64))
        roi_indices.append(np.asarray(load_surf_data(fetch(f"roi_{code}")), dtype=np.int32))

    # The label table is identical across hemispheres so we read it from the first one
    label_image = GiftiImage.from_filename(fetch(f"roi_{options.hemisphere_codes[0]}"))

    return Dataset(
        name="numerosity-timing",
        response=np.concatenate(responses),
        stimulus=load_1d_prf_lognumerosity_stimulus(),
        roi_index=np.concatenate(roi_indices),
        roi_mapping=label_image.labeltable.get_labels_as_dict(),
        hemisphere=options.hemisphere,
        split=options.split,
        files=fetch.files,
    )


def load_hcp_surface(fetch: FileFetcher, options: Options) -> Dataset:
    """Load a standardized cortical surface mesh and the atlas that belongs to it."""
    mesh = PolyMesh(fetch(f"{options.surface_type}_lh"), fetch(f"{options.surface_type}_rh"))

    with np.load(fetch("atlas")) as archive:
        atlas = {"left": archive["left"], "right": archive["right"]}

    # The whole archive is downloaded anyway, so we expose every surface it holds
    files = {key: fetch(key) for key in fetch.keys}

    return Dataset(
        name="hcp-999999-surface",
        mesh=mesh,
        atlas=atlas,
        files=files,
    )


def _ecog_stimulus(apertures: np.ndarray) -> PRFStimulus:
    """Turn the bar apertures of the ECoG experiment into a stimulus on a grid in degrees."""
    # The apertures are stored as (height, width, num_frames) but a design is (num_frames, height, width)
    design = np.moveaxis(np.asarray(apertures, dtype=np.float64), 2, 0)
    height, width = design.shape[1:]
    pixel_size = _ECOG_STIMULUS_DEGREES / max(height, width)

    # The participants viewed the screen directly rather than through a mirror, so screen pixels and visual
    # field coordinates differ only in the sign of the vertical axis: x grows to the right across the columns
    # and y grows upwards, which is against the row order. A pRF above and left of fixation therefore has a
    # positive 'mu_y' and a negative 'mu_x', as in the source study.
    x = (np.arange(width) - (width - 1) / 2) * pixel_size
    y = ((height - 1) / 2 - np.arange(height)) * pixel_size
    xv, yv = np.meshgrid(x, y)

    # The y coordinate comes first because it varies along the first design axis after time
    return PRFStimulus(design=design, grid=np.stack((yv, xv), axis=-1), dimension_labels=["y", "x"])


def load_visual_ecog_broadband(fetch: FileFetcher, options: Options) -> Dataset:  # noqa: ARG001 (loaders take options)
    """Load the broadband response of a single subject to a moving bar stimulus."""
    path = fetch("response")

    # Naming the variables keeps scipy from reading the object system, which it cannot represent; the tables
    # that live there are read separately below
    contents = loadmat(path, variable_names=["datats", "stimulus"], squeeze_me=True, struct_as_record=False)

    channels = read_matlab_tables(path)["channels"]
    roi = channels["wangarea"].map(_ECOG_ROI_OF_WANG_AREA)
    roi_index = np.asarray(pd.Categorical(roi, categories=_ECOG_ROI_NAMES).codes, dtype=np.int32)

    return Dataset(
        name="visual-ecog-broadband",
        response=np.asarray(contents["datats"], dtype=np.float64),
        stimulus=_ecog_stimulus(contents["stimulus"]),
        roi_index=roi_index,
        roi_mapping=dict(enumerate(_ECOG_ROI_NAMES)),
        units=channels,
        band="broadband",
        files=fetch.files,
    )


def load_dataset(  # noqa: PLR0913
    name: str,
    *,
    dest_dir: str | os.PathLike | None = None,
    hemisphere: str | None = None,
    split: str | None = None,
    surface_type: str | None = None,
    download: bool = True,
) -> Dataset:
    """
    Load an example dataset, downloading it on first use.

    The files of a dataset are cached in a data directory, so they are downloaded only once. Datasets are
    distributed in two ways: As a single archive, which means that the whole archive is downloaded but only the files
    that a dataset needs are extracted and kept; or as individual files of which only the ones that are need are
    downloaded.

    Parameters
    ----------
    name : str
        Name of the dataset. Must be one of :func:`~prfmodel.examples.list_datasets`.
    dest_dir : str or os.PathLike, optional
        Directory in which the dataset files are stored. If `None` (the default), uses the directory from
        :func:`~prfmodel.examples.get_data_dir`.
    hemisphere : str, optional
        Hemisphere(s) to load the response for. Must be either `"both"`, `"left"`, or `"right"`. For
        `"both"`, the data of both hemispheres are concatenated (left is first). If `None` (the default),
        uses the default of the dataset.
    split : str, optional
        Data split to load the response from, for datasets that provide splits. Required for those datasets.
    surface_type : str, optional
        Surface type to load, for datasets that provide surfaces. Must be either `"flat"`, `"inflated"`,
        `"pia"` (or `"pial"`), or `"wm"` (for white matter).
    download : bool, default=True
        Whether files may be downloaded when they are missing. If `False`, missing files raise
        `FileNotFoundError` instead.

    Returns
    -------
    Dataset
        The dataset. Which of its fields are populated depends on the dataset; see
        :func:`~prfmodel.examples.describe_dataset`.

    Raises
    ------
    ValueError
        If `name` is not the name of an available dataset, if an option is not accepted by the dataset, or if
        the value of an option is not valid.
    FileNotFoundError
        If a file is missing and `download` is `False`.
    prfmodel.examples.ChecksumError
        If a downloaded file does not match its expected checksum.

    See Also
    --------
    prfmodel.examples.list_datasets : List the available example datasets.
    prfmodel.examples.describe_dataset : Describe an example dataset without loading it.
    prfmodel.examples.get_data_dir : Resolve the directory in which example datasets are stored.

    Examples
    --------
    Load a surface mesh and the atlas that belongs to it.

    >>> from prfmodel.examples import load_dataset
    >>> surface = load_dataset("hcp-999999-surface", surface_type="flat")  # doctest: +SKIP
    >>> sorted(surface.mesh.parts)  # doctest: +SKIP
    ['left', 'right']
    >>> surface.mesh.parts["left"].n_vertices  # doctest: +SKIP
    59292

    Load a response together with the raw design of the stimulus that produced it.

    >>> dataset = load_dataset("7t-retbar-visual", hemisphere="both")  # doctest: +SKIP
    >>> dataset.response.shape  # doctest: +SKIP
    (118584, 120)

    Load two splits of the same response, for cross-validation.

    >>> odd = load_dataset("numerosity-timing", hemisphere="left", split="odd")  # doctest: +SKIP
    >>> even = load_dataset("numerosity-timing", hemisphere="left", split="even")  # doctest: +SKIP
    >>> odd.response.shape  # doctest: +SKIP
    (5436, 176)
    >>> odd.response.shape == even.response.shape  # doctest: +SKIP
    True

    Load an intracranial response together with the channels table that describes it.

    >>> broadband = load_dataset("visual-ecog-broadband")  # doctest: +SKIP
    >>> broadband.response.shape  # doctest: +SKIP
    (136, 224)
    >>> broadband.units.loc[0, ["name", "group", "wangarea"]].tolist()  # doctest: +SKIP
    ['GA33', 'grid', 'none']

    """
    spec = _get_spec(name)
    options = validate_options(spec, hemisphere=hemisphere, split=split, surface_type=surface_type)

    fetcher = FileFetcher(
        spec,
        data_dir=get_data_dir(dest_dir),
        checksums=load_checksums(),
        download=download,
    )

    return spec.loader(fetcher, options)
