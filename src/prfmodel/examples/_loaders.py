"""Load example datasets."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from nibabel.freesurfer import read_label
from nibabel.gifti import GiftiImage
from nibabel.nifti1 import Nifti1Image
from nilearn.surface import PolyMesh
from nilearn.surface import load_surf_data
from scipy import sparse as sp
from scipy.io import loadmat
from prfmodel.examples._dataset import Dataset
from prfmodel.examples._fetch import FileFetcher
from prfmodel.examples._fetch import get_data_dir
from prfmodel.examples._matlab import read_matlab_tables
from prfmodel.examples._options import validate_options
from prfmodel.examples._registry import _get_spec
from prfmodel.examples._registry import load_checksums
from prfmodel.examples._stimuli import _BAR_STIMULUS_FRAME_DURATION
from prfmodel.examples._stimuli import _NCSF_FRAME_DURATION
from prfmodel.examples._stimuli import _load_2d_bar_design_and_grid
from prfmodel.examples._stimuli import _load_chn_aperture_stimulus
from prfmodel.examples._stimuli import _load_ncsf_csf_stimulus
from prfmodel.examples._stimuli import load_1d_prf_lognumerosity_stimulus
from prfmodel.stimuli import PRFStimulus
from prfmodel.stimuli import Stimulus

if TYPE_CHECKING:
    import os
    from prfmodel.examples._options import Options

# The bar apertures were resampled to 100 by 100 pixels spanning 16.6 degrees of visual angle before the
# source study fit its pRF models, so one pixel is 0.166 degrees wide.
_ECOG_STIMULUS_DEGREES = 16.6

# Each bar position was shown for 500 ms followed by a 350 ms blank, so one bar step and one response frame span
# 850 ms.
_ECOG_FRAME_DURATION = 0.85

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

# Hemisphere code -> (id in vertex_hemisphere_id, file prefix, PolyMesh part name)
_NUMEROSITY_HEMISPHERES = {"L": (0, "lh", "left"), "R": (1, "rh", "right")}


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


_NCSF_TEMPLATE_MAPS = ("angle", "eccen", "sigma")
_NCSF_HEMISPHERE_NAMES = {"L": "left", "R": "right"}


def load_ncsf_visual(fetch: FileFetcher, options: Options) -> Dataset:
    """Load the surface response of a single subject to a contrast sensitivity or a moving bar stimulus."""
    # Only the CSF task was split, and checking before fetching avoids a download that would be refused
    if options.task == "prf" and options.split != "all":
        msg = f"Task 'prf' of dataset '7t-ncsf-visual' has only the split 'all' but '{options.split}' was requested"
        raise ValueError(msg)

    responses = []
    roi_indices = []
    units = []

    for code in options.hemisphere_codes:
        response_path = fetch(f"response_{options.task}_{options.split}_{code}")
        responses.append(np.asarray(load_surf_data(response_path), dtype=np.float64))
        roi_indices.append(np.asarray(load_surf_data(fetch(f"roi_{code}")), dtype=np.int32))
        # The template polar angle is unsigned in both hemispheres, so the hemisphere tells the visual hemifield
        maps = {
            f"benson14_{key}": np.asarray(load_surf_data(fetch(f"benson14_{key}_{code}")), dtype=np.float64)
            for key in _NCSF_TEMPLATE_MAPS
        }
        units.append(pd.DataFrame({"hemisphere": _NCSF_HEMISPHERE_NAMES[code], **maps}))

    # The label table is identical across hemispheres so we read it from the first one
    label_image = GiftiImage.from_filename(fetch(f"roi_{options.hemisphere_codes[0]}"))

    stimulus: Stimulus

    if options.task == "csf":
        stimulus = _load_ncsf_csf_stimulus()
    else:
        with np.load(fetch("stimulus_prf")) as archive:
            stimulus = PRFStimulus(
                design=np.asarray(archive["design"], dtype=np.float64),
                grid=np.asarray(archive["grid"], dtype=np.float64),
                frame_duration=_NCSF_FRAME_DURATION,
                dimension_labels=["y", "x"],
            )

    return Dataset(
        name="7t-ncsf-visual",
        response=np.concatenate(responses),
        stimulus=stimulus,
        roi_index=np.concatenate(roi_indices),
        roi_mapping=label_image.labeltable.get_labels_as_dict(),
        units=pd.concat(units, ignore_index=True),
        hemisphere=options.hemisphere,
        split=options.split,
        task=options.task,
        files=fetch.files,
    )


def load_aot_visual(fetch: FileFetcher, options: Options) -> Dataset:
    """Load the gray matter voxel response of a single subject to a moving bar stimulus and its cortical surface."""
    mask = Nifti1Image.from_filename(fetch("mask"))
    is_in_mask = np.asarray(mask.dataobj).astype(bool)

    # Boolean indexing selects the voxels in C order and yields shape (num_voxels, num_frames)
    bold = np.asarray(Nifti1Image.from_filename(fetch("response")).dataobj)
    response = np.asarray(bold[is_in_mask], dtype=np.float64)

    mesh = PolyMesh(fetch(f"{options.surface_type}_lh"), fetch(f"{options.surface_type}_rh"))

    # Projecting the volume onto the surface requires both the pial and the white matter surface
    for key in ("pia_lh", "pia_rh", "wm_lh", "wm_rh"):
        fetch(key)

    # The response covers the whole design so we do not split it like load_2d_prf_bar_stimulus does
    design, grid = _load_2d_bar_design_and_grid()

    return Dataset(
        name="7t-aot-visual",
        response=response,
        stimulus=PRFStimulus(
            design=design,
            grid=grid,
            frame_duration=_BAR_STIMULUS_FRAME_DURATION,
            dimension_labels=["y", "x"],
        ),
        mask=mask,
        mesh=mesh,
        files=fetch.files,
    )


_CHN_ROIS = ("v1", "v2", "v3")
_HEMISPHERE_PARTS = {"L": "left", "R": "right"}


def load_chn_visual(fetch: FileFetcher, options: Options) -> Dataset:
    """Load the native surface response of a single subject to a fixed or logarithmic moving bar stimulus."""
    responses = []
    roi_indices = []
    mesh_parts = {}

    for code in options.hemisphere_codes:
        # The file holds a single (num_vertices, num_frames) array, which nilearn.surface.load_surf_data would transpose
        image = GiftiImage.from_filename(fetch(f"response_{options.task}_{options.split}_{code}"))
        responses.append(np.asarray(image.darrays[0].data, dtype=np.float64))

        roi_index = np.zeros(responses[-1].shape[0], dtype=np.int32)

        for value, roi in enumerate(_CHN_ROIS, start=1):
            roi_index[read_label(fetch(f"{roi}_{code}"))] = value

        roi_indices.append(roi_index)
        mesh_parts[_HEMISPHERE_PARTS[code]] = fetch(f"inflated_{code}")

    response = np.concatenate(responses)

    # Runs differ in their sequence of bar positions but sessions repeat them, so the aperture depends on the run only
    run = options.split.split("_")[1]  # type: ignore[union-attr]
    stimulus = _load_chn_aperture_stimulus(fetch(f"aperture_{options.task}_{run}"), num_frames=response.shape[1])

    return Dataset(
        name="3t-chn-visual",
        response=response,
        stimulus=stimulus,
        roi_index=np.concatenate(roi_indices),
        roi_mapping={0: "other", **{value: roi.upper() for value, roi in enumerate(_CHN_ROIS, start=1)}},
        mesh=PolyMesh(**mesh_parts),
        hemisphere=options.hemisphere,
        split=options.split,
        task=options.task,
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
    return PRFStimulus(
        design=design,
        grid=np.stack((yv, xv), axis=-1),
        frame_duration=_ECOG_FRAME_DURATION,
        dimension_labels=["y", "x"],
    )


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
        files=fetch.files,
    )


def _voxels_to_vertices(
    response: np.ndarray,
    valid: np.ndarray,
    offsets: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    """Convert voxel timecourses to percent-signal-change (PSC).

    Average them over the voxels assigned to each vertex.
    Vertices without any valid voxel get NaN.
    """
    mean = response.mean(axis=1)
    keep = valid & np.isfinite(response).all(axis=1) & (mean != 0)

    psc = np.zeros_like(response)
    psc[keep] = (response[keep] / mean[keep, None] - 1.0) * 100.0

    num_vertices = len(offsets) - 1
    vertex_of_entry = np.repeat(np.arange(num_vertices), np.diff(offsets))
    is_kept = keep[rows]

    # Sparse assignment matrix: entry (v, r) is 1 if voxel r is assigned to vertex v and is valid
    assignment = sp.csr_matrix(
        (np.ones(is_kept.sum()), (vertex_of_entry[is_kept], rows[is_kept])),
        shape=(num_vertices, len(keep)),
    )
    count = np.asarray(assignment.sum(axis=1)).ravel()

    vertex_response = np.full((num_vertices, response.shape[1]), np.nan)
    has_voxels = count > 0
    vertex_response[has_voxels] = (assignment @ psc)[has_voxels] / count[has_voxels, None]

    return vertex_response


def load_numerosity_four_conditions(fetch: FileFetcher, options: Options) -> Dataset:
    """Load the fMRI response of a single subject to numerosity stimuli in four stimulus conditions (surface data).

    The voxel responses of the requested task are assigned to the vertices of the cortical surface:

    1. Exclude the voxels that the dataset marks as invalid because they have zeros at some time frames (they fell
        outside the scanned volume).
    2. Convert the remaining voxels to percent signal change (PSC) relative to their mean over time.
    3. Take the mean PSC of the voxels assigned to each vertex (`surface_mapping_method2`). Vertices without any
        remaining voxel are NaN.

    The requested hemispheres are returned with the left hemisphere first. The FreeSurfer white surfaces share
    this vertex order and are returned as the mesh. The table in `units` holds the hemisphere of each vertex and
    its geodesic distance in mm within its ROI (`roi_distance_mm`, NaN for vertices outside the ROIs).
    """
    if options.task is None:
        msg = "Dataset 'numerosity-four-conditions' requires a task"
        raise ValueError(msg)

    task = options.task
    task_label = task.capitalize()  # Keys inside validity.npz, e.g. "Circ_valid"

    # These files cover the whole brain (voxels), so they are fetched whichever hemisphere is requested
    with np.load(fetch(f"response_{task}")) as archive:
        response = np.asarray(archive["response"], dtype=np.float64)

    with np.load(fetch("validity")) as archive:
        valid = np.asarray(archive[f"{task_label}_valid"], dtype=bool)

    with np.load(fetch("surface_mapping_method2")) as archive:
        offsets = np.asarray(archive["method2_vertex_voxel_offsets"], dtype=np.int64)
        rows = np.asarray(archive["method2_vertex_voxel_rows"], dtype=np.int64)

    with np.load(fetch("surface_metadata")) as archive:
        vertex_roi_index = np.asarray(archive["vertex_roi_index"], dtype=np.int32)
        vertex_hemisphere_id = np.asarray(archive["vertex_hemisphere_id"], dtype=np.int8)

    with np.load(fetch("label_names")) as archive:
        roi_ids = np.asarray(archive["roi_ids"], dtype=np.int32)
        roi_names = np.asarray(archive["roi_names"])

    # Geodesic distance of each vertex within its ROI (NaN outside); the ROIs do not overlap
    roi_distance = np.full(len(vertex_roi_index), np.nan)

    with np.load(fetch("roi_data")) as archive:
        for key in archive.files:
            if key.endswith("_distances_mm"):
                distances = np.asarray(archive[key], dtype=np.float64)
                is_in_roi = np.isfinite(distances)

                if np.isfinite(roi_distance[is_in_roi]).any():
                    msg = f"The ROI distances in '{key}' overlap with the distances of another ROI"
                    raise ValueError(msg)

                roi_distance[is_in_roi] = distances[is_in_roi]

    # Project all voxels first, then select the hemisphere (the mapping refers to all voxels)
    vertex_response = _voxels_to_vertices(response, valid, offsets, rows)

    if len(vertex_response) != len(vertex_roi_index):
        msg = "The number of vertices in the voxel-to-vertex mapping and the surface metadata differ"
        raise ValueError(msg)

    # Only the surface files of the requested hemispheres are fetched; left comes first
    order_parts = []
    mesh_files = {}

    for code in options.hemisphere_codes:
        hemi_id, prefix, part = _NUMEROSITY_HEMISPHERES[code]
        order_parts.append(np.flatnonzero(vertex_hemisphere_id == hemi_id))
        mesh_files[part] = fetch(f"{prefix}_white")

        # Expose the flat surfaces and their vertex indices on the full surface (kept in `files`)
        fetch(f"{prefix}_flat")
        fetch(f"{prefix}_vertex_indices")

    order = np.concatenate(order_parts)
    mesh = PolyMesh(**mesh_files)

    # The response rows must line up with the mesh vertices of each hemisphere
    for indices, part in zip(order_parts, mesh_files, strict=True):
        num_mesh_vertices = mesh.parts[part].n_vertices

        if num_mesh_vertices != len(indices):
            msg = (
                f"The {part} hemisphere has {len(indices)} vertices in the surface metadata but "
                f"{num_mesh_vertices} in its white surface"
            )
            raise ValueError(msg)

    roi_mapping = {
        -2: "overlap",
        -1: "unassigned",
        **{int(i): str(name) for i, name in zip(roi_ids, roi_names, strict=True)},
    }

    units = pd.DataFrame(
        {
            "hemisphere": np.where(vertex_hemisphere_id[order] == 0, "left", "right"),
            "roi_distance_mm": roi_distance[order],
        },
    )

    return Dataset(
        name="numerosity-four-conditions",
        response=vertex_response[order],
        roi_index=vertex_roi_index[order],
        roi_mapping=roi_mapping,
        units=units,
        mesh=mesh,
        hemisphere=options.hemisphere,
        task=task,
        files=fetch.files,
    )


def load_dataset(  # noqa: PLR0913
    name: str,
    *,
    dest_dir: str | os.PathLike | None = None,
    hemisphere: str | None = None,
    split: str | None = None,
    surface_type: str | None = None,
    task: str | None = None,
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
        Data split to load the response from, for datasets that provide splits. Required for those datasets unless
        they have a default split, which :func:`~prfmodel.examples.describe_dataset` reports.
    surface_type : str, optional
        Surface type to load, for datasets that provide surfaces. Must be either `"flat"`, `"inflated"`,
        `"pia"` (or `"pial"`), or `"wm"` (for white matter). If `None` (the default), uses the default of the
        dataset.
    task : str, optional
        Experimental task to load the response and stimulus of, for datasets that provide several tasks. If `None`
        (the default), uses the default of the dataset.
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

    Load the gray matter voxel response of a single subject together with its own inflated cortical surface.

    >>> dataset = load_dataset("7t-aot-visual", surface_type="inflated")  # doctest: +SKIP
    >>> dataset.response.shape  # doctest: +SKIP
    (105867, 340)
    >>> dataset.mask.shape  # doctest: +SKIP
    (71, 81, 91)

    Load two splits of the same response, for cross-validation.

    >>> odd = load_dataset("numerosity-timing", hemisphere="left", split="odd")  # doctest: +SKIP
    >>> even = load_dataset("numerosity-timing", hemisphere="left", split="even")  # doctest: +SKIP
    >>> odd.response.shape  # doctest: +SKIP
    (5436, 176)
    >>> odd.response.shape == even.response.shape  # doctest: +SKIP
    True

    Load the same stimulus in two sessions, here with a bar that is warped logarithmically with eccentricity.

    >>> first = load_dataset("3t-chn-visual", split="ses-01_run-01", task="logbar")  # doctest: +SKIP
    >>> second = load_dataset("3t-chn-visual", split="ses-02_run-01", task="logbar")  # doctest: +SKIP
    >>> first.response.shape  # doctest: +SKIP
    (221709, 305)
    >>> first.roi_mapping  # doctest: +SKIP
    {0: 'other', 1: 'V1', 2: 'V2', 3: 'V3'}

    Load the response to a contrast sensitivity stimulus, averaged over all runs or, for cross-validation, over the
    odd runs only.

    >>> csf = load_dataset("7t-ncsf-visual", task="csf")  # doctest: +SKIP
    >>> csf.response.shape  # doctest: +SKIP
    (118584, 214)
    >>> odd = load_dataset("7t-ncsf-visual", task="csf", split="odd")  # doctest: +SKIP
    >>> odd.units.columns.tolist()  # doctest: +SKIP
    ['hemisphere', 'benson14_angle', 'benson14_eccen', 'benson14_sigma']

    Load an intracranial response together with the channels table that describes it.

    >>> broadband = load_dataset("visual-ecog-broadband")  # doctest: +SKIP
    >>> broadband.response.shape  # doctest: +SKIP
    (136, 224)
    >>> broadband.units.loc[0, ["name", "group", "wangarea"]].tolist()  # doctest: +SKIP
    ['GA33', 'grid', 'none']

    """
    spec = _get_spec(name)
    options = validate_options(spec, hemisphere=hemisphere, split=split, surface_type=surface_type, task=task)

    fetcher = FileFetcher(
        spec,
        data_dir=get_data_dir(dest_dir),
        checksums=load_checksums(),
        download=download,
    )

    return spec.loader(fetcher, options)
