"""Test loading example datasets."""

from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nibabel.affines import apply_affine
from prfmodel.examples import Dataset
from prfmodel.examples import load_dataset
from prfmodel.examples._fetch import FileFetcher
from prfmodel.examples._options import Options
from prfmodel.examples._options import validate_options
from prfmodel.examples._registry import DatasetSpec
from prfmodel.examples._registry import _get_spec
from prfmodel.examples._registry import load_checksums
from prfmodel.stimuli import CSFStimulus
from prfmodel.stimuli import PRFStimulus

_NUM_AOT_VOXELS = 105867
_NUM_AOT_VERTICES_L = 153144
_NUM_AOT_FRAMES = 340
_AOT_VOLUME_SHAPE = (71, 81, 91)

# The '7t-retbar-visual' response is given on the vertices of these meshes
_NUM_HCP_VERTICES_L = 59292
_NUM_HCP_VERTICES_R = 59292

_NUM_RETBAR_FRAMES = 120

_NUM_NUMEROSITY_NODES_L = 5436
_NUM_NUMEROSITY_NODES_R = 4889
_NUM_NUMEROSITY_FRAMES = 176
_NUM_NUMEROSITY_ROIS = 8

_NUM_CHN_VERTICES_L = 221709
_NUM_CHN_FRAMES = 305
_CHN_GRID_SIZE = 108

_NUM_ECOG_CHANNELS = 136
_NUM_ECOG_FRAMES = 224
_NUM_ECOG_ROIS = 15
_ECOG_GRID_SIZE = 100

# The bar apertures span 16.6 degrees of visual angle over the 100 pixels of the grid
_ECOG_PIXEL_DEGREES = 0.166

# The broadband response is a percentage change that reaches into the hundreds
_ECOG_BROADBAND_FLOOR = 100.0

_NUM_NCSF_FRAMES = {"csf": 214, "prf": 248}
_NCSF_FRAME_DURATION = 1.5
_NCSF_GRID_SIZE = 100

# The synthetic nCSF files have a different number of frames per response file, which shows which one was loaded
_NUM_FAKE_NCSF_VERTICES = 4
_NUM_FAKE_NCSF_FRAMES = {"csf_all": 2, "csf_odd": 3, "csf_even": 4, "prf_all": 5}
_FAKE_NCSF_ROIS = {0: "unknown", 1: "V1"}

# Values of the right hemisphere are offset from those of the left one, which shows the order of the hemispheres
_FAKE_NCSF_HEMISPHERE_OFFSET = {"L": 0.0, "R": 100.0}

# Numerosity four conditions dataset
_NUMEROSITY_FOUR_TASKS = ("area", "average", "circ", "dens", "size")

# The real 'numerosity-four-conditions' volume and surfaces
_NUM_NUMEROSITY_FOUR_VOXELS = 597042
_NUM_NUMEROSITY_FOUR_FRAMES = 44
_NUMEROSITY_FOUR_VOLUME_SHAPE = (165, 192, 147)
_NUM_NUMEROSITY_FOUR_VERTICES_L = 155826
_NUM_NUMEROSITY_FOUR_VERTICES_R = 155716

# In surface (tkRAS) coordinates, a voxel lies this far from where the affine of the mask places it
_NUMEROSITY_FOUR_SURFACE_OFFSET = np.array([-0.5, 17.5, -18.5])

# The synthetic 'numerosity-four-conditions' volume has five voxels in its mask. The voxel table lists them in C
# order and then appends one zero-filled voxel outside the mask, like the real table appends subcortical voxels.
_FAKE_NUMEROSITY_FOUR_MASK = np.zeros((2, 3, 2), dtype=np.uint8)
_FAKE_NUMEROSITY_FOUR_MASK[[0, 0, 1, 1, 1], [0, 2, 0, 1, 2], [1, 0, 1, 0, 1]] = 1
_FAKE_NUMEROSITY_FOUR_OUTSIDE_VOXEL = (0, 0, 0)
_FAKE_NUMEROSITY_FOUR_AFFINE = np.array(
    [[1.0, 0.0, 0.0, -84.0], [0.0, 1.0, 0.0, -124.0], [0.0, 0.0, 1.0, -59.0], [0.0, 0.0, 0.0, 1.0]],
)
_NUM_FAKE_NUMEROSITY_FOUR_VOXELS = 5
_NUM_FAKE_NUMEROSITY_FOUR_FRAMES = 3
_NUM_FAKE_NUMEROSITY_FOUR_VERTICES_PER_HEMISPHERE = 3

# Values of the voxels in the mask, followed by those of the appended voxel, which the loader drops
_FAKE_NUMEROSITY_FOUR_HEMISPHERE_ID = np.array([0, 1, 0, 1, 0, -1], dtype=np.int8)
_FAKE_NUMEROSITY_FOUR_ROI_INDEX = np.array([0, 1, -1, 2, -1, -1], dtype=np.int32)
_FAKE_NUMEROSITY_FOUR_ROIS = {-2: "overlap", -1: "unassigned", 0: "NTO", 1: "NLO", 2: "NPO"}
_FAKE_NUMEROSITY_FOUR_INVALID_VOXEL = 2


pytest_skip_examples = pytest.mark.examples


def _write_gifti(path: Path, arrays: list[np.ndarray], labels: dict[int, str] | None = None) -> None:
    image = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(array) for array in arrays])

    for key, name in (labels or {}).items():
        label = nib.gifti.GiftiLabel(key=key)
        label.label = name
        image.labeltable.labels.append(label)

    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, path)


# Writes tiny stand-ins for the files of the nCSF dataset at the paths that the registry expects
def _write_fake_ncsf_files(data_dir: Path) -> None:
    spec = _get_spec("7t-ncsf-visual")
    vertices = np.arange(_NUM_FAKE_NCSF_VERTICES, dtype=np.float32)

    for key, member in spec.files.items():
        path = data_dir / member
        offset = _FAKE_NCSF_HEMISPHERE_OFFSET.get(key[-1], 0.0)

        if key.startswith("response_"):
            num_frames = _NUM_FAKE_NCSF_FRAMES[key.removeprefix("response_")[:-2]]
            _write_gifti(path, [vertices + offset + frame for frame in range(num_frames)])
        elif key.startswith("benson14_"):
            _write_gifti(path, [vertices + offset])
        elif key.startswith("roi_"):
            _write_gifti(path, [np.array([0, 1, 1, 0], dtype=np.int32)], labels=_FAKE_NCSF_ROIS)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, design=np.zeros((5, 2, 3)), grid=np.zeros((2, 3, 2)))


def _fake_numerosity_four_response(task: str) -> np.ndarray:
    """Return the response of the synthetic voxels in the mask to a task.

    Voxel v of task t has the response (100 + c, 100 - c, 100) with c = 10 v + t + 1, which shows both which voxel
    and which task was loaded, and that the values are returned as stored.
    """
    change = 10.0 * np.arange(_NUM_FAKE_NUMEROSITY_FOUR_VOXELS) + _NUMEROSITY_FOUR_TASKS.index(task) + 1
    return np.stack([100.0 + change, 100.0 - change, np.full(_NUM_FAKE_NUMEROSITY_FOUR_VOXELS, 100.0)], axis=1)


# Writes tiny stand-ins for the files of the four-condition numerosity dataset at the paths that the registry expects
def _write_fake_numerosity_four_files(
    data_dir: Path,
    voxel_coordinates: np.ndarray | None = None,
    hemisphere_id: np.ndarray = _FAKE_NUMEROSITY_FOUR_HEMISPHERE_ID,
) -> None:
    spec = _get_spec("numerosity-four-conditions")

    if voxel_coordinates is None:
        voxel_coordinates = np.argwhere(_FAKE_NUMEROSITY_FOUR_MASK)

    # The appended voxel lies outside the mask, has a zero response, and is invalid, like in the real table
    voxel_mapping = np.vstack([voxel_coordinates, _FAKE_NUMEROSITY_FOUR_OUTSIDE_VOXEL]).astype(np.int32)
    valid = np.arange(_NUM_FAKE_NUMEROSITY_FOUR_VOXELS + 1) != _FAKE_NUMEROSITY_FOUR_INVALID_VOXEL
    valid[-1] = False

    for key, member in spec.files.items():
        path = data_dir / member
        path.parent.mkdir(parents=True, exist_ok=True)

        if key.startswith("response_"):
            response = _fake_numerosity_four_response(key.removeprefix("response_"))
            np.savez(path, response=np.vstack([response, np.zeros((1, _NUM_FAKE_NUMEROSITY_FOUR_FRAMES))]))
        elif key == "mask":
            # The real mask is stored in Fortran order
            np.savez(path, mask=np.asfortranarray(_FAKE_NUMEROSITY_FOUR_MASK), affine=_FAKE_NUMEROSITY_FOUR_AFFINE)
        elif key == "voxel_metadata":
            np.savez(
                path,
                voxel_mapping=voxel_mapping,
                voxel_roi_index=_FAKE_NUMEROSITY_FOUR_ROI_INDEX,
                voxel_hemisphere_id=hemisphere_id,
            )
        elif key == "validity":
            np.savez(path, **{f"{task.capitalize()}_valid": valid for task in _NUMEROSITY_FOUR_TASKS})
        elif key == "label_names":
            np.savez(path, roi_ids=np.array([0, 1, 2]), roi_names=np.array(["NTO", "NLO", "NPO"]))
        elif key.endswith("_white"):
            num_vertices = _NUM_FAKE_NUMEROSITY_FOUR_VERTICES_PER_HEMISPHERE
            coords = np.arange(num_vertices * 3, dtype=np.float32).reshape(num_vertices, 3)

            # Real surfaces end with volume information, and nibabel warns when reading a surface without it
            volume_info = {
                "head": np.array([2, 0, 20], dtype=np.int32),
                "valid": "1  # volume info valid",
                "filename": "volume.nii",
                "volume": np.array([256, 256, 256]),
                "voxelsize": np.array([1.0, 1.0, 1.0]),
                "xras": np.array([-1.0, 0.0, 0.0]),
                "yras": np.array([0.0, 0.0, -1.0]),
                "zras": np.array([0.0, 1.0, 0.0]),
                "cras": np.array([0.0, 0.0, 0.0]),
            }
            nib.freesurfer.write_geometry(path, coords, np.array([[0, 1, 2]], dtype=np.int32), volume_info=volume_info)
        elif key.endswith("_vertex_indices"):
            np.save(path, np.arange(_NUM_FAKE_NUMEROSITY_FOUR_VERTICES_PER_HEMISPHERE))
        else:
            # The loader only exposes the paths of the remaining files, so their contents do not matter
            path.write_bytes(b"placeholder")


def test_load_dataset_unknown_name():
    """Test that loading an unknown dataset lists the valid names."""
    with pytest.raises(ValueError, match="numerosity-timing"):
        load_dataset("does-not-exist")


def test_load_dataset_unsupported_option():
    """Test that an option a dataset does not accept is refused rather than ignored."""
    with pytest.raises(ValueError, match="does not accept the option 'split'"):
        load_dataset("7t-retbar-visual", split="odd")


def test_load_dataset_missing_split():
    """Test that a dataset with splits refuses to load without one."""
    with pytest.raises(ValueError, match="requires the option 'split'"):
        load_dataset("numerosity-timing", hemisphere="left")


def test_load_dataset_invalid_split():
    """Test that an unknown split is refused."""
    with pytest.raises(ValueError, match="must be one of"):
        load_dataset("numerosity-timing", split="third")


def test_load_dataset_unsupported_task():
    """Test that a task is refused by a dataset that provides only one."""
    with pytest.raises(ValueError, match="does not accept the option 'task'"):
        load_dataset("numerosity-timing", split="odd", task="logbar")


def test_load_dataset_invalid_task():
    """Test that an unknown task is refused."""
    with pytest.raises(ValueError, match="must be one of"):
        load_dataset("3t-chn-visual", split="ses-01_run-01", task="wedge")


def test_validate_options_default_task():
    """Test that the first task of a dataset is used when none is requested."""
    options = validate_options(_get_spec("3t-chn-visual"), split="ses-01_run-01")

    assert options.task == "fixedbar"
    assert options.hemisphere == "left"


def test_validate_options_no_task():
    """Test that datasets without tasks resolve the task to None."""
    assert validate_options(_get_spec("numerosity-timing"), split="odd").task is None


def test_validate_options_default_split():
    """Test that a dataset with a default split uses it when none is requested."""
    assert validate_options(_get_spec("7t-ncsf-visual")).split == "all"
    assert validate_options(_get_spec("7t-ncsf-visual"), split="odd").split == "odd"


def test_load_ncsf_visual_refuses_a_split_of_the_prf_task(tmp_path: Path):
    """Test that the pRF task, which was not split, refuses a split before fetching anything."""
    with pytest.raises(ValueError, match="only the split 'all'"):
        load_dataset("7t-ncsf-visual", task="prf", split="odd", dest_dir=tmp_path, download=False)

    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(("task", "split"), [("csf", "all"), ("csf", "odd"), ("csf", "even"), ("prf", "all")])
@pytest.mark.parametrize("hemisphere", ["left", "right", "both"])
def test_load_ncsf_visual_from_local_files(tmp_path: Path, hemisphere: str, task: str, split: str):
    """Test that the nCSF loader assembles the response, regions of interest, template maps, and stimulus."""
    _write_fake_ncsf_files(tmp_path)

    dataset = load_dataset(
        "7t-ncsf-visual",
        hemisphere=hemisphere,
        task=task,
        split=split,
        dest_dir=tmp_path,
        download=False,
    )

    codes = {"left": ["L"], "right": ["R"], "both": ["L", "R"]}[hemisphere]
    num_units = _NUM_FAKE_NCSF_VERTICES * len(codes)
    vertices = np.arange(_NUM_FAKE_NCSF_VERTICES)
    expected_first_frame = np.concatenate([vertices + _FAKE_NCSF_HEMISPHERE_OFFSET[code] for code in codes])

    assert dataset.response.shape == (num_units, _NUM_FAKE_NCSF_FRAMES[f"{task}_{split}"])
    np.testing.assert_array_equal(dataset.response[:, 0], expected_first_frame)
    assert dataset.roi_index.shape == (num_units,)
    assert dataset.roi_mapping == _FAKE_NCSF_ROIS
    assert dataset.units.columns.tolist() == ["hemisphere", "benson14_angle", "benson14_eccen", "benson14_sigma"]
    expected_hemispheres = np.repeat([{"L": "left", "R": "right"}[code] for code in codes], _NUM_FAKE_NCSF_VERTICES)
    np.testing.assert_array_equal(dataset.units["hemisphere"], expected_hemispheres)
    np.testing.assert_array_equal(dataset.units["benson14_eccen"], expected_first_frame)
    assert (dataset.hemisphere, dataset.task, dataset.split) == (hemisphere, task, split)
    assert ("stimulus_prf" in dataset.files) == (task == "prf")

    if task == "csf":
        assert isinstance(dataset.stimulus, CSFStimulus)
    else:
        assert isinstance(dataset.stimulus, PRFStimulus)
        assert dataset.stimulus.dimension_labels == ["y", "x"]


def test_load_numerosity_four_conditions_defaults_to_the_average(tmp_path: Path):
    """Test that the four-condition dataset loads the average of the conditions when no task is requested."""
    _write_fake_numerosity_four_files(tmp_path)

    dataset = load_dataset("numerosity-four-conditions", dest_dir=tmp_path, download=False)

    assert dataset.task == "average"
    assert "response_average" in dataset.files
    assert not any(key.startswith("response_") and key != "response_average" for key in dataset.files)


def test_load_numerosity_four_conditions_invalid_task():
    """Test that a task name that the dataset does not provide is refused."""
    with pytest.raises(ValueError, match="must be one of"):
        load_dataset("numerosity-four-conditions", task="density")


def test_load_numerosity_four_conditions_has_no_hemisphere_option():
    """Test that the four-condition dataset, whose response covers the whole brain, refuses a hemisphere."""
    with pytest.raises(ValueError, match="does not accept the option 'hemisphere'"):
        load_dataset("numerosity-four-conditions", hemisphere="left")


def test_validate_options_numerosity_four_conditions():
    """Test that the four-condition dataset keeps the requested task and defaults to the average."""
    spec = _get_spec("numerosity-four-conditions")

    assert validate_options(spec, task="circ").task == "circ"
    assert validate_options(spec).task == "average"


@pytest.mark.parametrize("task", _NUMEROSITY_FOUR_TASKS)
def test_load_numerosity_four_conditions_from_local_files(tmp_path: Path, task: str):
    """Test that the loader returns the raw response of the mask voxels together with the mask and the surfaces."""
    _write_fake_numerosity_four_files(tmp_path)

    dataset = load_dataset("numerosity-four-conditions", task=task, dest_dir=tmp_path, download=False)

    # The voxel appended after the mask voxels is dropped and the response is returned as stored
    np.testing.assert_array_equal(dataset.response, _fake_numerosity_four_response(task))
    assert dataset.task == task
    assert dataset.hemisphere is None

    # The rows of the response are the mask voxels in C order, as the contract of the mask requires
    np.testing.assert_array_equal(np.asarray(dataset.mask.dataobj), _FAKE_NUMEROSITY_FOUR_MASK)
    np.testing.assert_array_equal(dataset.mask.affine, _FAKE_NUMEROSITY_FOUR_AFFINE)
    assert np.count_nonzero(dataset.mask.dataobj) == dataset.response.shape[0]

    np.testing.assert_array_equal(dataset.roi_index, _FAKE_NUMEROSITY_FOUR_ROI_INDEX[:-1])
    assert dataset.roi_mapping == _FAKE_NUMEROSITY_FOUR_ROIS

    assert dataset.units.columns.tolist() == ["hemisphere", "valid"]
    assert dataset.units["hemisphere"].tolist() == ["left", "right", "left", "right", "left"]
    assert dataset.units["valid"].tolist() == [True, True, False, True, True]

    assert sorted(dataset.mesh.parts) == ["left", "right"]
    assert dataset.mesh.n_vertices == 2 * _NUM_FAKE_NUMEROSITY_FOUR_VERTICES_PER_HEMISPHERE

    assert set(dataset.files) == {
        f"response_{task}",
        "mask",
        "voxel_metadata",
        "validity",
        "label_names",
        "surface_mapping_method2",
        "surface_metadata",
        "roi_data",
        "lh_white",
        "rh_white",
        "lh_flat",
        "rh_flat",
        "lh_vertex_indices",
        "rh_vertex_indices",
    }


def test_load_numerosity_four_conditions_refuses_a_voxel_table_out_of_mask_order(tmp_path: Path):
    """Test that a voxel table whose rows are not the mask voxels in C order is refused."""
    _write_fake_numerosity_four_files(tmp_path, voxel_coordinates=np.argwhere(_FAKE_NUMEROSITY_FOUR_MASK)[::-1])

    with pytest.raises(ValueError, match="not the voxels of the mask in C order"):
        load_dataset("numerosity-four-conditions", dest_dir=tmp_path, download=False)


def test_load_numerosity_four_conditions_refuses_a_mask_voxel_without_hemisphere(tmp_path: Path):
    """Test that a voxel of the mask on the midline, which belongs to neither hemisphere, is refused."""
    hemisphere_id = _FAKE_NUMEROSITY_FOUR_HEMISPHERE_ID.copy()
    hemisphere_id[0] = -1
    _write_fake_numerosity_four_files(tmp_path, hemisphere_id=hemisphere_id)

    with pytest.raises(ValueError, match="neither the left nor the right hemisphere"):
        load_dataset("numerosity-four-conditions", dest_dir=tmp_path, download=False)


def test_load_dataset_invalid_hemisphere():
    """Test that an unknown hemisphere is refused."""
    with pytest.raises(ValueError, match="must be one of"):
        load_dataset("7t-retbar-visual", hemisphere="middle")


def test_load_dataset_invalid_surface_type():
    """Test that an unknown surface type is refused."""
    with pytest.raises(ValueError, match="must be one of"):
        load_dataset("hcp-999999-surface", surface_type="folded")


def test_load_dataset_validates_before_downloading(fake_registry: DatasetSpec):
    """Test that options are validated before anything is downloaded."""
    with pytest.raises(ValueError, match="does not accept the option 'split'"):
        load_dataset("fake-dataset", split="odd", dest_dir="never_created")


def test_load_dataset_returns_a_dataset(fake_registry: DatasetSpec, data_dir: Path):
    """Test that loading a dataset returns a dataset object holding the files it fetched."""
    dataset = load_dataset("fake-dataset", dest_dir=data_dir)

    assert isinstance(dataset, Dataset)
    assert sorted(dataset.files) == ["first", "second"]


def test_hemisphere_codes_are_a_tuple():
    """Test that a single hemisphere resolves to a one-element tuple rather than a bare string."""
    assert Options(hemisphere="left").hemisphere_codes == ("L",)
    assert Options(hemisphere="right").hemisphere_codes == ("R",)
    assert Options(hemisphere="both").hemisphere_codes == ("L", "R")


def test_dataset_str_skips_missing_fields():
    """Test that the string representation leaves out fields that a dataset does not provide."""
    text = str(Dataset(name="example", hemisphere="left"))

    assert text == "Dataset(name=example, hemisphere=left, files=[])"


def test_dataset_str_shows_mask_shape():
    """Test that the string representation shows the shape of a mask image rather than its header."""
    mask = nib.Nifti1Image(np.ones((2, 3, 4), dtype=np.uint8), affine=np.eye(4))
    text = str(Dataset(name="example", mask=mask))

    assert text == "Dataset(name=example, mask=Nifti1Image[2, 3, 4], files=[])"


def test_dataset_str_shows_units_shape():
    """Test that the string representation shows the shape of a units frame rather than its contents."""
    units = pd.DataFrame({"name": ["GA33", "GA34"], "group": ["grid", "grid"]})
    text = str(Dataset(name="example", units=units))

    assert text == "Dataset(name=example, units=frame[2, 2], files=[])"


@pytest_skip_examples
@pytest.mark.parametrize("surface_type", ["flat", "inflated", "pia", "pial", "wm"])
def test_load_aot_visual(surface_type: str):
    """Test that the AOT dataset returns the masked response, the mask, the stimulus, and the surface."""
    dataset = load_dataset("7t-aot-visual", surface_type=surface_type)

    assert dataset.response.shape == (_NUM_AOT_VOXELS, _NUM_AOT_FRAMES)
    assert dataset.mask.shape == _AOT_VOLUME_SHAPE
    assert np.count_nonzero(dataset.mask.get_fdata()) == _NUM_AOT_VOXELS
    assert isinstance(dataset.stimulus, PRFStimulus)
    assert dataset.stimulus.design.shape[0] == _NUM_AOT_FRAMES
    assert dataset.mesh.parts["left"].n_vertices == _NUM_AOT_VERTICES_L
    assert {"pia_lh", "pia_rh", "wm_lh", "wm_rh"} <= set(dataset.files)


@pytest_skip_examples
@pytest.mark.parametrize("surface_type", ["flat", "inflated", "pia", "pial", "wm"])
def test_load_hcp_surface(surface_type: str):
    """Test that the surface dataset returns a mesh, an atlas, and the paths of every surface."""
    dataset = load_dataset("hcp-999999-surface", surface_type=surface_type)

    assert dataset.mesh.n_vertices == _NUM_HCP_VERTICES_L + _NUM_HCP_VERTICES_R
    assert set(dataset.atlas) == {"left", "right"}
    assert dataset.atlas["left"].shape == (_NUM_HCP_VERTICES_L,)
    assert "wm_lh" in dataset.files


@pytest_skip_examples
def test_load_hcp_surface_extracts_only_what_is_used():
    """Test that the unused parts of the surface archive are not extracted."""
    dataset = load_dataset("hcp-999999-surface")
    data_dir = dataset.files["flat_lh"].parent.parent

    assert not (data_dir / "anatomicals").exists()
    assert not (data_dir / "overlays.svg").exists()
    assert not (data_dir / "surfaces" / "bks").exists()


@pytest_skip_examples
@pytest.mark.parametrize(
    ("hemisphere", "num_units"),
    [
        ("left", _NUM_HCP_VERTICES_L),
        ("right", _NUM_HCP_VERTICES_R),
        ("both", _NUM_HCP_VERTICES_L + _NUM_HCP_VERTICES_R),
    ],
)
def test_load_retbar_visual(hemisphere: str, num_units: int):
    """Test that the visual dataset returns the response and the raw design of the stimulus."""
    dataset = load_dataset("7t-retbar-visual", hemisphere=hemisphere)

    assert dataset.response.shape == (num_units, _NUM_RETBAR_FRAMES)
    assert dataset.files["design"].exists()
    assert dataset.stimulus is None


@pytest_skip_examples
@pytest.mark.parametrize("split", ["odd", "even"])
@pytest.mark.parametrize(
    ("hemisphere", "num_units"),
    [
        ("left", _NUM_NUMEROSITY_NODES_L),
        ("right", _NUM_NUMEROSITY_NODES_R),
        ("both", _NUM_NUMEROSITY_NODES_L + _NUM_NUMEROSITY_NODES_R),
    ],
)
def test_load_numerosity_timing(hemisphere: str, num_units: int, split: str):
    """Test that the numerosity dataset returns the response, the regions of interest, and the stimulus."""
    dataset = load_dataset("numerosity-timing", hemisphere=hemisphere, split=split)

    assert dataset.response.shape == (num_units, _NUM_NUMEROSITY_FRAMES)
    assert dataset.roi_index.shape == (num_units,)
    assert len(dataset.roi_mapping) == _NUM_NUMEROSITY_ROIS
    assert dataset.roi_mapping[0] == "NTO"
    assert dataset.split == split
    assert isinstance(dataset.stimulus, PRFStimulus)


@pytest_skip_examples
def test_load_numerosity_timing_downloads_only_what_is_loaded(tmp_path: Path):
    """Test that loading one split of one hemisphere downloads only the two files it needs."""
    load_dataset("numerosity-timing", hemisphere="left", split="odd", dest_dir=tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "sub-S1_hemi-L_desc-numerosity_dseg.label.gii",
        "sub-S1_hemi-L_desc-odd_bold.func.gii",
    ]


@pytest_skip_examples
@pytest.mark.parametrize("task", ["fixedbar", "logbar"])
@pytest.mark.parametrize("split", ["ses-01_run-01", "ses-01_run-02", "ses-02_run-01"])
def test_load_chn_visual(split: str, task: str):
    """Test that the CHN dataset returns the response, the regions of interest, the stimulus, and the surface."""
    dataset = load_dataset("3t-chn-visual", hemisphere="left", split=split, task=task)

    assert dataset.response.shape == (_NUM_CHN_VERTICES_L, _NUM_CHN_FRAMES)
    assert dataset.roi_index.shape == (_NUM_CHN_VERTICES_L,)
    assert dataset.roi_mapping == {0: "other", 1: "V1", 2: "V2", 3: "V3"}
    assert set(np.unique(dataset.roi_index)) == {0, 1, 2, 3}
    assert dataset.mesh.parts["left"].n_vertices == _NUM_CHN_VERTICES_L
    assert isinstance(dataset.stimulus, PRFStimulus)
    assert dataset.stimulus.design.shape == (_NUM_CHN_FRAMES, _CHN_GRID_SIZE, _CHN_GRID_SIZE)
    assert dataset.split == split
    assert dataset.task == task


@pytest_skip_examples
def test_load_chn_visual_downloads_only_what_is_loaded(tmp_path: Path):
    """Test that loading one split, task, and hemisphere downloads only the files it needs."""
    load_dataset("3t-chn-visual", hemisphere="left", split="ses-01_run-02", task="logbar", dest_dir=tmp_path)

    assert sorted(p.name for p in tmp_path.rglob("*") if p.is_file()) == [
        "sub-01_hemi-L_space-fsnative_v1.label",
        "sub-01_hemi-L_space-fsnative_v2.label",
        "sub-01_hemi-L_space-fsnative_v3.label",
        "sub-01_ses-01_acq-MEMPRvNav_rec-RMS_hemi-L_inflated.surf.gii",
        "sub-01_ses-01_task-logbar_run-2_hemi-L_space-fsnative_desc-preproc_bold.func.gii",
        "task-logbar_run-2_desc-down_aperture.mat",
    ]


@pytest_skip_examples
@pytest.mark.parametrize(("task", "split"), [("csf", "all"), ("csf", "odd"), ("csf", "even"), ("prf", "all")])
def test_load_ncsf_visual(task: str, split: str):
    """Test that the nCSF dataset returns the response, the regions of interest, the template maps, and the stimulus."""
    dataset = load_dataset("7t-ncsf-visual", hemisphere="both", task=task, split=split)

    num_units = _NUM_HCP_VERTICES_L + _NUM_HCP_VERTICES_R

    assert dataset.response.shape == (num_units, _NUM_NCSF_FRAMES[task])
    assert dataset.roi_index.shape == (num_units,)
    assert dataset.roi_mapping[1] == "V1"
    assert dataset.units.shape == (num_units, 4)
    assert (dataset.units["hemisphere"] == "left").sum() == _NUM_HCP_VERTICES_L
    assert dataset.stimulus.frame_duration == _NCSF_FRAME_DURATION

    if task == "prf":
        assert dataset.stimulus.design.shape == (_NUM_NCSF_FRAMES["prf"], _NCSF_GRID_SIZE, _NCSF_GRID_SIZE)


@pytest_skip_examples
def test_load_ncsf_visual_downloads_only_what_is_loaded(tmp_path: Path):
    """Test that loading the odd split of the CSF task of one hemisphere downloads only the files it needs."""
    load_dataset("7t-ncsf-visual", hemisphere="left", task="csf", split="odd", dest_dir=tmp_path)

    assert sorted(p.name for p in tmp_path.rglob("*") if p.is_file()) == [
        "sub-16112020_hemi-L_space-fsLR_den-59k_atlas-benson14_angle.func.gii",
        "sub-16112020_hemi-L_space-fsLR_den-59k_atlas-benson14_dseg.label.gii",
        "sub-16112020_hemi-L_space-fsLR_den-59k_atlas-benson14_eccen.func.gii",
        "sub-16112020_hemi-L_space-fsLR_den-59k_atlas-benson14_sigma.func.gii",
        "sub-16112020_task-csf_split-odd_hemi-L_space-fsLR_den-59k_desc-detrend3_bold.func.gii",
    ]


@pytest_skip_examples
def test_load_visual_ecog_broadband():
    """Test that the ECoG recording returns its response together with its stimulus."""
    dataset = load_dataset("visual-ecog-broadband")

    assert dataset.response.shape == (_NUM_ECOG_CHANNELS, _NUM_ECOG_FRAMES)
    assert np.all(np.isfinite(dataset.response))
    assert isinstance(dataset.stimulus, PRFStimulus)
    assert dataset.stimulus.design.shape == (_NUM_ECOG_FRAMES, _ECOG_GRID_SIZE, _ECOG_GRID_SIZE)


@pytest_skip_examples
def test_load_visual_ecog_broadband_holds_a_percentage_change():
    """Test that the response is the broadband power change in percent rather than a log ratio."""
    assert load_dataset("visual-ecog-broadband").response.max() > _ECOG_BROADBAND_FLOOR


@pytest_skip_examples
def test_load_visual_ecog_broadband_downloads_only_what_is_loaded(tmp_path: Path):
    """Test that loading the dataset downloads the single file that holds its response."""
    load_dataset("visual-ecog-broadband", dest_dir=tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "p10_freq_spectra-timeseries_avg-runs_bbS.mat",
    ]


@pytest_skip_examples
def test_load_visual_ecog_broadband_describes_every_channel():
    """Test that the channels frame has one row per response row, in the same order."""
    dataset = load_dataset("visual-ecog-broadband")
    channels = dataset.units

    assert len(channels) == _NUM_ECOG_CHANNELS
    assert channels.loc[0, "name"] == "GA33"
    assert set(channels["group"]) == {"HDgrid", "grid", "strip"}

    # Every channel is a common average referenced electrode of the right hemisphere
    assert set(channels["reference"]) == {"car"}
    assert set(channels["hemisphere"]) == {"R"}


@pytest_skip_examples
def test_load_visual_ecog_broadband_groups_channels_into_visual_areas():
    """Test that the Wang atlas labels of the channels are grouped into the areas of the source study."""
    dataset = load_dataset("visual-ecog-broadband")
    areas = [dataset.roi_mapping[index] for index in dataset.roi_index]

    assert len(dataset.roi_mapping) == _NUM_ECOG_ROIS
    assert dataset.roi_index.shape == (_NUM_ECOG_CHANNELS,)

    # The ventral and dorsal halves of V3 are one area, and TO1 and TO2 are one area
    assert set(areas) <= set(dataset.roi_mapping.values())
    assert "V3v" not in areas
    assert areas[:3] == ["none", "none", "none"]
    assert areas[dataset.units["name"].tolist().index("GA51")] == "IPS"


@pytest_skip_examples
def test_load_visual_ecog_broadband_puts_the_stimulus_in_degrees_of_visual_angle():
    """Test that the stimulus grid spans the visual angle the bar apertures were shown at."""
    grid = load_dataset("visual-ecog-broadband").stimulus.grid

    y, x = grid[..., 0], grid[..., 1]

    # x grows to the right across the columns and y grows upwards, against the row order
    assert x[0, 0] < x[0, -1]
    assert y[0, 0] > y[-1, 0]
    assert x[0, -1] - x[0, 0] == pytest.approx((_ECOG_GRID_SIZE - 1) * _ECOG_PIXEL_DEGREES)
    assert grid.mean() == pytest.approx(0.0)


@pytest_skip_examples
def test_load_numerosity_four_conditions():
    """Test that the four-condition dataset returns the voxels of the mask with their ROIs, units, and the surfaces."""
    dataset = load_dataset("numerosity-four-conditions", task="area")

    assert dataset.response.shape == (_NUM_NUMEROSITY_FOUR_VOXELS, _NUM_NUMEROSITY_FOUR_FRAMES)
    assert dataset.mask.shape == _NUMEROSITY_FOUR_VOLUME_SHAPE
    assert np.count_nonzero(dataset.mask.dataobj) == _NUM_NUMEROSITY_FOUR_VOXELS
    assert dataset.roi_index.shape == (_NUM_NUMEROSITY_FOUR_VOXELS,)
    assert dataset.roi_mapping[-1] == "unassigned"
    assert set(np.unique(dataset.roi_index)) <= set(dataset.roi_mapping)
    assert len(dataset.units) == _NUM_NUMEROSITY_FOUR_VOXELS
    assert set(dataset.units["hemisphere"]) == {"left", "right"}
    assert dataset.mesh.parts["left"].n_vertices == _NUM_NUMEROSITY_FOUR_VERTICES_L
    assert dataset.mesh.parts["right"].n_vertices == _NUM_NUMEROSITY_FOUR_VERTICES_R
    assert dataset.task == "area"

    # Invalid voxels fell outside the scanned volume in some runs, so they have zeros at some frames
    assert (dataset.response[dataset.units["valid"]] != 0).all()


@pytest_skip_examples
def test_load_numerosity_four_conditions_tasks_share_the_voxels():
    """Test that every task has a response on the same voxels but with different values."""
    responses = {
        task: load_dataset("numerosity-four-conditions", task=task).response for task in _NUMEROSITY_FOUR_TASKS
    }

    assert len({response.shape for response in responses.values()}) == 1
    assert not np.allclose(responses["area"], responses["dens"])


@pytest_skip_examples
def test_load_numerosity_four_conditions_maps_vertices_to_rows_of_the_response():
    """Test that the voxel-to-vertex mapping in the files refers to rows of the response and covers the mesh."""
    dataset = load_dataset("numerosity-four-conditions")

    with np.load(dataset.files["surface_mapping_method2"]) as archive:
        offsets = archive["method2_vertex_voxel_offsets"]
        rows = archive["method2_vertex_voxel_rows"]

    assert len(offsets) == dataset.mesh.n_vertices + 1
    assert rows.max() < dataset.response.shape[0]

    # Every voxel that a vertex is mapped to has the ROI and the hemisphere of the vertices it is mapped to
    with np.load(dataset.files["surface_metadata"]) as archive:
        vertex_roi_index = archive["vertex_roi_index"]
        vertex_hemisphere_id = archive["vertex_hemisphere_id"]

    vertex_of_row = np.repeat(np.arange(dataset.mesh.n_vertices), np.diff(offsets))
    voxel_hemisphere = dataset.units["hemisphere"].to_numpy()[rows]
    np.testing.assert_array_equal(voxel_hemisphere == "right", vertex_hemisphere_id[vertex_of_row] == 1)
    assert (dataset.roi_index[rows] == vertex_roi_index[vertex_of_row]).mean() > 0.99  # noqa: PLR2004


@pytest_skip_examples
def test_load_numerosity_four_conditions_documents_the_offset_of_the_affine():
    """Test that the voxels lie at the documented offset from where the affine of the mask places them."""
    spec = _get_spec("numerosity-four-conditions")
    dataset = load_dataset("numerosity-four-conditions")
    fetch = FileFetcher(spec, data_dir=dataset.files["mask"].parent.parent, checksums=load_checksums())

    # The nearest voxel of each vertex and its distance in mm, as computed by the authors in surface coordinates
    with np.load(fetch("surface_mapping_method1")) as archive:
        voxel_of_vertex = archive["method1_vertex_to_voxel"]
        distance = archive["method1_distance_mm"]

    with np.load(fetch("surface_metadata")) as archive:
        vertex_coordinates = archive["vertex_coordinates"]

    has_voxel = voxel_of_vertex >= 0
    voxels = np.argwhere(np.asarray(dataset.mask.dataobj))[voxel_of_vertex[has_voxel]]
    voxel_coordinates = apply_affine(dataset.mask.affine, voxels)

    shifted_voxel_coordinates = voxel_coordinates + _NUMEROSITY_FOUR_SURFACE_OFFSET
    shifted = np.linalg.norm(vertex_coordinates[has_voxel] - shifted_voxel_coordinates, axis=1)
    np.testing.assert_allclose(shifted, distance[has_voxel], atol=1e-3)


@pytest_skip_examples
def test_load_numerosity_four_conditions_downloads_only_what_is_loaded(tmp_path: Path):
    """Test that loading one task downloads only its response and the files of the volume and the surfaces."""
    load_dataset("numerosity-four-conditions", task="circ", dest_dir=tmp_path)

    spec = _get_spec("numerosity-four-conditions")
    keys = [
        "response_circ", "mask", "voxel_metadata", "validity", "label_names",
        "surface_mapping_method2", "surface_metadata", "roi_data",
        "lh_white", "rh_white", "lh_flat", "rh_flat", "lh_vertex_indices", "rh_vertex_indices",
    ]  # fmt: skip

    assert sorted(p.name for p in tmp_path.rglob("*") if p.is_file()) == sorted(Path(spec.files[k]).name for k in keys)
