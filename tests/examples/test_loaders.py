"""Test loading example datasets."""

from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from prfmodel.examples import Dataset
from prfmodel.examples import load_dataset
from prfmodel.examples._options import Options
from prfmodel.examples._options import validate_options
from prfmodel.examples._registry import DatasetSpec
from prfmodel.examples._registry import _get_spec
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

# The synthetic files of 'numerosity-four-conditions' have six vertices on the surface and seven voxels in the volume
_NUM_FAKE_NUMEROSITY_FOUR_FRAMES = 3
_NUM_FAKE_NUMEROSITY_FOUR_VERTICES_PER_HEMISPHERE = 3

# Vertices alternate between hemispheres, so a selection shows that the loader picks them by id rather than position
_FAKE_NUMEROSITY_FOUR_HEMISPHERE_ID = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
_FAKE_NUMEROSITY_FOUR_ROI_INDEX = np.array([0, 1, 2, -1, -2, 0], dtype=np.int32)
_FAKE_NUMEROSITY_FOUR_ROIS = {-2: "overlap", -1: "unassigned", 0: "NTO", 1: "NLO", 2: "NPO"}

# Vertex 3 maps only to voxel 6, which is invalid, and vertex 4 maps to two voxels whose responses are averaged
_FAKE_NUMEROSITY_FOUR_VOXELS_OF_VERTEX = ([0], [1], [2], [6], [3, 4], [5])
_FAKE_NUMEROSITY_FOUR_INVALID_VOXEL = 6

# Voxel v of task t has the percent signal change (10 v + t + 1, -(10 v + t + 1), 0), so the first frame of a vertex
# is this base plus t + 1, which shows both which vertex and which task was loaded
_FAKE_NUMEROSITY_FOUR_FIRST_FRAME_BASE = np.array([0.0, 10.0, 20.0, np.nan, 35.0, 50.0])
_FAKE_NUMEROSITY_FOUR_ROI_DISTANCE = np.array([0.5, 1.5, 2.5, np.nan, np.nan, 5.5])


_NUMEROSITY_FOUR_HEMISPHERE_IDS = {"left": [0], "right": [1], "both": [0, 1]}
_NUMEROSITY_FOUR_PREFIXES = {"left": "lh", "right": "rh"}


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


# Writes tiny stand-ins for the files of the four-condition numerosity dataset at the paths that the registry expects
def _write_fake_numerosity_four_files(data_dir: Path, num_left_white_vertices: int = 3) -> None:
    spec = _get_spec("numerosity-four-conditions")
    num_voxels = _FAKE_NUMEROSITY_FOUR_INVALID_VOXEL + 1
    valid = np.arange(num_voxels) != _FAKE_NUMEROSITY_FOUR_INVALID_VOXEL
    offsets = np.cumsum([0] + [len(voxels) for voxels in _FAKE_NUMEROSITY_FOUR_VOXELS_OF_VERTEX])
    rows = np.concatenate(_FAKE_NUMEROSITY_FOUR_VOXELS_OF_VERTEX)

    for key, member in spec.files.items():
        path = data_dir / member
        path.parent.mkdir(parents=True, exist_ok=True)

        if key.startswith("response_"):
            task_number = _NUMEROSITY_FOUR_TASKS.index(key.removeprefix("response_")) + 1
            change = 10.0 * np.arange(num_voxels) + task_number
            response = np.stack([100.0 + change, 100.0 - change, np.full(num_voxels, 100.0)], axis=1)
            np.savez(path, response=response)
        elif key == "validity":
            np.savez(path, **{f"{task.capitalize()}_valid": valid for task in _NUMEROSITY_FOUR_TASKS})
        elif key == "surface_mapping_method2":
            np.savez(path, method2_vertex_voxel_offsets=offsets, method2_vertex_voxel_rows=rows)
        elif key == "surface_metadata":
            np.savez(
                path,
                vertex_roi_index=_FAKE_NUMEROSITY_FOUR_ROI_INDEX,
                vertex_hemisphere_id=_FAKE_NUMEROSITY_FOUR_HEMISPHERE_ID,
            )
        elif key == "label_names":
            np.savez(path, roi_ids=np.array([0, 1, 2]), roi_names=np.array(["NTO", "NLO", "NPO"]))
        elif key == "roi_data":
            # One distance array over all vertices per ROI (NaN outside it), like the real file
            np.savez(
                path,
                **{
                    f"roi_{roi}_distances_mm": np.where(
                        roi == _FAKE_NUMEROSITY_FOUR_ROI_INDEX, _FAKE_NUMEROSITY_FOUR_ROI_DISTANCE, np.nan,
                    )
                    for roi in (0, 1, 2)
                },
            )
        elif key.endswith("_white"):
            num_vertices = (
                num_left_white_vertices if key == "lh_white" else _NUM_FAKE_NUMEROSITY_FOUR_VERTICES_PER_HEMISPHERE
            )
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

def test_load_numerosity_four_conditions_requires_a_task(tmp_path: Path):
    """Test that the four-condition dataset refuses to load without a task before fetching anything."""
    with pytest.raises(ValueError, match="requires the option 'task'"):
        load_dataset("numerosity-four-conditions", hemisphere="left", dest_dir=tmp_path, download=False)

    assert not any(tmp_path.iterdir())


def test_load_numerosity_four_conditions_invalid_task():
    """Test that a task name that the dataset does not provide is refused."""
    with pytest.raises(ValueError, match="must be one of"):
        load_dataset("numerosity-four-conditions", task="density")


def test_validate_options_numerosity_four_conditions():
    """Test that the four-condition dataset keeps the requested task and defaults to both hemispheres."""
    options = validate_options(_get_spec("numerosity-four-conditions"), task="circ")

    assert options.task == "circ"
    assert options.hemisphere == "both"
    assert options.hemisphere_codes == ("L", "R")


@pytest.mark.parametrize("task", _NUMEROSITY_FOUR_TASKS)
@pytest.mark.parametrize("hemisphere", ["left", "right", "both"])
def test_load_numerosity_four_conditions_from_local_files(tmp_path: Path, hemisphere: str, task: str):
    """Test that the loader projects the voxels onto the vertices and keeps only the requested hemispheres."""
    _write_fake_numerosity_four_files(tmp_path)

    dataset = load_dataset(
        "numerosity-four-conditions",
        hemisphere=hemisphere,
        task=task,
        dest_dir=tmp_path,
        download=False,
    )

    vertices = np.concatenate(
        [np.flatnonzero(h == _FAKE_NUMEROSITY_FOUR_HEMISPHERE_ID) for h in _NUMEROSITY_FOUR_HEMISPHERE_IDS[hemisphere]],
    )
    task_number = _NUMEROSITY_FOUR_TASKS.index(task) + 1

    assert dataset.response.shape == (len(vertices), _NUM_FAKE_NUMEROSITY_FOUR_FRAMES)
    np.testing.assert_allclose(dataset.response[:, 0], _FAKE_NUMEROSITY_FOUR_FIRST_FRAME_BASE[vertices] + task_number)
    np.testing.assert_array_equal(dataset.roi_index, _FAKE_NUMEROSITY_FOUR_ROI_INDEX[vertices])
    assert dataset.roi_mapping == _FAKE_NUMEROSITY_FOUR_ROIS
    assert dataset.mesh.n_vertices == len(vertices)
    assert (dataset.hemisphere, dataset.task) == (hemisphere, task)

    parts = ["left", "right"] if hemisphere == "both" else [hemisphere]
    assert sorted(dataset.mesh.parts) == parts

    # Only the surfaces of the requested hemispheres are exposed, together with the files of the whole brain
    surface_keys = {
        f"{_NUMEROSITY_FOUR_PREFIXES[part]}_{kind}" for part in parts for kind in ("white", "flat", "vertex_indices")
    }
    shared_keys = {
        f"response_{task}", "validity", "surface_mapping_method2", "surface_metadata", "label_names", "roi_data",
    }
    assert set(dataset.files) == surface_keys | shared_keys


def test_load_numerosity_four_conditions_marks_vertices_without_valid_voxels(tmp_path: Path):
    """Test that a vertex whose voxels are all invalid has a response of NaN rather than zero."""
    _write_fake_numerosity_four_files(tmp_path)

    dataset = load_dataset(
        "numerosity-four-conditions", hemisphere="right", task="area", dest_dir=tmp_path, download=False,
    )

    # Vertex 3 is the second vertex of the right hemisphere and maps only to the invalid voxel
    assert np.isnan(dataset.response[1]).all()
    assert np.isfinite(np.delete(dataset.response, 1, axis=0)).all()


@pytest.mark.parametrize(("hemisphere", "other"), [("left", "rh"), ("right", "lh")])
def test_load_numerosity_four_conditions_needs_no_surface_of_the_other_hemisphere(
    tmp_path: Path,
    hemisphere: str,
    other: str,
):
    """Test that loading one hemisphere does not need the surface files of the other one."""
    _write_fake_numerosity_four_files(tmp_path)
    spec = _get_spec("numerosity-four-conditions")

    for kind in ("white", "flat", "vertex_indices"):
        (tmp_path / spec.files[f"{other}_{kind}"]).unlink()

    # With downloading disabled, fetching any of the deleted files would raise FileNotFoundError
    dataset = load_dataset(
        "numerosity-four-conditions", hemisphere=hemisphere, task="area", dest_dir=tmp_path, download=False,
    )

    assert sorted(dataset.mesh.parts) == [hemisphere]


def test_load_numerosity_four_conditions_refuses_a_mesh_that_does_not_match(tmp_path: Path):
    """Test that a white surface with a different number of vertices than the surface metadata is refused."""
    _write_fake_numerosity_four_files(tmp_path, num_left_white_vertices=4)

    with pytest.raises(ValueError, match="left hemisphere has 3 vertices in the surface metadata but 4"):
        load_dataset("numerosity-four-conditions", hemisphere="left", task="area", dest_dir=tmp_path, download=False)


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
@pytest.mark.parametrize("hemisphere", ["left", "right", "both"])
def test_load_numerosity_four_conditions(hemisphere: str):
    """Test that the four-condition dataset returns a response, regions of interest, and a mesh that line up."""
    dataset = load_dataset("numerosity-four-conditions", hemisphere=hemisphere, task="area")

    num_units = dataset.response.shape[0]

    assert dataset.roi_index.shape == (num_units,)
    assert dataset.mesh.n_vertices == num_units
    assert sorted(dataset.mesh.parts) == (["left", "right"] if hemisphere == "both" else [hemisphere])
    assert dataset.roi_mapping[-1] == "unassigned"
    assert set(np.unique(dataset.roi_index)) <= set(dataset.roi_mapping)
    assert (dataset.hemisphere, dataset.task) == (hemisphere, "area")


@pytest_skip_examples
def test_load_numerosity_four_conditions_puts_the_left_hemisphere_first():
    """Test that both hemispheres are the left and the right hemisphere concatenated in that order."""
    left = load_dataset("numerosity-four-conditions", hemisphere="left", task="area")
    right = load_dataset("numerosity-four-conditions", hemisphere="right", task="area")
    both = load_dataset("numerosity-four-conditions", hemisphere="both", task="area")

    np.testing.assert_array_equal(both.response, np.concatenate([left.response, right.response]))
    np.testing.assert_array_equal(both.roi_index, np.concatenate([left.roi_index, right.roi_index]))


@pytest_skip_examples
def test_load_numerosity_four_conditions_tasks_share_the_vertices():
    """Test that every task has a response on the same vertices but with different values."""
    responses = {
        task: load_dataset("numerosity-four-conditions", hemisphere="left", task=task).response
        for task in _NUMEROSITY_FOUR_TASKS
    }

    assert len({response.shape[0] for response in responses.values()}) == 1
    assert not np.allclose(responses["area"], responses["dens"], equal_nan=True)


@pytest_skip_examples
def test_load_numerosity_four_conditions_downloads_only_what_is_loaded(tmp_path: Path):
    """Test that loading one task of one hemisphere downloads only its surfaces and the files of the whole brain."""
    load_dataset("numerosity-four-conditions", hemisphere="left", task="circ", dest_dir=tmp_path)

    spec = _get_spec("numerosity-four-conditions")
    keys = [
        "response_circ", "validity", "surface_mapping_method2", "surface_metadata", "label_names", "roi_data",
        "lh_white", "lh_flat", "lh_vertex_indices",
    ]  # fmt: skip

    assert sorted(p.name for p in tmp_path.rglob("*") if p.is_file()) == sorted(Path(spec.files[k]).name for k in keys)

