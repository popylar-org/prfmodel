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

pytest_skip_examples = pytest.mark.examples


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
