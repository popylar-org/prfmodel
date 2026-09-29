"""Test reading MATLAB tables out of MAT-files."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.io import loadmat
from scipy.io import savemat
from prfmodel.examples import read_matlab_tables

# Two tables copied out of a public ECoG dataset (OpenNeuro ds004194, CC0) with
# scripts/export_matlab_table_fixture.py. Between them they cover what MATLAB writes into a table: text
# columns, double columns, and a table whose rows describe the channels of a recording.
FIXTURE = Path(__file__).parent / "data" / "matlab_tables.mat"

EXPECTED_SHAPES = {"channels": (11, 48), "events": (224, 9)}
EXPECTED_CHANNELS = ["OT05", "OT06", "OT07", "OT08", "OT14", "OT15", "OT16", "BT15", "BT16", "sT2", "sT3"]

# The coordinates of channel OT05, as they appear in the electrodes.tsv of the source dataset.
EXPECTED_POSITION = [-53.667, -47.382, 18.258]


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    """Read the fixture once for the whole module."""
    return read_matlab_tables(FIXTURE)


def test_reads_every_table(tables: dict[str, pd.DataFrame]):
    """Test that read_matlab_tables returns one frame per table variable, keyed by variable name."""
    assert sorted(tables) == ["channels", "events"]
    assert all(isinstance(table, pd.DataFrame) for table in tables.values())


def test_shapes_match_the_tables(tables: dict[str, pd.DataFrame]):
    """Test that the frames have the rows and columns that the MATLAB tables declare."""
    assert {name: table.shape for name, table in tables.items()} == EXPECTED_SHAPES


def test_column_names_match_matlab(tables: dict[str, pd.DataFrame]):
    """Test that the frames use the variable names of the MATLAB tables, in their original order."""
    assert list(tables["events"].columns[:3]) == ["duration", "ISI", "trial_type"]
    assert list(tables["channels"].columns[:3]) == ["name", "type", "units"]


def test_reads_text_columns(tables: dict[str, pd.DataFrame]):
    """Test that MATLAB text columns become columns of str, in their original order."""
    assert tables["channels"]["name"].tolist() == EXPECTED_CHANNELS
    assert tables["events"]["trial_name"].iloc[0] == "VERTICAL-L-R-1"


def test_reads_numeric_columns(tables: dict[str, pd.DataFrame]):
    """Test that MATLAB numeric columns become float columns with their values intact."""
    channels = tables["channels"]
    assert channels["x"].dtype == np.float64

    position = channels.loc[0, ["x", "y", "z"]].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(position, EXPECTED_POSITION)


def test_keeps_rows_together(tables: dict[str, pd.DataFrame]):
    """Test that a row of a frame holds the values that belong to the same row of the MATLAB table."""
    channels = tables["channels"].set_index("name")

    assert channels.loc["OT15", "group"] == "OT"
    assert channels.loc["OT15", "wangarea"] == "TO1"
    assert channels.loc["sT2", "group"] == "sT"


def test_reads_the_same_tables_from_every_file_that_shares_them(tables: dict[str, pd.DataFrame]):
    """Test that the reader does not depend on which arrays a file happens to hold besides its tables."""
    # The fixture was copied out of a larger file, so it holds the same tables without any of its arrays.
    assert loadmat(FIXTURE, variable_names=["datats"]).get("datats") is None
    assert tables["channels"]["name"].tolist() == EXPECTED_CHANNELS


def test_returns_nothing_for_a_file_without_tables(tmp_path: Path):
    """Test that read_matlab_tables returns an empty mapping for a MAT-file that holds only arrays."""
    path = tmp_path / "arrays.mat"
    savemat(path, {"response": np.arange(6.0).reshape(2, 3)})

    assert read_matlab_tables(path) == {}


def test_rejects_a_file_that_is_not_a_mat_file(tmp_path: Path):
    """Test that read_matlab_tables raises when handed something that is not a MAT-file at all."""
    path = tmp_path / "not_a_mat_file.bin"
    path.write_bytes(b"\x00" * 256)

    with pytest.raises(ValueError, match="version 5 MAT-file"):
        read_matlab_tables(path)


def test_rejects_an_unsupported_mat_file_version(tmp_path: Path):
    """Test that read_matlab_tables raises for a MAT-file version it cannot read, such as v7.3."""
    path = tmp_path / "v73.mat"
    savemat(path, {"response": np.arange(6.0).reshape(2, 3)})

    raw = bytearray(path.read_bytes())
    raw[124:126] = (0x0200).to_bytes(2, "little")
    path.write_bytes(bytes(raw))

    with pytest.raises(ValueError, match="only version 5"):
        read_matlab_tables(path)
