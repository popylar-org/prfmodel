"""Test the specifications of the available example datasets."""

import re
from importlib.resources import files
import pytest
from prfmodel.examples import describe_dataset
from prfmodel.examples import list_datasets
from prfmodel.examples._registry import _get_registry
from prfmodel.examples._registry import load_checksums

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def test_list_datasets():
    """Test that all datasets are listed in alphabetical order."""
    assert list_datasets() == ["7t-retbar-visual", "hcp-999999-surface", "numerosity-timing"]


def test_checksums_are_packaged():
    """Test that the checksum manifest is shipped with the package."""
    assert files("prfmodel.data").joinpath("checksums.json").is_file()


def test_published_files_have_a_checksum():
    """Test that every file of a published dataset has a valid checksum entry."""
    checksums = load_checksums()

    for spec in _get_registry().values():
        for member in spec.files.values():
            assert member in checksums, f"'{member}' of '{spec.name}' has no checksum"
            assert _SHA256_PATTERN.match(checksums[member]["sha256"])
            assert checksums[member]["size"] > 0


def test_every_dataset_has_a_licence_and_citation():
    """Test that every dataset records how it may be used and how it should be cited."""
    for spec in _get_registry().values():
        assert spec.licence
        assert spec.citation
        assert spec.summary


def test_describe_dataset_reports_the_essentials():
    """Test that a description names the options, licence, and citation of a dataset."""
    description = describe_dataset("hcp-999999-surface")

    assert "hcp-999999-surface" in description
    assert "flat, inflated, pia, wm" in description
    assert "CC BY 4.0" in description
    assert "Van Essen" in description


def test_describe_dataset_reports_per_file_downloads():
    """Test that a dataset served as individual files says only what is loaded is downloaded."""
    assert "only the files that are loaded" in describe_dataset("numerosity-timing")


def test_every_dataset_has_exactly_one_source():
    """Test that each dataset is served either as one archive or as individual files."""
    for spec in _get_registry().values():
        assert spec.is_archive != bool(spec.file_urls)

        if spec.file_urls:
            assert set(spec.file_urls) == set(spec.files)


def test_describe_dataset_unknown_name():
    """Test that describing an unknown dataset lists the valid names."""
    with pytest.raises(ValueError, match="7t-retbar-visual"):
        describe_dataset("does-not-exist")
