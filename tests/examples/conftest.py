"""Test configuration for example datasets."""

import hashlib
import zipfile
from pathlib import Path
import pytest
from prfmodel.examples import _registry
from prfmodel.examples._dataset import Dataset
from prfmodel.examples._fetch import FileFetcher
from prfmodel.examples._options import Options
from prfmodel.examples._registry import DatasetSpec


def _load_fake_dataset(fetch: FileFetcher, options: Options) -> Dataset:  # noqa: ARG001
    return Dataset(name="fake-dataset", files={"first": fetch("first"), "second": fetch("second")})


@pytest.fixture
def fake_files() -> dict[str, bytes]:
    """Contents of the files in the fake dataset archive."""
    return {"first.txt": b"first file contents", "nested/second.txt": b"second file contents" * 100}


@pytest.fixture
def fake_archive(tmp_path: Path, fake_files: dict[str, bytes]) -> Path:
    """Path of a zip archive with the fake dataset files and one file that the registry does not list."""
    archive_path = tmp_path / "source" / "fake.zip"
    archive_path.parent.mkdir(parents=True)

    with zipfile.ZipFile(archive_path, "w") as archive:
        for member, content in fake_files.items():
            archive.writestr(member, content)

        archive.writestr("unlisted.txt", b"should never be extracted")

    return archive_path


@pytest.fixture
def fake_checksums(fake_files: dict[str, bytes]) -> dict[str, dict[str, object]]:
    """Checksums that the fake dataset files are expected to match."""
    return {
        member: {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
        for member, content in fake_files.items()
    }


@pytest.fixture
def fake_spec(fake_archive: Path) -> DatasetSpec:
    """Specification of a fake dataset served from a local archive."""
    return DatasetSpec(
        name="fake-dataset",
        summary="A fake dataset for testing.",
        url=fake_archive.as_uri(),
        files={"first": "first.txt", "second": "nested/second.txt"},
        loader=_load_fake_dataset,
        licence="CC BY 4.0",
        citation="Nobody. (2026). A fake dataset.",
    )


@pytest.fixture
def fake_registry(monkeypatch: pytest.MonkeyPatch, fake_spec: DatasetSpec) -> DatasetSpec:
    """Make the fake dataset the only available dataset."""
    monkeypatch.setattr(_registry, "_REGISTRY", {fake_spec.name: fake_spec})
    return fake_spec


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Path of an empty data directory."""
    return tmp_path / "data"


@pytest.fixture
def fetcher(fake_spec: DatasetSpec, data_dir: Path, fake_checksums: dict[str, dict[str, object]]) -> FileFetcher:
    """Create a file fetcher for the fake dataset."""
    return FileFetcher(fake_spec, data_dir=data_dir, checksums=fake_checksums)
