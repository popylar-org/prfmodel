"""Test downloading, caching, and verifying example dataset files."""

import sys
import zipfile
from pathlib import Path
import pytest
from prfmodel.examples import ChecksumError
from prfmodel.examples import get_data_dir
from prfmodel.examples._fetch import FileFetcher
from prfmodel.examples._fetch import _download_file
from prfmodel.examples._registry import DatasetSpec

_ENV_VARS = ("PRFMODEL_DATA_DIR", "XDG_CACHE_HOME", "LOCALAPPDATA")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every environment variable that the data directory is resolved from."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_fetch_extracts_and_returns_path(fetcher: FileFetcher, fake_files: dict[str, bytes]):
    """Test that fetching a file extracts it from the archive and returns its path."""
    path = fetcher("second")

    assert path.exists()
    assert path.read_bytes() == fake_files["nested/second.txt"]
    assert fetcher.files == {"second": path}


def test_fetch_extracts_only_listed_members(fetcher: FileFetcher, data_dir: Path):
    """Test that archive members that the registry does not list are not extracted."""
    fetcher("first")

    assert not (data_dir / "unlisted.txt").exists()


def test_fetch_removes_the_archive(fetcher: FileFetcher, data_dir: Path):
    """Test that the downloaded archive is deleted once its members have been extracted."""
    fetcher("first")

    assert list(data_dir.glob("*.zip")) == []


def test_fetch_does_not_redownload_cached_files(
    fetcher: FileFetcher,
    monkeypatch: pytest.MonkeyPatch,
    fake_files: dict[str, bytes],
):
    """Test that a dataset whose files are all present is not downloaded again."""
    fetcher("first")

    def fail_download(*args: object, **kwargs: object) -> None:
        pytest.fail("A cached dataset must not be downloaded again")

    monkeypatch.setattr("prfmodel.examples._fetch._download_file", fail_download)

    assert fetcher("second").read_bytes() == fake_files["nested/second.txt"]


def test_fetch_checksum_mismatch_raises_and_leaves_no_file(
    fake_spec: DatasetSpec,
    data_dir: Path,
    fake_checksums: dict[str, dict[str, object]],
):
    """Test that a file that does not match its checksum is reported and not kept."""
    corrupted = dict(fake_checksums)
    corrupted["first.txt"] = {"sha256": "0" * 64, "size": 1}

    fetcher = FileFetcher(fake_spec, data_dir=data_dir, checksums=corrupted)

    with pytest.raises(ChecksumError, match=r"first\.txt"):
        fetcher("first")

    assert not (data_dir / "first.txt").exists()
    assert list(data_dir.glob("*.part")) == []


def test_fetch_without_download_raises(
    fake_spec: DatasetSpec,
    data_dir: Path,
    fake_checksums: dict[str, dict[str, object]],
):
    """Test that a missing file raises when downloading is disabled."""
    fetcher = FileFetcher(fake_spec, data_dir=data_dir, checksums=fake_checksums, download=False)

    with pytest.raises(FileNotFoundError, match=r"first\.txt"):
        fetcher("first")


def test_fetch_rejects_member_outside_data_dir(
    tmp_path: Path,
    data_dir: Path,
    fake_checksums: dict[str, dict[str, object]],
):
    """Test that an archive member that would escape the data directory is refused."""
    archive_path = tmp_path / "traversal.zip"

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escaped.txt", b"escaped")

    spec = DatasetSpec(
        name="traversal",
        summary="",
        url=archive_path.as_uri(),
        files={"escaped": "../escaped.txt"},
        loader=lambda fetch, options: None,  # type: ignore[arg-type,return-value]
    )
    fetcher = FileFetcher(spec, data_dir=data_dir, checksums=fake_checksums)

    with pytest.raises(ValueError, match="outside"):
        fetcher("escaped")

    assert not (tmp_path / "escaped.txt").exists()


def test_download_rejects_disallowed_scheme(tmp_path: Path):
    """Test that a URL with a scheme that is not allowed is refused."""
    with pytest.raises(ValueError, match="must start with"):
        _download_file("ftp://example.com/file.zip", tmp_path / "file.zip", "file.zip")


def test_get_data_dir_prefers_the_argument(monkeypatch: pytest.MonkeyPatch):
    """Test that an explicit directory takes precedence over the environment."""
    monkeypatch.setenv("PRFMODEL_DATA_DIR", "from_env")

    assert get_data_dir("from_argument") == Path("from_argument")


def test_get_data_dir_uses_the_environment(clean_env: None, monkeypatch: pytest.MonkeyPatch):
    """Test that PRFMODEL_DATA_DIR is used verbatim when no directory is given."""
    monkeypatch.setenv("PRFMODEL_DATA_DIR", "from_env")

    assert get_data_dir() == Path("from_env")


def test_get_data_dir_uses_xdg_cache_home(clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test that an absolute XDG_CACHE_HOME is used when PRFMODEL_DATA_DIR is not set."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    assert get_data_dir() == tmp_path / "prfmodel" / "data"


def test_get_data_dir_ignores_relative_xdg_cache_home(clean_env: None, monkeypatch: pytest.MonkeyPatch):
    """Test that a relative XDG_CACHE_HOME is invalid and falls through to the platform default."""
    monkeypatch.setenv("XDG_CACHE_HOME", "relative_cache")

    assert get_data_dir() != Path("relative_cache") / "prfmodel" / "data"


@pytest.mark.skipif(sys.platform not in ("linux", "darwin"), reason="platform-specific default")
def test_get_data_dir_platform_default(clean_env: None):
    """Test that the platform default is used when nothing is set."""
    expected = (
        Path.home() / "Library" / "Caches" / "prfmodel" / "data"
        if sys.platform == "darwin"
        else Path.home() / ".cache" / "prfmodel" / "data"
    )

    assert get_data_dir() == expected


def test_get_data_dir_does_not_create_the_directory(clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test that resolving the data directory has no side effects."""
    monkeypatch.setenv("PRFMODEL_DATA_DIR", str(tmp_path / "missing"))

    assert not get_data_dir().exists()


def test_fetch_file_downloads_only_what_is_asked_for(file_fetcher: FileFetcher, data_dir: Path):
    """Test that a dataset served as individual files downloads only the file that is requested."""
    file_fetcher("first")

    assert (data_dir / "first.txt").exists()
    assert not (data_dir / "nested" / "second.txt").exists()


def test_fetch_file_verifies_the_checksum(
    fake_file_spec: DatasetSpec,
    data_dir: Path,
    fake_checksums: dict[str, dict[str, object]],
):
    """Test that a directly downloaded file that does not match its checksum is reported and not kept."""
    corrupted = dict(fake_checksums)
    corrupted["first.txt"] = {"sha256": "0" * 64, "size": 1}

    fetcher = FileFetcher(fake_file_spec, data_dir=data_dir, checksums=corrupted)

    with pytest.raises(ChecksumError, match=r"first\.txt"):
        fetcher("first")

    assert not (data_dir / "first.txt").exists()
    assert list(data_dir.glob("*.part")) == []


def test_fetch_file_does_not_redownload_cached_files(
    file_fetcher: FileFetcher,
    monkeypatch: pytest.MonkeyPatch,
    fake_files: dict[str, bytes],
):
    """Test that a file that is already present is not downloaded again."""
    file_fetcher("first")

    def fail_download(*args: object, **kwargs: object) -> None:
        pytest.fail("A cached file must not be downloaded again")

    monkeypatch.setattr("prfmodel.examples._fetch._download_file", fail_download)

    assert file_fetcher("first").read_bytes() == fake_files["first.txt"]


def test_fetch_file_without_download_raises(
    fake_file_spec: DatasetSpec,
    data_dir: Path,
    fake_checksums: dict[str, dict[str, object]],
):
    """Test that a missing file raises when downloading is disabled."""
    fetcher = FileFetcher(fake_file_spec, data_dir=data_dir, checksums=fake_checksums, download=False)

    with pytest.raises(FileNotFoundError, match=r"first\.txt"):
        fetcher("first")


def test_spec_requires_exactly_one_source(fake_archive: Path):
    """Test that a dataset specification must set either an archive URL or per-file URLs."""
    with pytest.raises(ValueError, match="either 'url' or 'file_urls'"):
        DatasetSpec(name="neither", summary="", files={}, loader=lambda fetch, options: None)

    with pytest.raises(ValueError, match="either 'url' or 'file_urls'"):
        DatasetSpec(
            name="both",
            summary="",
            files={"a": "a.txt"},
            loader=lambda fetch, options: None,
            url=fake_archive.as_uri(),
            file_urls={"a": fake_archive.as_uri()},
        )


def test_spec_requires_a_url_for_every_file(fake_archive: Path):
    """Test that a dataset served as individual files must give a URL for each of them."""
    with pytest.raises(ValueError, match="URL for every file"):
        DatasetSpec(
            name="incomplete",
            summary="",
            files={"a": "a.txt", "b": "b.txt"},
            loader=lambda fetch, options: None,
            file_urls={"a": fake_archive.as_uri()},
        )
