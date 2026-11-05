"""Test downloading, caching, and verifying example dataset files."""

import http.server
import sys
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path
import pytest
from prfmodel.examples import ChecksumError
from prfmodel.examples import get_data_dir
from prfmodel.examples._fetch import _MAX_ATTEMPTS
from prfmodel.examples._fetch import FileFetcher
from prfmodel.examples._fetch import _download_file
from prfmodel.examples._registry import DatasetSpec

_ENV_VARS = ("PRFMODEL_DATA_DIR", "XDG_CACHE_HOME", "LOCALAPPDATA")
_PAYLOAD = b"payload" * 1000


class _StallingServer(http.server.ThreadingHTTPServer):
    """Serve a payload, stalling halfway through the body for the first `stalls` requests."""

    def __init__(self, stalls: int):
        super().__init__(("127.0.0.1", 0), _StallingHandler)
        self.stalls = stalls
        self.requests = 0
        self.release = threading.Event()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/file.bin"


class _StallingHandler(http.server.BaseHTTPRequestHandler):
    server: _StallingServer

    def do_GET(self) -> None:
        self.server.requests += 1

        self.send_response(200)
        self.send_header("Content-Length", str(len(_PAYLOAD)))
        self.end_headers()

        if self.server.requests <= self.server.stalls:
            self.wfile.write(_PAYLOAD[: len(_PAYLOAD) // 2])
            self.wfile.flush()
            # Keep the connection open without sending the rest until the test is over
            self.server.release.wait()
        else:
            self.wfile.write(_PAYLOAD)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 (name is set by the base class)
        pass


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every environment variable that the data directory is resolved from."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make a stalled download time out and be retried within a fraction of a second."""
    monkeypatch.setattr("prfmodel.examples._fetch._TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr("prfmodel.examples._fetch._BACKOFF_SECONDS", 0.0)


@pytest.fixture
def stalling_server(request: pytest.FixtureRequest) -> Iterator[_StallingServer]:
    """Run a server that stalls for as many requests as the test parametrizes, then serves the payload."""
    server = _StallingServer(stalls=request.param)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield server

    server.release.set()
    server.shutdown()
    server.server_close()
    thread.join()


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


@pytest.mark.parametrize("stalling_server", [1], indirect=True)
def test_download_retries_a_stalled_transfer(fast_retries: None, stalling_server: _StallingServer, tmp_path: Path):
    """Test that a connection that stalls mid-transfer times out and the download starts over."""
    dest_path = tmp_path / "file.bin"

    _download_file(stalling_server.url, dest_path, "file.bin")

    assert stalling_server.requests == 2  # noqa: PLR2004
    # The retry must overwrite the partial data of the stalled attempt, not append to it
    assert dest_path.read_bytes() == _PAYLOAD
    assert list(tmp_path.glob("*.part")) == []


@pytest.mark.parametrize("stalling_server", [_MAX_ATTEMPTS], indirect=True)
def test_download_gives_up_after_repeated_stalls(fast_retries: None, stalling_server: _StallingServer, tmp_path: Path):
    """Test that a download that keeps stalling raises after the last attempt and leaves nothing behind."""
    dest_path = tmp_path / "file.bin"

    with pytest.raises(OSError, match=f"after {_MAX_ATTEMPTS} attempts"):
        _download_file(stalling_server.url, dest_path, "file.bin")

    assert stalling_server.requests == _MAX_ATTEMPTS
    assert not dest_path.exists()
    assert list(tmp_path.glob("*.part")) == []


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
