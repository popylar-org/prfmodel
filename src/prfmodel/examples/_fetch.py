"""Download, cache, and verify example dataset files."""

from __future__ import annotations
import hashlib
import logging
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from prfmodel.examples._registry import DatasetSpec

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = ("http:", "https:", "file:")
_CHUNK_SIZE = 1024 * 1024
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = 1.0


class ChecksumError(OSError):
    """
    Exception raised when a file does not match its expected checksum.

    Parameters
    ----------
    path : pathlib.Path
        Path of the file that was checked.
    expected : str
        Expected SHA-256 hex digest.
    actual : str
        SHA-256 hex digest that was computed.

    """

    def __init__(self, path: Path, expected: str, actual: str):
        super().__init__(
            f"File '{path}' has SHA-256 {actual} but {expected} was expected. The download may be corrupted "
            f"or the data source may have changed. Delete the file and retry. If the error persists, please "
            f"report it.",
        )


def get_data_dir(dest_dir: str | os.PathLike | None = None) -> Path:
    r"""
    Resolve the directory in which example datasets are stored.

    Parameters
    ----------
    dest_dir : str or os.PathLike, optional
        Directory to use. If given, it is used as-is and takes precedence over the environment.

    Returns
    -------
    pathlib.Path
        The resolved directory. The directory is not created.

    Notes
    -----
    The directory is resolved in the following order:

    1. The ``dest_dir`` argument, when given.
    2. The ``PRFMODEL_DATA_DIR`` environment variable, when set and non-empty.
    3. ``$XDG_CACHE_HOME/prfmodel/data``, when ``XDG_CACHE_HOME`` is set to an absolute path.
    4. A platform default: ``%LOCALAPPDATA%\\prfmodel\\Cache\\data`` on Windows,
       ``~/Library/Caches/prfmodel/data`` on macOS, and ``~/.cache/prfmodel/data`` elsewhere.

    Examples
    --------
    Resolve the directory that is used when none is requested.

    >>> from prfmodel.examples import get_data_dir
    >>> get_data_dir()  # doctest: +SKIP
    PosixPath('/home/user/.cache/prfmodel/data')

    Pass a directory to use it as-is, whatever the environment says.

    >>> get_data_dir("my_data")  # doctest: +SKIP
    PosixPath('my_data')

    """
    if dest_dir is not None:
        return Path(dest_dir)

    env_dir = os.environ.get("PRFMODEL_DATA_DIR")

    if env_dir:
        return Path(env_dir)

    xdg_dir = os.environ.get("XDG_CACHE_HOME")

    # A relative XDG_CACHE_HOME is invalid according to the specification, so we fall through to the default
    if xdg_dir and Path(xdg_dir).is_absolute():
        return Path(xdg_dir) / "prfmodel" / "data"

    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        base_dir = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        return base_dir / "prfmodel" / "Cache" / "data"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "prfmodel" / "data"

    return Path.home() / ".cache" / "prfmodel" / "data"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _check_url(url: str) -> None:
    if not url.startswith(_ALLOWED_SCHEMES):
        msg = f"File URL must start with one of {_ALLOWED_SCHEMES} but was '{url}'"
        raise ValueError(msg)


def _open_url(url: str, description: str):  # noqa: ANN202 (returns a private urllib response type)
    last_error: urllib.error.URLError | TimeoutError | None = None

    for attempt in range(_MAX_ATTEMPTS):
        try:
            # We audit the URL scheme in _check_url but ruff still flags the call
            return urllib.request.urlopen(url)  # noqa: S310
        except urllib.error.HTTPError as error:  # noqa: PERF203 (a retry loop must catch per attempt)
            msg = (
                f"Could not download {description} from '{url}' ({error.code} {error.reason}). The file "
                f"identifier may have changed; please report this."
            )
            raise OSError(msg) from error
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt < _MAX_ATTEMPTS - 1:
                delay = _BACKOFF_SECONDS * 2**attempt
                logger.warning("Download of %s failed (%s); retrying in %.0f s", description, error, delay)
                time.sleep(delay)

    msg = f"Could not download {description} from '{url}' after {_MAX_ATTEMPTS} attempts."
    raise OSError(msg) from last_error


def _download_file(url: str, dest_path: Path, description: str, expected_sha256: str | None = None) -> None:
    """Download a single file, writing it atomically so a failure leaves nothing behind."""
    _check_url(url)

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # The temporary file lives in the destination directory so that os.replace stays on one filesystem
    handle, temp_name = tempfile.mkstemp(dir=dest_path.parent, prefix=f"{dest_path.name}.", suffix=".part")
    temp_path = Path(temp_name)

    try:
        with _open_url(url, description) as response, os.fdopen(handle, "wb") as out_file:
            total = int(response.headers.get("Content-Length") or 0)

            with tqdm(total=total or None, unit="B", unit_scale=True, desc=description) as progress:
                for chunk in iter(lambda: response.read(_CHUNK_SIZE), b""):
                    out_file.write(chunk)
                    progress.update(len(chunk))

        if expected_sha256 is not None:
            actual = _sha256(temp_path)

            if actual != expected_sha256:
                raise ChecksumError(dest_path, expected_sha256, actual)

        os.replace(temp_path, dest_path)  # noqa: PTH105 (pathlib has no atomic replace)
    finally:
        temp_path.unlink(missing_ok=True)


def _extract_member(archive: zipfile.ZipFile, member: str, data_dir: Path, expected_sha256: str | None) -> None:
    """Extract a single archive member atomically, refusing members that escape the data directory."""
    dest_path = data_dir / member

    if not dest_path.resolve().is_relative_to(data_dir.resolve()):
        msg = f"Archive member '{member}' would be extracted outside of '{data_dir}'"
        raise ValueError(msg)

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    handle, temp_name = tempfile.mkstemp(dir=dest_path.parent, prefix=f"{dest_path.name}.", suffix=".part")
    temp_path = Path(temp_name)

    try:
        with archive.open(member) as source, os.fdopen(handle, "wb") as out_file:
            for chunk in iter(lambda: source.read(_CHUNK_SIZE), b""):
                out_file.write(chunk)

        if expected_sha256 is not None:
            actual = _sha256(temp_path)

            if actual != expected_sha256:
                raise ChecksumError(dest_path, expected_sha256, actual)

        os.replace(temp_path, dest_path)  # noqa: PTH105 (pathlib has no atomic replace)
    finally:
        temp_path.unlink(missing_ok=True)


class FileFetcher:
    """
    Serve the files of a dataset from a local directory, downloading them once when they are missing.

    A fetcher is called with the logical name of a file and returns the path it was cached at. It records
    every name it served, which is what :attr:`prfmodel.examples.Dataset.files` is built from.

    Parameters
    ----------
    spec : prfmodel.examples._registry.DatasetSpec
        Specification of the dataset whose files should be served.
    data_dir : pathlib.Path
        Directory in which the dataset files are cached.
    checksums : dict of str to dict
        Expected checksums, keyed by the path of a file relative to `data_dir`.
    download : bool, default=True
        Whether missing files may be downloaded. If `False`, a missing file raises `FileNotFoundError`.

    """

    def __init__(
        self,
        spec: DatasetSpec,
        data_dir: Path,
        checksums: dict[str, dict[str, object]],
        download: bool = True,
    ):
        self._spec = spec
        self._data_dir = data_dir
        self._checksums = checksums
        self._download = download
        self._files: dict[str, Path] = {}

    @property
    def keys(self) -> tuple[str, ...]:
        """Logical names of all files that the dataset holds."""
        return tuple(self._spec.files)

    @property
    def files(self) -> dict[str, Path]:
        """Paths of the files that have been served so far, keyed by their logical name."""
        return dict(self._files)

    def path_for(self, key: str) -> Path:
        """
        Return the path a file is cached at, without fetching it.

        Parameters
        ----------
        key : str
            Logical name of the file.

        Returns
        -------
        pathlib.Path
            The path of the file inside the data directory.

        """
        return self._data_dir / self._spec.files[key]

    def __call__(self, key: str) -> Path:
        """Return the path of a file, downloading it or the dataset archive first when it is missing."""
        path = self.path_for(key)

        if not path.exists():
            if self._spec.url is None:
                self._fetch_file(key)
            else:
                self._fetch_archive(self._spec.url)

        if not path.exists():
            msg = (
                f"File '{key}' of dataset '{self._spec.name}' is missing at '{path}' and could not be "
                f"obtained from the dataset source."
            )
            raise FileNotFoundError(msg)

        self._files[key] = path

        return path

    def _expected_sha256(self, member: str) -> str | None:
        return self._checksums.get(member, {}).get("sha256")  # type: ignore[return-value]

    def _fetch_file(self, key: str) -> None:
        member = self._spec.files[key]

        if not self._download:
            msg = (
                f"File '{key}' of dataset '{self._spec.name}' is not available at "
                f"'{self._data_dir / member}' and downloading is disabled."
            )
            raise FileNotFoundError(msg)

        _download_file(
            self._spec.file_urls[key],
            self._data_dir / member,
            member,
            self._expected_sha256(member),
        )

    def _fetch_archive(self, url: str) -> None:
        missing = [member for member in self._spec.files.values() if not (self._data_dir / member).exists()]

        if not missing:
            return

        if not self._download:
            msg = (
                f"Dataset '{self._spec.name}' is not available at '{self._data_dir}' and downloading is "
                f"disabled. Missing file: '{self._data_dir / missing[0]}'."
            )
            raise FileNotFoundError(msg)

        self._data_dir.mkdir(parents=True, exist_ok=True)
        archive_path = self._data_dir / f"{self._spec.name}.zip"

        try:
            _download_file(url, archive_path, f"{self._spec.name}.zip")

            with zipfile.ZipFile(archive_path) as archive:
                for member in missing:
                    _extract_member(archive, member, self._data_dir, self._expected_sha256(member))
        finally:
            # The archive is several times the size of what we extract and nothing reads it again
            archive_path.unlink(missing_ok=True)
