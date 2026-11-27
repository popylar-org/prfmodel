"""Regenerate the checksum manifest of the example datasets.

Hashes the files that the registry lists and writes them to ``src/prfmodel/data/checksums.json``. Run this
whenever a dataset is added to the registry or one of its files is republished.

By default, files from the example-data cache are reused and only missing files are downloaded. Pass
``--dataset`` to recompute only the entries of the named datasets and keep the other entries as they are.
Pass ``--data-dir`` to use a different cache directory, or ``--fresh`` to download fresh copies into a
temporary directory instead of reusing the cache. Files in the cache are hashed as they are, so use
``--fresh`` when verifying the published files.

Usage
-----
    python scripts/build_checksums.py
    python scripts/build_checksums.py --check
    python scripts/build_checksums.py --dataset numerosity-four-conditions
    python scripts/build_checksums.py --check --dataset numerosity-four-conditions
    python scripts/build_checksums.py --check --fresh
"""

import argparse
import hashlib
import json
import logging
import sys
import tempfile
from pathlib import Path
from prfmodel.examples._fetch import FileFetcher
from prfmodel.examples._fetch import get_data_dir
from prfmodel.examples._registry import DatasetSpec
from prfmodel.examples._registry import _get_registry

logger = logging.getLogger(__name__)

_MANIFEST_PATH = Path(__file__).parent.parent / "src" / "prfmodel" / "data" / "checksums.json"
_CHUNK_SIZE = 1024 * 1024


def _hash_stream(stream: object) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0

    for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):  # type: ignore[attr-defined]
        digest.update(chunk)
        size += len(chunk)

    return digest.hexdigest(), size


def _hash_spec(spec: DatasetSpec, data_dir: Path) -> dict[str, dict[str, object]]:
    """Hash the files of a dataset, reusing cached files and downloading only what is missing."""
    fetcher = FileFetcher(spec, data_dir=data_dir, checksums={})
    entries = {}

    for key, member in spec.files.items():
        with fetcher(key).open("rb") as stream:
            sha256, size = _hash_stream(stream)

        entries[member] = {"sha256": sha256, "size": size}

    return entries


def build_manifest(data_dir: Path, datasets: list[str] | None = None) -> dict[str, object]:
    """
    Hash the files that the registry lists and build the manifest.

    Parameters
    ----------
    data_dir : pathlib.Path
        Directory in which dataset files are cached. Missing files are downloaded into this directory.
    datasets : list of str, optional
        Names of the datasets to hash. When given, only the entries of these datasets are recomputed and
        the entries of all other datasets are kept as they are. Entries of files that no dataset publishes
        any more are removed.

    """
    registry = _get_registry()
    selected = list(registry.values()) if datasets is None else [registry[name] for name in dict.fromkeys(datasets)]

    entries: dict[str, dict[str, object]] = {}

    for spec in selected:
        logger.info("Hashing dataset '%s'", spec.name)
        entries.update(_hash_spec(spec, data_dir))

    if datasets is None:
        files = entries
    else:
        # Keep the entries of datasets that were not selected and replace the selected ones
        files = dict(json.loads(_MANIFEST_PATH.read_text())["files"])
        files.update(entries)

    # Drop files that the registry no longer publishes, such as a dataset file that was replaced
    published = {member for spec in registry.values() for member in spec.files.values()}
    files = {member: value for member, value in files.items() if member in published}

    return {"version": 1, "files": dict(sorted(files.items()))}


def main() -> int:
    """Write the manifest, or compare it against the one on disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the manifest on disk is out of date")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(_get_registry()),
        metavar="NAME",
        help="only recompute this dataset; repeat to select several",
    )
    data_source = parser.add_mutually_exclusive_group()
    data_source.add_argument(
        "--data-dir",
        type=Path,
        help="directory with cached files; defaults to the example-data cache",
    )
    data_source.add_argument(
        "--fresh",
        action="store_true",
        help="download fresh copies into a temporary directory instead of reusing the cache",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.fresh:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = build_manifest(Path(temp_dir), args.dataset)
    else:
        data_dir = args.data_dir.expanduser() if args.data_dir is not None else get_data_dir()
        manifest = build_manifest(data_dir, args.dataset)

    content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"

    if args.check:
        if _MANIFEST_PATH.read_text() != content:
            logger.error("'%s' is out of date; rerun this script without --check", _MANIFEST_PATH)
            return 1

        logger.info("'%s' is up to date", _MANIFEST_PATH)
        return 0

    _MANIFEST_PATH.write_text(content)
    logger.info("Wrote %d checksums to '%s'", len(manifest["files"]), _MANIFEST_PATH)  # type: ignore[arg-type]

    return 0


if __name__ == "__main__":
    sys.exit(main())
