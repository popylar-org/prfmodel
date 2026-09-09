"""Regenerate the checksum manifest of the example datasets.

Downloads every published dataset archive, hashes the files that the registry lists, and writes them to
``src/prfmodel/data/checksums.json``. Run this whenever a dataset is added to the registry or one of the
archives it points at is republished.

Usage
-----
    python scripts/build_checksums.py
    python scripts/build_checksums.py --check
"""

import argparse
import hashlib
import json
import logging
import sys
import tempfile
import zipfile
from pathlib import Path
from prfmodel.examples._fetch import _download_file
from prfmodel.examples._registry import _get_registry

logger = logging.getLogger(__name__)

_MANIFEST_PATH = Path(__file__).parent.parent / "src" / "prfmodel" / "data" / "checksums.json"
_CHUNK_SIZE = 1024 * 1024


def build_manifest() -> dict[str, object]:
    """Download every published archive and hash the files that the registry lists."""
    entries: dict[str, dict[str, object]] = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        for spec in _get_registry().values():
            if not spec.is_published:
                logger.warning("Skipping unpublished dataset '%s'", spec.name)
                continue

            archive_path = Path(temp_dir) / f"{spec.name}.zip"
            _download_file(spec.url, archive_path, f"{spec.name}.zip")

            with zipfile.ZipFile(archive_path) as archive:
                for member in spec.files.values():
                    digest = hashlib.sha256()
                    size = 0

                    with archive.open(member) as stream:
                        for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
                            digest.update(chunk)
                            size += len(chunk)

                    entries[member] = {"sha256": digest.hexdigest(), "size": size}

            archive_path.unlink()

    return {"version": 1, "files": dict(sorted(entries.items()))}


def main() -> int:
    """Write the manifest, or compare it against the one on disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the manifest on disk is out of date")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    manifest = build_manifest()
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
