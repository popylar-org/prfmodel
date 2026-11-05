"""Export the files of the '7t-ncsf-visual' example dataset from the prfmodel workshop data.

Reads the ``data`` folder of the prfmodel workshop (subject ``sub-16112020``) and writes:

``<output>/``
    The files that are published on figshare: the GIFTI files of the subject, copied byte for byte, and
    ``task-prf_stimulus.npz`` with the pRF ``design`` (frames first, row 0 at the top of the screen) and its
    coordinate ``grid`` in degrees of visual angle, with the vertical coordinate first.
``src/prfmodel/data/stimuli/csf_stimulus.npz``
    The spatial frequency (``sf``, cycles per degree) and Michelson contrast (``contrast``, percent) of each frame
    of the CSF task, which ships with the package. Blank frames are stored as they were recorded, with both zero.

The noise ceiling of the workshop is not exported because how it was computed is not documented and it cannot be
reproduced from the odd and even splits. The surfaces are not exported because they are identical to those of the
'hcp-999999-surface' dataset.

Usage
-----
    python scripts/export_ncsf_figshare.py <workshop data dir> <output dir>
"""

import argparse
import hashlib
import logging
import math
import shutil
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

_CSF_STIMULUS_PATH = Path(__file__).parent.parent / "src" / "prfmodel" / "data" / "stimuli" / "csf_stimulus.npz"

_SUBJECT = "sub-16112020"
_SPACE = "space-fsLR_den-59k"

# The pRF design spans the whole screen of 39.3 cm, viewed from 220 cm
_SCREEN_SIZE_CM = 39.3
_SCREEN_DISTANCE_CM = 220.0


def _gifti_names() -> list[str]:
    names = []

    for hemi in ("L", "R"):
        names += [
            f"{_SUBJECT}_task-csf_hemi-{hemi}_{_SPACE}_desc-detrend3_bold.func.gii",
            f"{_SUBJECT}_task-csf_split-odd_hemi-{hemi}_{_SPACE}_desc-detrend3_bold.func.gii",
            f"{_SUBJECT}_task-csf_split-even_hemi-{hemi}_{_SPACE}_desc-detrend3_bold.func.gii",
            f"{_SUBJECT}_task-prf_hemi-{hemi}_{_SPACE}_desc-detrend3_bold.func.gii",
            *[f"{_SUBJECT}_hemi-{hemi}_{_SPACE}_atlas-benson14_{key}.func.gii" for key in ("angle", "eccen", "sigma")],
            f"{_SUBJECT}_hemi-{hemi}_{_SPACE}_atlas-benson14_dseg.label.gii",
        ]

    return names


# Coordinate grid of the pRF design, in degrees, with the vertical coordinate first. The outer pixels lie on the edges
# of the screen rather than half a pixel inside them, as in the workshop and in prfpy, so that fits are comparable
def _prf_grid(num_y: int, num_x: int) -> np.ndarray:
    radius = math.degrees(math.atan(_SCREEN_SIZE_CM / 2.0 / _SCREEN_DISTANCE_CM))

    # Row 0 is the top of the screen, so the vertical coordinate decreases down the rows
    y, x = np.meshgrid(np.linspace(radius, -radius, num_y), np.linspace(-radius, radius, num_x), indexing="ij")

    return np.stack((y, x), axis=-1)


def _log_file(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    logger.info("%s  %12d  %s", digest, path.stat().st_size, path.name)


def export(data_dir: Path, output_dir: Path) -> None:
    """Write the figshare files and the packaged CSF stimulus."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in _gifti_names():
        shutil.copyfile(data_dir / _SUBJECT / name, output_dir / name)
        _log_file(output_dir / name)

    stimulus_dir = data_dir / "stimuli"

    # The design is stored as (num_y, num_x, num_frames) but prfmodel expects the frames first
    design = np.moveaxis(np.load(stimulus_dir / "prf_design_matrix.npy"), -1, 0)
    prf_path = output_dir / "task-prf_stimulus.npz"
    np.savez_compressed(prf_path, design=design, grid=_prf_grid(*design.shape[1:]))
    _log_file(prf_path)

    np.savez_compressed(
        _CSF_STIMULUS_PATH,
        sf=np.load(stimulus_dir / "csf_sf_sequence.npy"),
        contrast=np.load(stimulus_dir / "csf_contrast_sequence.npy"),
    )
    _log_file(_CSF_STIMULUS_PATH)


def main() -> None:
    """Parse the command line and export the files."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path, help="the 'data' folder of the prfmodel workshop")
    parser.add_argument("output", type=Path, help="folder to write the figshare files to")
    args = parser.parse_args()

    export(args.data, args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
