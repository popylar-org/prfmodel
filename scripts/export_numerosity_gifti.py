"""Export the odd/even numerosity time series of one subject to GIFTI.

Writes three files per hemisphere into ``data/numerosity-timing-fmri/gifti``:

``sub-<S>_hemi-<L|R>_desc-odd_bold.func.gii``
    Odd-run average, one ``DataArray`` per timepoint (``NIFTI_INTENT_TIME_SERIES``).
``sub-<S>_hemi-<L|R>_desc-even_bold.func.gii``
    Even-run average, same layout.
``sub-<S>_hemi-<L|R>_desc-numerosity_dseg.label.gii``
    Integer ROI index per vertex (``NIFTI_INTENT_LABEL``), zero-based into
    ``MAP_NAMES``, with a label table naming each index.

All three share one row ordering: the numerosity ROIs concatenated in ``MAP_NAMES``
order. Only the numerosity maps are exported -- the timing maps overlap them (e.g.
NPCS and TPCS share 727 gray nodes in S1 Left), so a single label per vertex would
be ambiguous if both families were included.

This script records how the published 'numerosity-timing' dataset was produced; it is not
part of the package and is not run by it. It needs ``h5py``, which is deliberately not a
prfmodel dependency, so install it separately to run this script.

Note that row *i* is the *i*-th ROI gray node, not vertex *i* of a surface mesh.
Rendering these on a surface needs the subject's own mrVista gray-node-to-surface
mapping, which is not distributed with this dataset.
"""

import argparse
import logging
import pathlib
import matplotlib as mpl
import nibabel as nib
import numpy as np
from h5py import File

logger = logging.getLogger(__name__)

MAP_NAMES = ["NTO", "NLO", "NPO", "NPCI", "NPCM", "NPCS", "NFI", "NFS"]
HEMISPHERES = {"Left": "L", "Right": "R"}
HALVES = {"odd": "NumerosityAllOddL1", "even": "NumerosityAllEvenL1"}
ANATOMICAL_STRUCTURE = {"Left": "CortexLeft", "Right": "CortexRight"}
TR = 2.1


def collect_hemisphere(
    subject: "File",
    hemisphere: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Concatenate the numerosity ROIs of one hemisphere into flat arrays.

    Parameters
    ----------
    subject : h5py.Group
        The ``time_series/<subject>`` group of ``time_series.mat``.
    hemisphere : str
        Either ``"Left"`` or ``"Right"``.

    Returns
    -------
    tuple of (dict of str to numpy.ndarray, numpy.ndarray)
        The per-half ``(n_vertices, n_timepoints)`` time series, and the integer
        ROI label of each row, indexing ``MAP_NAMES`` from zero.
    """
    per_half: dict[str, list[np.ndarray]] = {half: [] for half in HALVES}
    labels: list[np.ndarray] = []

    for index, map_name in enumerate(MAP_NAMES):
        if map_name not in subject or hemisphere not in subject[map_name]:
            logger.warning("skipping missing ROI: %s %s", map_name, hemisphere)
            continue
        roi = subject[map_name][hemisphere]

        reference = None
        for half, key in HALVES.items():
            per_half[half].append(np.asarray(roi[key]["Scan1"], dtype=np.float32))
            # Both halves index the same gray nodes; bail out rather than silently
            # interleaving two different vertex orderings.
            current = np.asarray(roi[key]["iCoords"], dtype=np.int64).ravel()
            if reference is None:
                reference = current
            elif not np.array_equal(reference, current):
                msg = f"iCoords differ between halves for {map_name} {hemisphere}"
                raise ValueError(msg)

        labels.append(np.full(reference.size, index, dtype=np.int32))

    return (
        {half: np.concatenate(arrays, axis=0) for half, arrays in per_half.items()},
        np.concatenate(labels),
    )


def make_functional_image(
    time_series: np.ndarray,
    hemisphere: str,
    subject: str,
    half: str,
) -> nib.gifti.GiftiImage:
    """Wrap a ``(n_vertices, n_timepoints)`` array as a GIFTI time series.

    Parameters
    ----------
    time_series : numpy.ndarray
        The signal to store, vertices along the first axis.
    hemisphere : str
        Either ``"Left"`` or ``"Right"``.
    subject : str
        Subject identifier, stored in the file metadata.
    half : str
        Either ``"odd"`` or ``"even"``, stored in the file metadata.

    Returns
    -------
    nibabel.gifti.GiftiImage
        One ``DataArray`` per timepoint.
    """
    meta = nib.gifti.GiftiMetaData(
        AnatomicalStructurePrimary=ANATOMICAL_STRUCTURE[hemisphere],
        Subject=subject,
        Half=half,
        TimeStep=str(TR),
        Description=(
            f"{half} runs of the numerosity mapping experiment, averaged; rows are the "
            f"numerosity ROI gray nodes concatenated in the order {', '.join(MAP_NAMES)}"
        ),
    )
    image = nib.gifti.GiftiImage(meta=meta)
    for timepoint in time_series.T:
        image.add_gifti_data_array(
            nib.gifti.GiftiDataArray(
                np.ascontiguousarray(timepoint, dtype=np.float32),
                intent="NIFTI_INTENT_TIME_SERIES",
                datatype="NIFTI_TYPE_FLOAT32",
            ),
        )
    return image


def make_label_image(
    labels: np.ndarray,
    hemisphere: str,
    subject: str,
) -> nib.gifti.GiftiImage:
    """Build the ROI label file.

    Parameters
    ----------
    labels : numpy.ndarray
        Integer ROI label per row, indexing ``MAP_NAMES`` from zero.
    hemisphere : str
        Either ``"Left"`` or ``"Right"``.
    subject : str
        Subject identifier, stored in the file metadata.

    Returns
    -------
    nibabel.gifti.GiftiImage
        A single label array.
    """
    table = nib.gifti.GiftiLabelTable()
    colours = mpl.colormaps["tab10"]
    for index, map_name in enumerate(MAP_NAMES):
        red, green, blue, _ = colours(index % colours.N)
        entry = nib.gifti.GiftiLabel(key=index, red=red, green=green, blue=blue, alpha=1.0)
        entry.label = map_name
        table.labels.append(entry)

    meta = nib.gifti.GiftiMetaData(
        AnatomicalStructurePrimary=ANATOMICAL_STRUCTURE[hemisphere],
        Subject=subject,
        Description=(
            "Numerosity ROI membership of every row of the matching desc-odd/desc-even files; "
            f"label i is MAP_NAMES[i] with MAP_NAMES = {', '.join(MAP_NAMES)}"
        ),
    )
    image = nib.gifti.GiftiImage(meta=meta, labeltable=table)
    image.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
            np.ascontiguousarray(labels, dtype=np.int32),
            intent="NIFTI_INTENT_LABEL",
            datatype="NIFTI_TYPE_INT32",
        ),
    )
    return image


def main() -> None:
    """Write the GIFTI files for one subject."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="S1")
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data/numerosity-timing-fmri"))
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    output = args.output or args.data / "gifti"
    output.mkdir(parents=True, exist_ok=True)

    with File(args.data / "time_series.mat", "r") as handle:
        subject = handle["time_series"][args.subject]

        for hemisphere, short in HEMISPHERES.items():
            per_half, labels = collect_hemisphere(subject, hemisphere)
            stem = f"sub-{args.subject}_hemi-{short}"

            for half, time_series in per_half.items():
                path = output / f"{stem}_desc-{half}_bold.func.gii"
                make_functional_image(time_series, hemisphere, args.subject, half).to_filename(path)
                logger.info("wrote %s %s", path, time_series.shape)

            path = output / f"{stem}_desc-numerosity_dseg.label.gii"
            make_label_image(labels, hemisphere, args.subject).to_filename(path)
            logger.info("wrote %s (%d vertices)", path, labels.size)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
