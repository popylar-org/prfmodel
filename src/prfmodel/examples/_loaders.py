"""Load example datasets."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from nibabel.gifti import GiftiImage
from nilearn.surface import PolyMesh
from nilearn.surface import load_surf_data
from prfmodel.examples._dataset import Dataset
from prfmodel.examples._fetch import FileFetcher
from prfmodel.examples._fetch import get_data_dir
from prfmodel.examples._options import validate_options
from prfmodel.examples._registry import _get_spec
from prfmodel.examples._registry import load_checksums
from prfmodel.examples._stimuli import load_1d_prf_lognumerosity_stimulus

if TYPE_CHECKING:
    import os
    from prfmodel.examples._options import Options


def load_retbar_visual(fetch: FileFetcher, options: Options) -> Dataset:
    """Load the response of a single subject to a moving bar stimulus."""
    responses = [
        np.asarray(load_surf_data(fetch(f"response_{code}")), dtype=np.float64) for code in options.hemisphere_codes
    ]

    # The raw design needs preprocessing before it can be turned into a stimulus, so we only expose its path
    fetch("design")

    return Dataset(
        name="7t-retbar-visual",
        response=np.concatenate(responses),
        hemisphere=options.hemisphere,
        files=fetch.files,
    )


def load_numerosity_timing(fetch: FileFetcher, options: Options) -> Dataset:
    """Load the response of a single subject to a sequence of visual numerosities."""
    responses = []
    roi_indices = []

    for code in options.hemisphere_codes:
        responses.append(np.asarray(load_surf_data(fetch(f"response_{options.split}_{code}")), dtype=np.float64))
        roi_indices.append(np.asarray(load_surf_data(fetch(f"roi_{code}")), dtype=np.int32))

    # The label table is identical across hemispheres so we read it from the first one
    label_image = GiftiImage.from_filename(fetch(f"roi_{options.hemisphere_codes[0]}"))

    return Dataset(
        name="numerosity-timing",
        response=np.concatenate(responses),
        stimulus=load_1d_prf_lognumerosity_stimulus(),
        roi_index=np.concatenate(roi_indices),
        roi_mapping=label_image.labeltable.get_labels_as_dict(),
        hemisphere=options.hemisphere,
        split=options.split,
        files=fetch.files,
    )


def load_hcp_surface(fetch: FileFetcher, options: Options) -> Dataset:
    """Load a standardized cortical surface mesh and the atlas that belongs to it."""
    mesh = PolyMesh(fetch(f"{options.surface_type}_lh"), fetch(f"{options.surface_type}_rh"))

    with np.load(fetch("atlas")) as archive:
        atlas = {"left": archive["left"], "right": archive["right"]}

    # The whole archive is downloaded anyway, so we expose every surface it holds
    files = {key: fetch(key) for key in fetch.keys}

    return Dataset(
        name="hcp-999999-surface",
        mesh=mesh,
        atlas=atlas,
        files=files,
    )


def load_dataset(  # noqa: PLR0913
    name: str,
    *,
    dest_dir: str | os.PathLike | None = None,
    hemisphere: str | None = None,
    split: str | None = None,
    surface_type: str | None = None,
    download: bool = True,
) -> Dataset:
    """
    Load an example dataset, downloading it on first use.

    The files of a dataset are cached in a data directory, so they are downloaded only once. Datasets are
    distributed as a single archive, which means that the whole archive is downloaded even when only part of
    it is loaded; only the files that a dataset needs are extracted and kept.

    Parameters
    ----------
    name : str
        Name of the dataset. Must be one of :func:`~prfmodel.examples.list_datasets`.
    dest_dir : str or os.PathLike, optional
        Directory in which the dataset files are stored. If `None` (the default), uses the directory from
        :func:`~prfmodel.examples.get_data_dir`.
    hemisphere : str, optional
        Hemisphere(s) to load the response for. Must be either `"both"`, `"left"`, or `"right"`. For
        `"both"`, the data of both hemispheres are concatenated (left is first). If `None` (the default),
        uses the default of the dataset.
    split : str, optional
        Data split to load the response from, for datasets that provide splits. Required for those datasets.
    surface_type : str, optional
        Surface type to load, for datasets that provide surfaces. Must be either `"flat"`, `"inflated"`,
        `"pia"` (or `"pial"`), or `"wm"` (for white matter).
    download : bool, default=True
        Whether files may be downloaded when they are missing. If `False`, missing files raise
        `FileNotFoundError` instead.

    Returns
    -------
    Dataset
        The dataset. Which of its fields are populated depends on the dataset; see
        :func:`~prfmodel.examples.describe_dataset`.

    Raises
    ------
    ValueError
        If `name` is not the name of an available dataset, if an option is not accepted by the dataset, or if
        the value of an option is not valid.
    FileNotFoundError
        If a file is missing and `download` is `False`.
    prfmodel.examples.ChecksumError
        If a downloaded file does not match its expected checksum.

    See Also
    --------
    prfmodel.examples.list_datasets : List the available example datasets.
    prfmodel.examples.describe_dataset : Describe an example dataset without loading it.
    prfmodel.examples.get_data_dir : Resolve the directory in which example datasets are stored.

    Examples
    --------
    .. code-block:: python

        from prfmodel.examples import load_dataset

        # A surface mesh and the atlas that belongs to it
        surface = load_dataset("hcp-999999-surface", surface_type="flat")
        surface.mesh

        # A response and the raw design of the stimulus that produced it
        dataset = load_dataset("7t-retbar-visual", hemisphere="both")
        dataset.response.shape

        # Two splits of the same response, for cross-validation
        odd = load_dataset("numerosity-timing", hemisphere="left", split="odd")
        even = load_dataset("numerosity-timing", hemisphere="left", split="even")

    """
    spec = _get_spec(name)
    options = validate_options(spec, hemisphere=hemisphere, split=split, surface_type=surface_type)

    if not spec.is_published:
        msg = (
            f"Dataset '{spec.name}' has not been published yet and cannot be downloaded. See "
            f"describe_dataset('{spec.name}') for details."
        )
        raise ValueError(msg)

    fetcher = FileFetcher(
        spec,
        data_dir=get_data_dir(dest_dir),
        checksums=load_checksums(),
        download=download,
    )

    return spec.loader(fetcher, options)
