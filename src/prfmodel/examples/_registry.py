"""Specifications of the available example datasets."""

from __future__ import annotations
import json
from dataclasses import dataclass
from dataclasses import field
from importlib.resources import files
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from prfmodel.examples._dataset import Dataset
    from prfmodel.examples._fetch import FileFetcher
    from prfmodel.examples._options import Options

_FIGSHARE_URL = "https://ndownloader.figshare.com/files"


@dataclass(frozen=True)
class DatasetSpec:
    """
    Specification of an example dataset.

    A dataset is distributed either as a single archive, given by `url`, or as individual files, given by
    `file_urls`. Exactly one of the two must be set.

    Parameters
    ----------
    name : str
        Name of the dataset, as passed to :func:`~prfmodel.examples.load_dataset`.
    summary : str
        One-paragraph description of what the dataset contains.
    url : str, optional
        URL of the archive that holds the dataset files, for a dataset distributed as one archive.
    files : collections.abc.Mapping of str to str
        Mapping from the logical name of a file to its path relative to the data directory. For a dataset
        distributed as one archive, this path is also the path of the file inside the archive.
    file_urls : collections.abc.Mapping of str to str
        Mapping from the logical name of a file to the URL it is downloaded from, for a dataset distributed
        as individual files. Only the files that are actually loaded are downloaded.
    loader : collections.abc.Callable
        Function that turns a file fetcher and a set of options into a :class:`~prfmodel.examples.Dataset`.
    options : frozenset of str
        Names of the :func:`~prfmodel.examples.load_dataset` options that this dataset accepts.
    splits : tuple of str
        Names of the data splits that this dataset provides. When non-empty, a split must be requested.
    hemispheres : tuple of str
        Hemisphere selections that this dataset accepts.
    surface_types : tuple of str
        Surface types that this dataset accepts.
    default_hemisphere : str
        Hemisphere selection that is used when none is requested.
    licence : str
        Licence that the dataset is distributed under.
    citation : str
        Reference that should be cited when the dataset is used.
    homepage : str
        URL of the page that describes the dataset.
    download_size : int
        Total size of the dataset in bytes. A dataset distributed as one archive downloads all of it even
        when only part of it is loaded, while a dataset distributed as individual files downloads only the
        files that are loaded.

    """

    name: str
    summary: str
    files: Mapping[str, str]
    loader: Callable[[FileFetcher, Options], Dataset]
    url: str | None = None
    file_urls: Mapping[str, str] = field(default_factory=dict)
    options: frozenset[str] = frozenset()
    splits: tuple[str, ...] = ()
    hemispheres: tuple[str, ...] = ()
    surface_types: tuple[str, ...] = ()
    default_hemisphere: str = "both"
    licence: str = ""
    citation: str = ""
    homepage: str = ""
    download_size: int = 0

    def __post_init__(self) -> None:
        if (self.url is None) == (not self.file_urls):
            msg = f"Dataset '{self.name}' must set either 'url' or 'file_urls' but not both"
            raise ValueError(msg)

        if self.file_urls and set(self.file_urls) != set(self.files):
            msg = f"Dataset '{self.name}' must give a URL for every file in 'files'"
            raise ValueError(msg)

    @property
    def is_archive(self) -> bool:
        """Whether this dataset is distributed as a single archive rather than as individual files."""
        return self.url is not None


def _make_registry() -> dict[str, DatasetSpec]:
    # Imported here because the loaders import this module for its specifications
    from prfmodel.examples import _loaders  # noqa: PLC0415 (deliberate to break an import cycle)

    specs = (
        DatasetSpec(
            name="7t-retbar-visual",
            summary=(
                "Blood oxygenation level-dependent response of a single subject to a moving bar stimulus, "
                "measured in a 7T scanner with a repetition time of 1.5 s and averaged over 43 runs across "
                "four sessions. The response is given for the 59k vertices of each hemisphere of a "
                "standardized cortical surface. The raw MATLAB design matrix of the stimulus is included as "
                "the 'design' file; it needs to be padded, binarized, and smoothed before it can be turned "
                "into a stimulus, which is why this dataset provides no ready-made one."
            ),
            url=f"{_FIGSHARE_URL}/26577941",
            files={
                "response_L": "sub-02_task-prf_space-59k_hemi-L_run-median_desc-bold.func.gii",
                "response_R": "sub-02_task-prf_space-59k_hemi-R_run-median_desc-bold.func.gii",
                "design": "vis_design.mat",
            },
            loader=_loaders.load_retbar_visual,
            options=frozenset({"hemisphere"}),
            hemispheres=("both", "left", "right"),
            default_hemisphere="both",
            licence="CC BY 4.0",
            citation="Knapen, T. (2021). fMRI Teaching Materials. figshare. https://doi.org/10.6084/m9.figshare.14096209",
            homepage="https://figshare.com/articles/dataset/fMRI_Teaching_Materials/14096209",
            download_size=51_000_000,
        ),
        DatasetSpec(
            name="numerosity-timing",
            summary=(
                "Blood oxygenation level-dependent response of a single subject to a sequence of visual "
                "numerosities, averaged separately over the two odd and the two even runs so that the two "
                "splits can be used for cross-validation. Rows are the gray nodes of the eight "
                "numerosity-selective regions of interest rather than surface vertices, so concatenating "
                "both hemispheres carries no anatomical adjacency."
            ),
            files={
                "response_odd_L": "sub-S1_hemi-L_desc-odd_bold.func.gii",
                "response_even_L": "sub-S1_hemi-L_desc-even_bold.func.gii",
                "response_odd_R": "sub-S1_hemi-R_desc-odd_bold.func.gii",
                "response_even_R": "sub-S1_hemi-R_desc-even_bold.func.gii",
                "roi_L": "sub-S1_hemi-L_desc-numerosity_dseg.label.gii",
                "roi_R": "sub-S1_hemi-R_desc-numerosity_dseg.label.gii",
            },
            file_urls={
                "response_odd_L": f"{_FIGSHARE_URL}/68359336",
                "response_even_L": f"{_FIGSHARE_URL}/68359342",
                "response_odd_R": f"{_FIGSHARE_URL}/68359327",
                "response_even_R": f"{_FIGSHARE_URL}/68359333",
                "roi_L": f"{_FIGSHARE_URL}/68359339",
                "roi_R": f"{_FIGSHARE_URL}/68359330",
            },
            loader=_loaders.load_numerosity_timing,
            options=frozenset({"hemisphere", "split"}),
            splits=("odd", "even"),
            hemispheres=("both", "left", "right"),
            default_hemisphere="left",
            licence="CC BY 4.0",
            citation=(
                "Hendrikx, E., Paul, J. M., van Ackooij, M., van der Stoep, N., & Harvey, B. M. (2024). "
                "Cortical quantity representations of visual numerosity and timing overlap increasingly into "
                "superior cortices but remain distinct. NeuroImage, 286, 120515. "
                "https://doi.org/10.1016/j.neuroimage.2024.120515 -- and the derived files as: "
                "Luken, M. (2026). Single-subject numerosity-timing example dataset. figshare. "
                "https://doi.org/10.6084/m9.figshare.33481645"
            ),
            homepage="https://doi.org/10.6084/m9.figshare.33481645",
            download_size=17_447_831,
        ),
        DatasetSpec(
            name="hcp-999999-surface",
            summary=(
                "Standardized cortical surface meshes based on the Human Connectome Project, together with "
                "the multi-modal parcellation atlas that belongs to them. Each hemisphere has 59k vertices, "
                "so responses from the '7t-retbar-visual' dataset can be projected onto these meshes."
            ),
            url=f"{_FIGSHARE_URL}/25768841",
            files={
                "flat_lh": "hcp_999999/surfaces/flat_lh.gii",
                "flat_rh": "hcp_999999/surfaces/flat_rh.gii",
                "inflated_lh": "hcp_999999/surfaces/inflated_lh.gii",
                "inflated_rh": "hcp_999999/surfaces/inflated_rh.gii",
                "pia_lh": "hcp_999999/surfaces/pia_lh.gii",
                "pia_rh": "hcp_999999/surfaces/pia_rh.gii",
                "wm_lh": "hcp_999999/surfaces/wm_lh.gii",
                "wm_rh": "hcp_999999/surfaces/wm_rh.gii",
                "atlas": "hcp_999999/surface-info/mmp_atlas.npz",
            },
            loader=_loaders.load_hcp_surface,
            options=frozenset({"surface_type"}),
            surface_types=("flat", "inflated", "pia", "wm"),
            licence="CC BY 4.0",
            citation=(
                "Van Essen, D. C., Ugurbil, K., Auerbach, E., Barch, D., Behrens, T. E. J., Bucholz, R., "
                "Chang, A., Chen, L., Corbetta, M., Curtiss, S. W., Della Penna, S., Feinberg, D., Glasser, "
                "M. F., Harel, N., Heath, A. C., Larson-Prior, L., Marcus, D., Michalareas, G., Moeller, S., "
                "... WU-Minn HCP Consortium. (2012). The Human Connectome Project: A data acquisition "
                "perspective. NeuroImage, 62(4), 2222-2231. https://doi.org/10.1016/j.neuroimage.2012.02.018"
            ),
            homepage="https://doi.org/10.6084/m9.figshare.13372958",
            download_size=61_000_000,
        ),
    )

    return {spec.name: spec for spec in specs}


_REGISTRY: dict[str, DatasetSpec] = {}


def _get_registry() -> dict[str, DatasetSpec]:
    if not _REGISTRY:
        _REGISTRY.update(_make_registry())

    return _REGISTRY


def _get_spec(name: str) -> DatasetSpec:
    registry = _get_registry()

    if name not in registry:
        msg = f"Dataset must be one of {tuple(sorted(registry))} but was '{name}'"
        raise ValueError(msg)

    return registry[name]


def load_checksums() -> dict[str, dict[str, object]]:
    """
    Load the expected checksums of all example dataset files.

    Returns
    -------
    dict of str to dict
        Mapping from the path of a file relative to the data directory to a dictionary with its `"sha256"`
        digest and its `"size"` in bytes.

    """
    path = files("prfmodel.data").joinpath("checksums.json")

    with path.open("r") as stream:
        return json.load(stream)["files"]


def list_datasets() -> list[str]:
    """
    List the names of the available example datasets.

    Returns
    -------
    list of str
        The dataset names, sorted alphabetically. Each can be passed to
        :func:`~prfmodel.examples.load_dataset`.

    Examples
    --------
    >>> from prfmodel.examples import list_datasets
    >>> list_datasets()
    ['7t-retbar-visual', 'hcp-999999-surface', 'numerosity-timing']

    """
    return sorted(_get_registry())


def _download_note(spec: DatasetSpec) -> str:
    if spec.is_archive:
        return "(the whole archive, also when loading part of it)"

    return "(in total; only the files that are loaded are downloaded)"


def describe_dataset(name: str) -> str:
    """
    Describe an example dataset without downloading it.

    Parameters
    ----------
    name : str
        Name of the dataset. Must be one of :func:`~prfmodel.examples.list_datasets`.

    Returns
    -------
    str
        A human-readable description with the contents, options, size, licence, and citation of the dataset.

    Raises
    ------
    ValueError
        If `name` is not the name of an available dataset.

    Examples
    --------
    >>> from prfmodel.examples import describe_dataset
    >>> print(describe_dataset("7t-retbar-visual"))  # doctest: +ELLIPSIS
    7t-retbar-visual
    ================
    <BLANKLINE>
    Blood oxygenation level-dependent response of a single subject to a moving bar stimulus, ...
    <BLANKLINE>
    Files: response_L, response_R, design
    Hemispheres: both, left, right (default 'both')
    Download: 51 MB (the whole archive, also when loading part of it)
    Licence: CC BY 4.0
    Homepage: https://figshare.com/articles/dataset/fMRI_Teaching_Materials/14096209
    <BLANKLINE>
    Cite as: Knapen, T. (2021). fMRI Teaching Materials. figshare. https://doi.org/10.6084/m9.figshare.14096209

    """
    spec = _get_spec(name)

    lines = [f"{spec.name}", "=" * len(spec.name), "", spec.summary, ""]

    lines += [f"Files: {', '.join(spec.files)}"]

    if spec.hemispheres:
        lines += [f"Hemispheres: {', '.join(spec.hemispheres)} (default '{spec.default_hemisphere}')"]

    if spec.splits:
        lines += [f"Splits: {', '.join(spec.splits)} (required)"]

    if spec.surface_types:
        lines += [f"Surface types: {', '.join(spec.surface_types)}"]

    lines += [
        f"Download: {spec.download_size / 1e6:.0f} MB {_download_note(spec)}",
        f"Licence: {spec.licence}",
        f"Homepage: {spec.homepage}",
        "",
        f"Cite as: {spec.citation}",
    ]

    return "\n".join(lines)
