"""Example stimuli and datasets.

Bundled stimuli (:func:`load_2d_prf_bar_stimulus`, :func:`load_1d_prf_lognumerosity_stimulus`) ship with the
package and need no download. Empirical datasets are fetched on first use with :func:`load_dataset` and cached
in a user data directory (see :func:`get_data_dir`).
"""

from prfmodel.examples._dataset import Dataset
from prfmodel.examples._fetch import ChecksumError
from prfmodel.examples._fetch import get_data_dir
from prfmodel.examples._loaders import load_dataset
from prfmodel.examples._registry import describe_dataset
from prfmodel.examples._registry import list_datasets
from prfmodel.examples._stimuli import load_1d_prf_lognumerosity_stimulus
from prfmodel.examples._stimuli import load_2d_prf_bar_stimulus

__all__ = [
    "ChecksumError",
    "Dataset",
    "describe_dataset",
    "get_data_dir",
    "list_datasets",
    "load_1d_prf_lognumerosity_stimulus",
    "load_2d_prf_bar_stimulus",
    "load_dataset",
]
