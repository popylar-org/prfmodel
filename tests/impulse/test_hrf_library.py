"""Test the bundled GLMsingle canonical HRF library."""

import hashlib
from importlib.resources import files
import numpy as np

_PATH = files("prfmodel.data.hrf").joinpath("getcanonicalhrflibrary.tsv")
# SHA-256 of the original file from GLMsingle (https://github.com/cvnlab/GLMsingle). The file is redistributed
# unmodified under the BSD 3-Clause License (see LICENSE-BSD), so its content must not change.
_SHA256 = "9e7b5c293819dc0d85b0006bc8712459db82d0d2a7695ce7219fde0efb5f0106"
_N_TIMEPOINTS = 501
_N_HRFS = 20


def test_hrf_library_is_packaged():
    """Test that the HRF library is shipped with the package."""
    assert _PATH.is_file()


def test_hrf_library_is_unmodified():
    """Test that the HRF library is identical to the GLMsingle original."""
    assert hashlib.sha256(_PATH.read_bytes()).hexdigest() == _SHA256


def test_hrf_library_shape():
    """Test that the HRF library contains 20 HRFs with 501 time points each."""
    with _PATH.open("r") as stream:
        hrfs = np.loadtxt(stream, delimiter="\t")

    assert hrfs.shape == (_N_TIMEPOINTS, _N_HRFS)
    assert np.all(np.isfinite(hrfs))
