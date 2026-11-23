"""Benchmark the polar grid against the cartesian grid for the pRF grid search.

The polar grid (:func:`~prfmodel.fitters.grid_values_2d_prf_polar`) places the candidates densely near the fovea,
where pRFs are small, and sparsely in the periphery. So this script compares how accurately each grid recovers
known pRFs for a given number of candidates.

1. pRFs are simulated with known centers and sizes. By default the eccentricities are log-uniform (dense near the
   fovea), the angles uniform, and the sizes grow with eccentricity. The true values lie between grid points, as
   they would in real data. Gaussian noise is added to the predicted time courses.
2. The grid search is run with the cartesian grid (:func:`~prfmodel.fitters.grid_values_2d_prf` with
   :class:`~prfmodel.models.prf.Gaussian2DPRFModel`) and with the polar grid (with
   :class:`~prfmodel.models.prf.PolarGaussian2DPRFModel`), each at several sizes. Polar estimates are converted to
   ``mu_x`` and ``mu_y`` with :func:`~prfmodel.models.prf.init_gaussian_from_polar`.
3. For each grid the script reports the number of candidates, the fit time, the error of the recovered center
   (distance to the true center) and size (relative error), overall and per eccentricity band, and the mean
   correlation of the best candidates.

Compare rows with similar numbers of candidates: a better grid has smaller errors for the same number of candidates,
or the same errors with fewer candidates (and so a faster fit).

The package is imported from the ``src/`` folder of this repository, ahead of any installed copy, so the script
always benchmarks the checked-out code. GPUs are hidden, so everything runs on the CPU. The Keras backend is chosen
as usual: the ``KERAS_BACKEND`` environment variable if set, else the ``"backend"`` entry in
``~/.keras/keras.json``, else TensorFlow.

Examples::

    uv run python scripts/benchmark_polar_grid.py
    uv run python scripts/benchmark_polar_grid.py --noise 0.0
    uv run python scripts/benchmark_polar_grid.py --mu-extent 1.0
    uv run python scripts/benchmark_polar_grid.py --cartesian-num-mu 11 21 --polar-num-ecc 8 16 --hrf

"""

import argparse
import os
import statistics
import sys
import time
import warnings
from itertools import pairwise
from pathlib import Path

# The repository's source folder, which is put first on the import path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"

# Eccentricity bands for the per-band center errors, as fractions of the largest simulated eccentricity
BAND_EDGES = (0.0, 0.1, 0.4, 1.0)


def _setup_environment() -> None:
    """Hide GPUs, silence progress bars and logs, and import prfmodel from this repository."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


def main() -> None:  # noqa: PLR0915 (a linear script)
    """Parse arguments, simulate pRFs, run both grid searches and report the results."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--units", type=int, default=2000, help="number of simulated pRFs")
    parser.add_argument("--noise", type=float, default=0.5, help="noise standard deviation relative to the signal's")
    parser.add_argument(
        "--ecc-dist",
        choices=("log", "area"),
        default="log",
        help="eccentricities log-uniform (dense near the fovea) or uniform over the visual field area",
    )
    parser.add_argument("--sigma-intercept", type=float, default=0.15, help="pRF size at eccentricity zero")
    parser.add_argument("--sigma-slope", type=float, default=0.15, help="increase of pRF size per unit eccentricity")
    parser.add_argument("--cartesian-num-mu", type=int, nargs="+", default=[11, 16, 21], help="sizes to compare")
    parser.add_argument("--polar-num-ecc", type=int, nargs="+", default=[8, 12, 16], help="sizes to compare")
    parser.add_argument("--num-sigma", type=int, default=20, help="values of sigma, the same for both grids")
    parser.add_argument(
        "--mu-extent",
        type=float,
        default=2.0,
        help="range of the pRF centers relative to the stimulus, the same for both grids (default of both helpers)",
    )
    parser.add_argument("--hrf", action="store_true", help="include the default impulse and scaling models")
    parser.add_argument("--batch-size", type=int, default=500, help="grid candidates per batch")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _setup_environment()

    import keras  # noqa: PLC0415 (imported after the environment is set up)
    import numpy as np  # noqa: PLC0415 (imported after the environment is set up)
    import pandas as pd  # noqa: PLC0415 (imported after the environment is set up)
    import prfmodel  # noqa: PLC0415 (imported after the environment is set up)
    from prfmodel.examples import load_2d_prf_bar_stimulus  # noqa: PLC0415 (imported after the environment is set up)
    from prfmodel.fitters import GridFitter  # noqa: PLC0415 (imported after the environment is set up)
    from prfmodel.fitters import grid_values_2d_prf  # noqa: PLC0415 (imported after the environment is set up)
    from prfmodel.fitters import grid_values_2d_prf_polar  # noqa: PLC0415 (imported after the environment is set up)
    from prfmodel.models.prf import Gaussian2DPRFModel  # noqa: PLC0415 (imported after the environment is set up)
    from prfmodel.models.prf import PolarGaussian2DPRFModel  # noqa: PLC0415 (imported after the environment is set up)
    from prfmodel.models.prf import init_gaussian_from_polar  # noqa: PLC0415 (imported after the environment is set up)

    rng = np.random.default_rng(args.seed)
    stimulus = load_2d_prf_bar_stimulus()

    # The grid stores y in grid[..., 0] (varying along rows) and x in grid[..., 1] (varying along columns)
    x_coords = stimulus.grid[0, :, 1]
    y_coords = stimulus.grid[:, 0, 0]
    resolution = float(abs(x_coords[1] - x_coords[0]))
    half_extent = max(float(np.ptp(x_coords)), float(np.ptp(y_coords))) / 2.0

    # Simulated pRFs inside the stimulus, between a twentieth and nine tenths of its half extent
    ecc_lo, ecc_hi = half_extent / 20.0, 0.9 * half_extent
    if args.ecc_dist == "log":
        ecc = np.exp(rng.uniform(np.log(ecc_lo), np.log(ecc_hi), args.units))
    else:
        ecc = np.sqrt(rng.uniform(ecc_lo**2, ecc_hi**2, args.units))
    angle = rng.uniform(-np.pi, np.pi, args.units)
    sigma = (args.sigma_intercept + args.sigma_slope * ecc) * rng.lognormal(0.0, 0.2, args.units)
    sigma = np.clip(sigma, resolution, half_extent)

    true = pd.DataFrame({"mu_x": ecc * np.cos(angle), "mu_y": ecc * np.sin(angle), "sigma": sigma})

    # Without '--hrf', the models predict the encoded pRF response only, which keeps the comparison fast
    model_kwargs = {} if args.hrf else {"impulse_model": None, "scaling_model": None}
    fixed = {"weight_deriv": [0.5], "baseline": [0.0], "amplitude": [1.0]} if args.hrf else {}
    for name, values in fixed.items():
        true[name] = values[0]

    signal = np.asarray(Gaussian2DPRFModel(**model_kwargs)(stimulus, true, dtype=args.dtype))
    noise_sd = args.noise * signal.std(axis=1, keepdims=True)
    data = (signal + noise_sd * rng.standard_normal(signal.shape)).astype(args.dtype)

    band_edges = [edge * ecc_hi for edge in BAND_EDGES]
    band_masks = [(ecc >= lo) & (ecc < hi) for lo, hi in pairwise(band_edges)]
    band_masks[-1] |= ecc >= band_edges[-1]

    def run(label: str, size: str, model: object, values: dict, to_cartesian: object) -> dict:
        fitter = GridFitter(model=model, stimulus=stimulus, dtype=args.dtype)
        parameter_values = values | fixed
        num_candidates = int(np.prod([len(v) for v in parameter_values.values()]))

        # A short warm-up, so the timed fit does not include one-off start-up costs
        fitter.fit(data[:10], parameter_values, batch_size=args.batch_size)

        start = time.perf_counter()
        history, params = fitter.fit(data, parameter_values, batch_size=args.batch_size)
        elapsed = time.perf_counter() - start

        params = to_cartesian(params)
        center_err = np.hypot(params["mu_x"] - true["mu_x"], params["mu_y"] - true["mu_y"]).to_numpy()
        sigma_err = np.abs(params["sigma"].to_numpy() / true["sigma"].to_numpy() - 1.0)

        return {
            "grid": label,
            "size": size,
            "candidates": num_candidates,
            "time_s": elapsed,
            "center_median": float(np.median(center_err)),
            "center_p90": float(np.percentile(center_err, 90)),
            "center_bands": [float(np.median(center_err[mask])) if mask.any() else float("nan") for mask in band_masks],
            "sigma_median_pct": 100.0 * float(np.median(sigma_err)),
            "mean_r": float(-np.mean(history.history["loss"])),
        }

    results = []
    with warnings.catch_warnings():
        # The grid helpers warn about alignment with the example stimulus, which does not affect the comparison
        warnings.simplefilter("ignore")

        for num_mu in args.cartesian_num_mu:
            values = grid_values_2d_prf(stimulus, num_mu=num_mu, mu_extent=args.mu_extent, num_sigma=args.num_sigma)
            results.append(
                run("cartesian", f"num_mu={num_mu}", Gaussian2DPRFModel(**model_kwargs), values, lambda p: p),
            )

        for num_ecc in args.polar_num_ecc:
            values = grid_values_2d_prf_polar(
                stimulus,
                num_ecc=num_ecc,
                mu_extent=args.mu_extent,
                num_sigma=args.num_sigma,
            )
            size = f"num_ecc={num_ecc}, num_angle={len(values['angle'])}"
            results.append(
                run("polar", size, PolarGaussian2DPRFModel(**model_kwargs), values, init_gaussian_from_polar),
            )

    band_labels = [f"{lo:.2g}-{hi:.2g}" for lo, hi in pairwise(band_edges)]
    band_counts = [int(mask.sum()) for mask in band_masks]

    lines = [
        (
            f"backend={keras.backend.backend()} units={args.units} frames={data.shape[1]} noise={args.noise} "
            f"ecc_dist={args.ecc_dist} hrf={args.hrf} num_sigma={args.num_sigma} mu_extent={args.mu_extent} "
            f"dtype={args.dtype}"
        ),
        f"prfmodel imported from {Path(prfmodel.__file__).parent}",
        (
            f"stimulus half extent={half_extent:.3g}, grid spacing={resolution:.3g}; "
            f"simulated ecc {ecc_lo:.3g}-{ecc_hi:.3g}, median sigma {statistics.median(sigma):.3g}"
        ),
        (
            "center errors are medians of the distance to the true center, in stimulus units; "
            "sigma error is the median relative error"
        ),
        "",
    ]
    header = (
        f"{'grid':<10} {'size':<26} {'cands':>6} {'time (s)':>9} {'ctr med':>8} {'ctr p90':>8} "
        + " ".join(f"{'ecc ' + label:>13}" for label in band_labels)
        + f" {'sigma %':>8} {'mean r':>7}"
    )
    # The band counts go under the band columns, which start after the first six columns (73 characters)
    lines += [header, " " * 73 + " ".join(f"{f'(n={count})':>13}" for count in band_counts), "-" * len(header)]
    for res in results:
        lines.append(
            f"{res['grid']:<10} {res['size']:<26} {res['candidates']:>6} {res['time_s']:>9.2f} "
            f"{res['center_median']:>8.3f} {res['center_p90']:>8.3f} "
            + " ".join(f"{value:>13.3f}" for value in res["center_bands"])
            + f" {res['sigma_median_pct']:>8.1f} {res['mean_r']:>7.3f}",
        )
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
