"""Benchmark the polar grid against the cartesian grid for the pRF grid search.

The polar grid (:func:`~prfmodel.fitters.grid_values_2d_prf_polar`) places the candidates densely near the fovea,
where pRFs are small, and sparsely in the periphery. So this script compares how accurately each grid recovers
known pRFs for a given number of candidates.

1. pRFs with known centers and sizes are chosen with ``--ecc-dist``:

   - ``log`` (default): eccentricities log-uniform (dense near the fovea), roughly what cortical magnification gives
     in early visual cortex. Angles are uniform and sizes grow with eccentricity.
   - ``area``: eccentricities uniform over the visual field area, so most pRFs are in the periphery.
   - ``aot``: the pRFs of real voxels from the ``7t-aot-visual`` example dataset (downloaded on first use). A random
     sample of voxels is fitted with a cartesian grid, and the voxels with a correlation of at least ``--min-r`` and a
     center inside the stimulus are kept. Their estimates are jittered within their grid cell, so that the true
     values do not lie on the points of any grid that is compared. The models then include the default impulse and
     scaling models, as the real data needs them.

   Gaussian noise is added to the predicted time courses of these pRFs.
2. The grid search is run with the cartesian grid (:func:`~prfmodel.fitters.grid_values_2d_prf` with
   :class:`~prfmodel.models.prf.Gaussian2DPRFModel`) and with the polar grid (with
   :class:`~prfmodel.models.prf.PolarGaussian2DPRFModel`), each at several sizes. Polar estimates are converted to
   ``mu_x`` and ``mu_y`` with :func:`~prfmodel.models.prf.init_gaussian_from_polar`.
3. For each grid the script reports the number of candidates, the fit time, the error of the recovered center
   (distance to the true center) and size (relative error), overall and per eccentricity band, and the mean
   correlation of the best candidates.
4. With ``--ecc-dist aot``, both grids are also fitted to the real time courses of the kept voxels. There is no
   ground truth there, so the script reports how far each grid's correlation falls short of the best correlation
   any grid reached for the same voxel, overall and per eccentricity band (from the screening fit).

Compare rows with similar numbers of candidates: a better grid has smaller errors (and a smaller shortfall) for the
same number of candidates, or the same errors with fewer candidates (and so a faster fit).

The package is imported from the ``src/`` folder of this repository, ahead of any installed copy, so the script
always benchmarks the checked-out code. GPUs are hidden, so everything runs on the CPU. The Keras backend is chosen
as usual: the ``KERAS_BACKEND`` environment variable if set, else the ``"backend"`` entry in
``~/.keras/keras.json``, else TensorFlow.

Examples::

    uv run python scripts/benchmark_polar_grid.py
    uv run python scripts/benchmark_polar_grid.py --noise 0.0
    uv run python scripts/benchmark_polar_grid.py --mu-extent 1.0
    uv run python scripts/benchmark_polar_grid.py --mu-extent 1.0 --ecc-dist aot
    uv run python scripts/benchmark_polar_grid.py --cartesian-num-mu 11 21 --polar-num-ecc 8 16 --hrf

"""

import argparse
import os
import sys
import time
import warnings
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

# The repository's source folder, which is put first on the import path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"

# Eccentricity bands for the per-band results, as fractions of the largest eccentricity of the pRFs (0.9 times half
# the largest extent of the stimulus)
BAND_EDGES = (0.0, 0.1, 0.4, 1.0)

# Values of the impulse and scaling parameters, which are not searched. The correlation loss ignores baseline and
# amplitude, so their values do not matter.
FIXED_HRF = {"weight_deriv": [0.5], "baseline": [0.0], "amplitude": [1.0]}

# Screening grid for the AOT voxels: 21 x 21 centers over the stimulus
SCREEN_NUM_MU = 21

# Voxels per screening fit, which bounds the memory of the (candidates, voxels, frames) array of the loss
SCREEN_CHUNK = 2000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--units", type=int, default=2000, help="number of pRFs (at most this many voxels for 'aot')")
    parser.add_argument("--noise", type=float, default=0.5, help="noise standard deviation relative to the signal's")
    parser.add_argument(
        "--ecc-dist",
        choices=("log", "area", "aot"),
        default="log",
        help="pRFs log-uniform in eccentricity, uniform over the visual field area, or from the AOT dataset",
    )
    parser.add_argument("--sigma-intercept", type=float, default=0.15, help="pRF size at eccentricity zero")
    parser.add_argument("--sigma-slope", type=float, default=0.15, help="increase of pRF size per unit eccentricity")
    parser.add_argument("--screen-units", type=int, default=20000, help="AOT voxels to screen ('aot' only)")
    parser.add_argument("--min-r", type=float, default=0.3, help="smallest screening correlation to keep a voxel")
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
    return parser.parse_args()


def _setup_environment() -> None:
    """Hide GPUs, silence progress bars and logs, and import prfmodel from this repository."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


def _import_prfmodel() -> SimpleNamespace:
    """Import Keras and prfmodel, which loads the backend, so only after the environment is set up."""
    import keras  # noqa: PLC0415 (after the environment is set up)
    import prfmodel  # noqa: PLC0415 (after the environment is set up)
    from prfmodel.examples import load_2d_prf_bar_stimulus  # noqa: PLC0415 (after the environment is set up)
    from prfmodel.examples import load_dataset  # noqa: PLC0415 (after the environment is set up)
    from prfmodel.fitters import GridFitter  # noqa: PLC0415 (after the environment is set up)
    from prfmodel.fitters import grid_values_2d_prf  # noqa: PLC0415 (after the environment is set up)
    from prfmodel.fitters import grid_values_2d_prf_polar  # noqa: PLC0415 (after the environment is set up)
    from prfmodel.models.prf import Gaussian2DPRFModel  # noqa: PLC0415 (after the environment is set up)
    from prfmodel.models.prf import PolarGaussian2DPRFModel  # noqa: PLC0415 (after the environment is set up)
    from prfmodel.models.prf import init_gaussian_from_polar  # noqa: PLC0415 (after the environment is set up)

    return SimpleNamespace(**locals())


def _geometry(stimulus: object) -> tuple[float, float]:
    """Return the spacing of the stimulus grid and half of its largest extent."""
    # The grid stores y in grid[..., 0] (varying along rows) and x in grid[..., 1] (varying along columns)
    x_coords = stimulus.grid[0, :, 1]
    y_coords = stimulus.grid[:, 0, 0]
    return float(abs(x_coords[1] - x_coords[0])), max(float(np.ptp(x_coords)), float(np.ptp(y_coords))) / 2.0


def _band_masks(ecc: np.ndarray, ctx: SimpleNamespace) -> list[np.ndarray]:
    """Return one boolean mask per eccentricity band; eccentricities beyond the last edge count to the last band."""
    masks = [(ecc >= lo) & (ecc < hi) for lo, hi in pairwise(ctx.band_edges)]
    masks[-1] |= ecc >= ctx.band_edges[-1]
    return masks


def _simulate_prfs(ctx: SimpleNamespace, rng: np.random.Generator) -> tuple[pd.DataFrame, str]:
    """Draw pRF centers and sizes inside the stimulus, between a twentieth and nine tenths of its half extent."""
    args = ctx.args
    ecc_lo = ctx.half_extent / 20.0
    if args.ecc_dist == "log":
        ecc = np.exp(rng.uniform(np.log(ecc_lo), np.log(ctx.ecc_hi), args.units))
    else:
        ecc = np.sqrt(rng.uniform(ecc_lo**2, ctx.ecc_hi**2, args.units))
    angle = rng.uniform(-np.pi, np.pi, args.units)
    sigma = (args.sigma_intercept + args.sigma_slope * ecc) * rng.lognormal(0.0, 0.2, args.units)
    sigma = np.clip(sigma, ctx.resolution, ctx.half_extent)

    truth = pd.DataFrame({"mu_x": ecc * np.cos(angle), "mu_y": ecc * np.sin(angle), "sigma": sigma})
    return truth, f"simulated pRFs, eccentricities {args.ecc_dist}"


def _screen_aot(
    ctx: SimpleNamespace,
    rng: np.random.Generator,
    dataset: object,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
    """Fit a cartesian grid to a random sample of AOT voxels and keep the well-fitted ones inside the stimulus.

    Returns the pRFs to simulate (the estimates jittered within their grid cell, so that they do not lie on grid
    points), the real time courses of the kept voxels, their estimated eccentricities, and a summary line.

    """
    args, pm = ctx.args, ctx.pm
    response = dataset.response
    # A random order, so that the first voxels that pass the screening are not from one region of the volume
    sample = response[rng.choice(len(response), size=min(args.screen_units, len(response)), replace=False)]
    sample = sample.astype(args.dtype)

    values = pm.grid_values_2d_prf(ctx.stimulus, num_mu=SCREEN_NUM_MU, mu_extent=1.0, num_sigma=args.num_sigma)
    fitter = pm.GridFitter(model=pm.Gaussian2DPRFModel(**ctx.model_kwargs), stimulus=ctx.stimulus, dtype=args.dtype)

    estimates, corr = [], []
    for start in range(0, len(sample), SCREEN_CHUNK):
        history, params = fitter.fit(
            sample[start : start + SCREEN_CHUNK],
            values | ctx.fixed,
            batch_size=args.batch_size,
        )
        estimates.append(params)
        corr.append(-np.asarray(history.history["loss"]))
    estimates = pd.concat(estimates, ignore_index=True)
    corr = np.concatenate(corr)
    ecc = np.hypot(estimates["mu_x"], estimates["mu_y"]).to_numpy()

    # Voxels excluded from the fit have a NaN correlation, which fails the comparison and drops them
    keep = np.flatnonzero((corr >= args.min_r) & (ecc <= ctx.ecc_hi))[: args.units]
    if len(keep) == 0:
        msg = f"No screened voxel has a correlation of at least {args.min_r} and a center inside the stimulus"
        raise SystemExit(msg)

    mu_step = float(values["mu_x"][1] - values["mu_x"][0])
    log_sigma_step = float(np.log(values["sigma"][1] / values["sigma"][0]))
    sigma_factor = np.exp(rng.uniform(-log_sigma_step / 2, log_sigma_step / 2, len(keep)))
    sigma = estimates["sigma"].to_numpy()[keep] * sigma_factor
    truth = pd.DataFrame(
        {
            "mu_x": estimates["mu_x"].to_numpy()[keep] + rng.uniform(-mu_step / 2, mu_step / 2, len(keep)),
            "mu_y": estimates["mu_y"].to_numpy()[keep] + rng.uniform(-mu_step / 2, mu_step / 2, len(keep)),
            "sigma": np.clip(sigma, ctx.resolution, ctx.half_extent),
        },
    )

    summary = (
        f"AOT voxels: screened {len(sample)} of {len(response)}, {int(np.sum(corr >= args.min_r))} with "
        f"r >= {args.min_r}, using {len(keep)} with centers within eccentricity {ctx.ecc_hi:.3g}"
    )
    return truth, sample[keep], ecc[keep], summary


def _grids(ctx: SimpleNamespace) -> list[tuple[str, str, object, dict, Callable]]:
    """Return the label, size, model, parameter values and conversion to mu_x and mu_y of every grid to compare."""
    args, pm = ctx.args, ctx.pm
    grids = []

    for num_mu in args.cartesian_num_mu:
        values = pm.grid_values_2d_prf(
            ctx.stimulus,
            num_mu=num_mu,
            mu_extent=args.mu_extent,
            num_sigma=args.num_sigma,
        )
        model = pm.Gaussian2DPRFModel(**ctx.model_kwargs)
        grids.append(("cartesian", f"num_mu={num_mu}", model, values | ctx.fixed, lambda params: params))

    for num_ecc in args.polar_num_ecc:
        values = pm.grid_values_2d_prf_polar(
            ctx.stimulus,
            num_ecc=num_ecc,
            mu_extent=args.mu_extent,
            num_sigma=args.num_sigma,
        )
        size = f"num_ecc={num_ecc}, num_angle={len(values['angle'])}"
        model = pm.PolarGaussian2DPRFModel(**ctx.model_kwargs)
        grids.append(("polar", size, model, values | ctx.fixed, pm.init_gaussian_from_polar))

    return grids


def _fit(
    ctx: SimpleNamespace,
    model: object,
    data: np.ndarray,
    values: dict,
) -> tuple[pd.DataFrame, np.ndarray, int, float]:
    """Run one grid search; return the estimates, the correlation per unit, the number of candidates and the time."""
    fitter = ctx.pm.GridFitter(model=model, stimulus=ctx.stimulus, dtype=ctx.args.dtype)
    num_candidates = int(np.prod([len(v) for v in values.values()]))

    # A short warm-up, so the timed fit does not include one-off start-up costs
    fitter.fit(data[:10], values, batch_size=ctx.args.batch_size)

    start = time.perf_counter()
    history, params = fitter.fit(data, values, batch_size=ctx.args.batch_size)
    elapsed = time.perf_counter() - start

    return params, -np.asarray(history.history["loss"]), num_candidates, elapsed


def _simulated_results(
    ctx: SimpleNamespace,
    grids: list[tuple],
    truth: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[list[dict], list[int]]:
    """Fit every grid to noisy predictions of the known pRFs and measure how well it recovers them."""
    params_true = truth.assign(**{name: values[0] for name, values in ctx.fixed.items()})
    model = ctx.pm.Gaussian2DPRFModel(**ctx.model_kwargs)
    signal = np.asarray(model(ctx.stimulus, params_true, dtype=ctx.args.dtype))
    noise_sd = ctx.args.noise * signal.std(axis=1, keepdims=True)
    data = (signal + noise_sd * rng.standard_normal(signal.shape)).astype(ctx.args.dtype)

    masks = _band_masks(np.hypot(truth["mu_x"], truth["mu_y"]).to_numpy(), ctx)

    rows = []
    for label, size, grid_model, values, to_cartesian in grids:
        params, corr, num_candidates, elapsed = _fit(ctx, grid_model, data, values)
        params = to_cartesian(params)
        center_err = np.hypot(params["mu_x"] - truth["mu_x"], params["mu_y"] - truth["mu_y"]).to_numpy()
        sigma_err = np.abs(params["sigma"].to_numpy() / truth["sigma"].to_numpy() - 1.0)

        rows.append(
            {
                "grid": label,
                "size": size,
                "candidates": num_candidates,
                "time_s": elapsed,
                "center_median": float(np.median(center_err)),
                "center_p90": float(np.percentile(center_err, 90)),
                "center_bands": [float(np.median(center_err[mask])) if mask.any() else np.nan for mask in masks],
                "sigma_median_pct": 100.0 * float(np.median(sigma_err)),
                "mean_r": float(np.mean(corr)),
            },
        )

    return rows, [int(mask.sum()) for mask in masks]


def _real_results(
    ctx: SimpleNamespace,
    grids: list[tuple],
    data: np.ndarray,
    ecc: np.ndarray,
) -> tuple[list[dict], list[int]]:
    """Fit every grid to real time courses and measure how far its correlation falls short of the best grid's."""
    masks = _band_masks(ecc, ctx)

    fits = []
    for label, size, model, values, _ in grids:
        _, corr, num_candidates, elapsed = _fit(ctx, model, data, values)
        fits.append((label, size, corr, num_candidates, elapsed))

    best = np.max(np.stack([corr for _, _, corr, _, _ in fits]), axis=0)

    rows = []
    for label, size, corr, num_candidates, elapsed in fits:
        shortfall = best - corr
        rows.append(
            {
                "grid": label,
                "size": size,
                "candidates": num_candidates,
                "time_s": elapsed,
                "mean_r": float(np.mean(corr)),
                "shortfall": float(np.mean(shortfall)),
                "shortfall_bands": [float(np.mean(shortfall[mask])) if mask.any() else np.nan for mask in masks],
            },
        )

    return rows, [int(mask.sum()) for mask in masks]


def _table(rows: list[dict], columns: list[tuple], counts: list[int], band_edges: list[float]) -> list[str]:
    """Format rows as a table, with the number of units per eccentricity band under the band columns.

    Each column is (title, key, width, format), left-aligned for a negative width. A key ending in '_bands' holds one
    value per band and expands into one column per band, titled with the band.

    """
    band_titles = [f"ecc {lo:.2g}-{hi:.2g}" for lo, hi in pairwise(band_edges)]
    titles, offset = [], 0
    for title, key, width, _ in columns:
        if key.endswith("_bands"):
            # The band columns start after the titles so far and the space that joins them
            offset = len(" ".join(titles)) + 1
            titles += [f"{band:>{BAND_WIDTH}}" for band in band_titles]
        else:
            titles.append(f"{title:<{-width}}" if width < 0 else f"{title:>{width}}")

    header = " ".join(titles)
    counts_line = " " * offset + " ".join(f"{f'(n={count})':>{BAND_WIDTH}}" for count in counts)
    lines = [header, counts_line, "-" * len(header)]

    for row in rows:
        cells = []
        for _, key, width, fmt in columns:
            if key.endswith("_bands"):
                cells += [f"{value:>{BAND_WIDTH}{fmt}}" for value in row[key]]
            else:
                cells.append(f"{row[key]:<{-width}{fmt}}" if width < 0 else f"{row[key]:>{width}{fmt}}")
        lines.append(" ".join(cells))

    return lines


# Width of every eccentricity band column
BAND_WIDTH = 13

COMMON_COLUMNS = [
    ("grid", "grid", -10, ""),
    ("size", "size", -26, ""),
    ("cands", "candidates", 6, ""),
    ("time (s)", "time_s", 9, ".2f"),
]

SIMULATED_COLUMNS = [
    *COMMON_COLUMNS,
    ("ctr med", "center_median", 8, ".3f"),
    ("ctr p90", "center_p90", 8, ".3f"),
    ("", "center_bands", BAND_WIDTH, ".3f"),
    ("sigma %", "sigma_median_pct", 8, ".1f"),
    ("mean r", "mean_r", 7, ".3f"),
]

REAL_COLUMNS = [
    *COMMON_COLUMNS,
    ("mean r", "mean_r", 7, ".3f"),
    ("shortfall", "shortfall", 9, ".4f"),
    ("", "shortfall_bands", BAND_WIDTH, ".4f"),
]


def main() -> None:
    """Choose the pRFs, run both grid searches and report the results."""
    args = _parse_args()
    _setup_environment()
    pm = _import_prfmodel()
    rng = np.random.default_rng(args.seed)

    dataset = pm.load_dataset("7t-aot-visual") if args.ecc_dist == "aot" else None
    stimulus = dataset.stimulus if dataset is not None else pm.load_2d_prf_bar_stimulus()

    # Real voxel time courses need a hemodynamic response, so the AOT mode always includes the impulse model
    hrf = args.hrf or dataset is not None
    resolution, half_extent = _geometry(stimulus)
    ecc_hi = 0.9 * half_extent
    ctx = SimpleNamespace(
        args=args,
        pm=pm,
        stimulus=stimulus,
        resolution=resolution,
        half_extent=half_extent,
        ecc_hi=ecc_hi,
        band_edges=[edge * ecc_hi for edge in BAND_EDGES],
        model_kwargs={} if hrf else {"impulse_model": None, "scaling_model": None},
        fixed=FIXED_HRF if hrf else {},
    )
    with warnings.catch_warnings():
        # The grid helpers warn about alignment with the example stimulus, which does not affect the comparison
        warnings.simplefilter("ignore")

        if dataset is None:
            truth, source = _simulate_prfs(ctx, rng)
            real_data = real_ecc = None
        else:
            truth, real_data, real_ecc, source = _screen_aot(ctx, rng, dataset)

        grids = _grids(ctx)
        simulated, simulated_counts = _simulated_results(ctx, grids, truth, rng)
        real = _real_results(ctx, grids, real_data, real_ecc) if real_data is not None else None

    true_ecc = np.hypot(truth["mu_x"], truth["mu_y"])
    lines = [
        (
            f"backend={pm.keras.backend.backend()} units={len(truth)} frames={stimulus.design.shape[0]} "
            f"noise={args.noise} ecc_dist={args.ecc_dist} hrf={hrf} num_sigma={args.num_sigma} "
            f"mu_extent={args.mu_extent} dtype={args.dtype}"
        ),
        f"prfmodel imported from {Path(pm.prfmodel.__file__).parent}",
        source,
        (
            f"stimulus half extent={half_extent:.3g}, grid spacing={resolution:.3g}; true pRFs: ecc "
            f"{true_ecc.min():.3g}-{true_ecc.max():.3g}, median sigma {truth['sigma'].median():.3g}"
        ),
        "",
        (
            "Known pRFs, noisy predicted time courses. Center errors are medians of the distance to the true center, "
            "in stimulus units; sigma error is the median relative error."
        ),
        *_table(simulated, SIMULATED_COLUMNS, simulated_counts, ctx.band_edges),
    ]

    if real is not None:
        real_rows, real_counts = real
        lines += [
            "",
            (
                "Real AOT time courses of the same voxels. Shortfall is the best correlation of any grid for a voxel "
                "minus this grid's, averaged (lower is better); bands use the screening eccentricity."
            ),
            *_table(real_rows, REAL_COLUMNS, real_counts, ctx.band_edges),
        ]

    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
