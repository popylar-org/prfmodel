"""Benchmark the pairwise (matrix product) correlation loss against broadcasting.

Compares two ways of computing the correlation loss in the grid search:

- ``pairwise``: ``CorrelationLoss(pairwise=True)``, the ``GridFitter`` default (matrix product).
- ``broadcast``: ``CorrelationLoss(pairwise=False)``, the previous behaviour (``(G, U, T)`` intermediate array).

Two modes:

- ``fit`` (default): times a full ``GridFitter.fit`` on random data, so model predictions are included. This is the
  number that matters in practice. The grid comes from :func:`~prfmodel.fitters.grid_values_2d_prf` with its
  default sizes (21 x 21 x 20 = 8820 candidates), which ``--num-mu`` and ``--num-sigma`` change.
- ``loss``: times only the loss on random inputs of shape ``(1, U, T)`` and ``(G, 1, T)``, which isolates the change.

Every configuration runs in a fresh Python process, so that its peak memory is measured on its own. The reported
memory is the increase in the process's peak resident set size during the timed runs, relative to after set-up.

The package is imported from the ``src/`` folder of this repository, ahead of any installed copy, so the script
always benchmarks the checked-out code.

GPUs are hidden, so everything runs on the CPU. The Keras backend is chosen as usual: the ``KERAS_BACKEND``
environment variable if set, else the ``"backend"`` entry in ``~/.keras/keras.json``, else TensorFlow. The first line
of the output names the backend that was used.

Examples::

    uv run python scripts/benchmark_grid_loss.py --mode loss
    uv run python scripts/benchmark_grid_loss.py
    uv run python scripts/benchmark_grid_loss.py --compile
    uv run python scripts/benchmark_grid_loss.py --units 20000 --batch-sizes 100 500

"""

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

METHODS = ("broadcast", "pairwise")

# The repository's source folder, which is put first on the import path of this process and of every run it spawns
SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def _use_repo_source() -> None:
    """Import prfmodel from this repository's source folder, ahead of any installed copy."""
    if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


def _peak_rss_mb() -> float:
    """Return the peak resident set size of this process so far, in MB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes
    return peak / 1024**2 if platform.system() == "Darwin" else peak / 1024


def _run_single(args: argparse.Namespace) -> dict:
    """Run one configuration in this process and return its timings and memory."""
    _use_repo_source()

    # Imported here so that the parent process, which only spawns runs, never loads a backend
    import keras  # noqa: PLC0415 (the backend is loaded per run)
    import numpy as np  # noqa: PLC0415 (only needed per run)
    import prfmodel  # noqa: PLC0415 (the backend is loaded per run)
    from keras import ops  # noqa: PLC0415 (the backend is loaded per run)
    from prfmodel.examples import load_2d_prf_bar_stimulus  # noqa: PLC0415 (the backend is loaded per run)
    from prfmodel.fitters import GridFitter  # noqa: PLC0415 (the backend is loaded per run)
    from prfmodel.fitters import grid_values_2d_prf  # noqa: PLC0415 (the backend is loaded per run)
    from prfmodel.fitters.losses import CorrelationLoss  # noqa: PLC0415 (the backend is loaded per run)
    from prfmodel.models.prf import Gaussian2DPRFModel  # noqa: PLC0415 (the backend is loaded per run)

    pairwise = args.method == "pairwise"
    rng = np.random.default_rng(0)
    stimulus = load_2d_prf_bar_stimulus()
    num_frames = stimulus.design.shape[0]

    if args.mode == "loss":
        loss = CorrelationLoss(pairwise=pairwise, reduction="none", dtype=args.dtype)
        y_true = ops.convert_to_tensor(rng.standard_normal((1, args.units, num_frames)), dtype=args.dtype)
        y_pred = ops.convert_to_tensor(rng.standard_normal((args.batch_size, 1, num_frames)), dtype=args.dtype)

        # The loss sees a single batch, so the batch is the whole grid
        num_candidates = args.batch_size

        def work() -> None:
            # Converting to NumPy waits for the result, which matters on backends that run asynchronously
            ops.convert_to_numpy(loss(y_true, y_pred))

    else:
        model = Gaussian2DPRFModel(impulse_model=None, scaling_model=None)
        loss = None if pairwise else CorrelationLoss(reduction="none")
        fitter = GridFitter(model=model, stimulus=stimulus, loss=loss, compile_step=args.compile, dtype=args.dtype)
        data = rng.standard_normal((args.units, num_frames)).astype(args.dtype)
        # The grid a user gets from the package's helper, which only covers the pRF parameters this model has
        grid = grid_values_2d_prf(stimulus, num_mu=args.num_mu, num_sigma=args.num_sigma)
        num_candidates = int(np.prod([len(values) for values in grid.values()]))

        def work() -> None:
            fitter.fit(data, grid, batch_size=args.batch_size)

    rss_after_setup = _peak_rss_mb()

    # The first run includes one-off costs (tracing, compilation, allocator warm-up) and is reported separately
    start = time.perf_counter()
    work()
    first = time.perf_counter() - start

    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        work()
        times.append(time.perf_counter() - start)

    return {
        "backend": keras.backend.backend(),
        "prfmodel_path": str(Path(prfmodel.__file__).parent),
        "mode": args.mode,
        "method": args.method,
        "units": args.units,
        "frames": num_frames,
        "batch_size": args.batch_size,
        "grid_candidates": num_candidates,
        "first_s": first,
        "median_s": statistics.median(times),
        "extra_peak_mb": _peak_rss_mb() - rss_after_setup,
    }


def _spawn(args: argparse.Namespace, method: str, batch_size: int) -> dict:
    """Run one configuration in a fresh process and return its result, or the error it failed with."""
    # The same interpreter as this process, so the run uses whatever environment 'uv run' (or plain python) chose
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single",
        "--method",
        method,
        "--batch-sizes",
        str(batch_size),
        "--mode",
        args.mode,
        "--units",
        str(args.units),
        "--num-mu",
        str(args.num_mu),
        "--num-sigma",
        str(args.num_sigma),
        "--repeats",
        str(args.repeats),
        "--dtype",
        args.dtype,
    ]
    if args.compile:
        cmd.append("--compile")

    # Hide GPUs from every backend and silence progress bars and TensorFlow's start-up logs
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "TQDM_DISABLE": "1"}
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)  # noqa: S603 (own script)
    if proc.returncode != 0:
        # A run that exhausts memory is usually killed without a traceback, so fall back to the exit code
        stderr = proc.stderr.strip()
        reason = stderr.splitlines()[-1] if stderr else f"exit code {proc.returncode}"
        return {"method": method, "batch_size": batch_size, "error": reason}

    return json.loads(proc.stdout.strip().splitlines()[-1])


def _print_table(results: list[dict]) -> None:
    """Write the results as an aligned table to standard output."""
    header = f"{'method':<10} {'batch':>6} {'first (s)':>10} {'median (s)':>11} {'extra peak (MB)':>16}"
    lines = [header, "-" * len(header)]
    for res in results:
        if "error" in res:
            lines.append(f"{res['method']:<10} {res['batch_size']:>6}  FAILED: {res['error']}")
        else:
            lines.append(
                f"{res['method']:<10} {res['batch_size']:>6} {res['first_s']:>10.3f} {res['median_s']:>11.3f} "
                f"{res['extra_peak_mb']:>16.0f}",
            )
    sys.stdout.write("\n".join(lines) + "\n")


def main() -> None:
    """Parse arguments and run the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("fit", "loss"), default="fit")
    parser.add_argument("--units", type=int, default=2000, help="number of voxels U")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[500, 1000, 2000],
        help="grid candidates per batch G; one run per value",
    )
    # The defaults of 'grid_values_2d_prf': 21 x 21 x 20 = 8820 grid candidates
    parser.add_argument("--num-mu", type=int, default=21, help="values of mu_x and of mu_y (fit mode)")
    parser.add_argument("--num-sigma", type=int, default=20, help="values of sigma (fit mode)")
    parser.add_argument("--repeats", type=int, default=3, help="timed runs after the first one")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--compile", action="store_true", help="use compile_step=True (fit mode)")
    parser.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single:
        args.batch_size = args.batch_sizes[0]
        sys.stdout.write(json.dumps(_run_single(args)) + "\n")
        return

    results = [_spawn(args, method, batch_size) for batch_size in args.batch_sizes for method in METHODS]

    first_ok = next((res for res in results if "error" not in res), None)
    if first_ok is not None:
        sys.stdout.write(
            f"backend={first_ok['backend']} mode={args.mode} units={args.units} frames={first_ok['frames']} "
            f"dtype={args.dtype} compile={args.compile}\n",
        )
        if args.mode == "fit":
            sys.stdout.write(f"grid candidates={first_ok['grid_candidates']}\n")
        sys.stdout.write(
            f"prfmodel imported from {first_ok['prfmodel_path']}\n",
        )
    _print_table(results)


if __name__ == "__main__":
    main()
