# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-26

### Changed

- **Breaking:** `PRFStimulus`, `CSFStimulus`, and `PRFStimulus.create_2d_bar_stimulus` have a new required keyword-only
  argument `frame_duration`: the duration of one design frame in seconds (e.g., the TR). `PRFStimulusTensors` and
  `CSFStimulusTensors` have a matching `frame_duration` field.
- **Breaking:** The `resolution` of impulse models (`TwoGammaImpulse`, `DerivativeTwoGammaImpulse`,
  `ShiftedGammaImpulse`, `SustainedImpulse`, `TransientImpulse`, and subclasses of `BaseImpulse`) defaults to `None`
  instead of `1.0`, and `BaseImpulse.resolution` can return `None`.
  - Inside a canonical model, an impulse model without a `resolution` is sampled at the `frame_duration` of the
    stimulus. Used on its own, it is sampled every second (`prfmodel.impulse.base.DEFAULT_RESOLUTION`).
  - Previously, impulse responses were sampled every second regardless of the stimulus. Predictions of canonical
    models with a default impulse model therefore change for every frame duration other than one second.
  - An explicit `resolution` is never overridden. If it differs from the `frame_duration` of the stimulus, canonical
    models and the `GridFitter`, `LeastSquaresFitter`, and `SGDFitter` constructors issue a
    `ResolutionMismatchWarning`, which is now defined in `prfmodel.exceptions`.
  - To migrate, pass the frame duration to the stimulus and drop a manually set `resolution`:

    ```python
    # 0.0.1
    stimulus = PRFStimulus(design=design, grid=grid)
    model = Gaussian2DPRFModel(impulse_model=DerivativeTwoGammaImpulse(resolution=TR))

    # 0.1.0
    stimulus = PRFStimulus(design=design, grid=grid, frame_duration=TR)
    model = Gaussian2DPRFModel()
    ```

- **Breaking:** `GridFitter.fit`, `LeastSquaresFitter.fit`, and `SGDFitter.fit` raise a `ShapeError` if `data` does not
  have the shape `(num_units, num_frames)`, with one row per unit in the parameters and one column per frame of the
  model prediction.
- **Breaking:** The fitters raise a `ValueError` for unknown parameter names and suggest the closest valid name. This
  applies to the keys of `parameter_values` in `GridFitter.fit`, to `slope_name` in `LeastSquaresFitter.fit`, and to
  `fixed_parameters` in `SGDFitter.fit`. Previously, unknown names were ignored.
- **Breaking:** The fitters exclude units whose time series is constant or contains NaN or infinite values and issue an
  `InvalidUnitsWarning`. The estimated parameters and losses of these units are NaN.
- **Breaking:** The values in `SGDHistory.history` are Python floats instead of backend tensors once `SGDFitter.fit`
  returns.
- **Breaking:** Importing `prfmodel` raises an `ImportError` if Keras is configured with a backend other than JAX,
  TensorFlow, or PyTorch, or if the configured backend is not installed. The error explains how to select or install a
  supported backend.

### Removed

- **Breaking:** The `gain` parameter of `CompressiveEncoder` and `Gaussian2DCSSPRFModel`, and the `gain` argument of
  `init_css_from_gaussian`. The gain traded off exactly against the `amplitude` of the scaling model, so it could not be
  identified in a fit; the CSS gain of Kay et al. (2013) is now the `amplitude` of the scaling model.
  - To migrate, remove `gain` from parameter data frames, `fixed_parameters`, and grid `parameter_values`. With the
    default `BaselineAmplitude` scaling model, a prediction from 0.0.1 is reproduced with `amplitude` set to
    `gain * amplitude`.

[unreleased]: https://github.com/popylar-org/prfmodel/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/popylar-org/prfmodel/releases/tag/v0.1.0
