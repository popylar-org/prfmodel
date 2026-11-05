# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `ParameterTransform` and `ParameterConstraint` work with dataframes when the PyTorch backend runs on an accelerator
  (e.g., Apple MPS). A `keras.ops` transformation such as `ops.log` returned a device tensor that pandas could not
  store (`TypeError: can't convert mps:0 device type tensor to numpy`). `ParameterTransform` now converts the result
  to NumPy, and `TensorFrame.to_dataframe` converts its tensors to NumPy as well.
- `ParameterTransform` computes the transformation of dataframe columns in the requested `dtype` (`SGDFitter`
  passes its own), instead of ignoring it. Before, float64 columns (the pandas default) failed on devices without
  float64 support, such as Apple MPS.
- Compiling the fitting step (`compile_step=True`) with the PyTorch backend works on Apple MPS. Inductor's layout
  optimization is disabled, because it failed when a fit recompiled the step for a batch of a different size
  (`TypeError: '>' not supported between instances of 'Node' and 'int'`, torch 2.15). On MPS, the number of buffers
  per fused kernel is also capped below Metal's limit of 31, which kernels could exceed
  (`number of constant buffers exceeds maximum supported (31)`).

## [0.2.0] - 2026-11-03

### Added

- Example dataset `7t-ncsf-visual` containing the surface response of a single subject to a contrast sensitivity
  stimulus (`task="csf"`, with `"odd"` and `"even"` splits for cross-validation) and to a moving bar stimulus
  (`task="prf"`), on the HCP fs_LR 59k mesh, with Benson et al. (2014) template visual areas, polar angle,
  eccentricity, and pRF size (Roelofzen et al., 2025). Its contrast sensitivity stimulus ships with the package.
- `load_dataset` no longer requires a `split` for datasets that have a default split; `describe_dataset`
  reports the default split

### Changed

- `plot_csf_stimulus_design` labels its y-axis "Contrast (%)" instead of "Contrast".

### Fixed

- `CSFTuning` and `CSFModel` no longer produce NaN predictions or NaN gradients in float32 for high contrasts, low
  contrast sensitivities, or steep `slope_crf`. The Naka-Rushton contrast response function is now evaluated in the
  log domain as a sigmoid, and `CSFTuning` no longer computes the contrast sensitivity itself, only its logarithm.
  Predictions that were finite before are unchanged up to floating-point rounding, except for blank frames (see
  below). The `eps` of `CSFTuning` now bounds the sensitivity, as in `predict_contrast_response`.
- `predict_contrast_response` evaluates the Naka-Rushton contrast response function in the same way. Its `eps` now
  bounds the sensitivity instead of the threshold `100 / sensitivity`, which changes results only for sensitivities
  below `eps` or above `100 / eps`.
- Blank frames respond with exactly zero in `predict_contrast_response`, `CSFTuning`, and `CSFModel`, instead of
  `eps^q / (eps^q + Q^q)`, which is far from zero for a shallow `slope_crf`. In `CSFTuning` and `CSFModel`, blank
  frames can have zero spatial frequency (as well as zero contrast) without making the gradients NaN.
- `plot_csf_stimulus_design` draws each spatial frequency at the frames that show it, instead of at consecutive
  frames from the first one, which was wrong when a frequency is shown in more than one block.
- `plot_csf_stimulus_design` and `plot_csf_stimulus_curve` leave out blank frames, which `plot_csf_stimulus_curve`
  divided by, and `plot_csf_stimulus_curve` draws the curve in order of spatial frequency instead of in frame order.
- The documentation of `predict_contrast_response` stated the Naka-Rushton threshold as `Q = 1 / S(f)` instead of
  `Q = 100 / S(f)`.
- The documentation now states that CSF models expect contrast as Michelson contrast in percent (0 to 100), on
  `CSFStimulus`, `predict_contrast_response`, `CSFTuning`, and in the CSF tutorial. The examples of `CSFStimulus` and
  `CSFModel` used contrast as a fraction, and the tutorial gave the contrast unit as cycles per degree.

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

[unreleased]: https://github.com/popylar-org/prfmodel/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/popylar-org/prfmodel/releases/tag/v0.2.0
[0.1.0]: https://github.com/popylar-org/prfmodel/releases/tag/v0.1.0
