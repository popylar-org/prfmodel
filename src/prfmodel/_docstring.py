"""Docstring substitution decorator and shared parameter snippets."""

import re
from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F", bound=Callable)


class Substitution:
    """Decorator that substitutes %(key)s placeholders in docstrings.

    The indentation of each placeholder line is automatically prepended to the
    continuation lines of the substituted snippet, so snippets can be defined
    with minimal (4-space) indentation and work correctly regardless of the
    surrounding docstring indentation level.

    This approach is inspired by pandas, numpy, and scipy.

    """

    def __init__(self, **kwargs: str):
        self.params = kwargs

    def __call__(self, func: F) -> F:
        if func.__doc__:
            func.__doc__ = re.sub(
                r"^([ \t]*)%\((\w+)\)s",
                self._replace,
                func.__doc__,
                flags=re.MULTILINE,
            )
        return func

    def _replace(self, match: re.Match) -> str:
        indent = match.group(1)
        key = match.group(2)
        if key not in self.params:
            return match.group(0)
        lines = self.params[key].split("\n")
        return "\n".join([indent + line for line in lines])


_PARAMS: dict[str, str] = {
    "dtype": (
        "dtype : str, optional\n"
        "    The dtype of the prediction result. If `None` (the default), uses the dtype from\n"
        "    :func:`prfmodel.utils.get_dtype`."
    ),
    "frame_duration": (
        "frame_duration : float\n"
        "    The duration of one time frame (in seconds), e.g., the repetition time (TR) in fMRI. Must be given\n"
        "    as a keyword argument. Canonical models sample their impulse models at this interval unless the\n"
        "    impulse model has an explicit `resolution`. See :doc:`Important details </important_details>`."
    ),
    "gaussian_norm_note": (
        "This model uses a proper Gaussian density that is normalized by its volume.\n"
        "This differs from other Gaussian pRF model implementations. See\n"
        ":doc:`Important details </important_details>`."
    ),
    "impulse_default_parameters": (
        "default_parameters : dict of float or str, optional\n"
        "    Dictionary with scalar default parameter values or name of default parameter set.\n"
        "    Dictionary keys must be valid parameter names. Default values are overridden by user-supplied parameters\n"
        "    in the :meth:`__call__` method."
    ),
    "impulse_default_parameters_hrf": (
        'default_parameters : dict of float or str, optional, default="glover_hrf"\n'
        "    Dictionary with scalar default parameter values or name of default parameter set. Available default\n"
        "    parameter sets are `glover_hrf` (default) and `spm_hrf`. See :mod:`~prfmodel.impulse.defaults` for\n"
        "    details. Dictionary keys must be valid parameter names. Default values are overridden by user-supplied\n"
        "    parameters in the :meth:`__call__` method."
    ),
    "impulse_duration": ("duration : float, default=32.0\n    The duration of the impulse response (in seconds)."),
    "impulse_norm": (
        'norm : str, optional, default="sum"\n'
        '    The normalization of the response. Can be `"sum"` (default), `"mean"`, `"max"`, `"norm"`, or\n'
        "    `None`. If `None`, no normalization is performed. See :doc:`Important details </important_details>`."
    ),
    "impulse_offset": ("offset : float, default=0.0\n    The offset of the impulse response (in seconds)."),
    "impulse_resolution": (
        "resolution : float, optional\n"
        "    The time resolution of the impulse response (in seconds), that is the interval between two points at\n"
        "    which the impulse response function is evaluated. If `None` (the default), a canonical model samples the\n"
        "    impulse response at the frame duration of its stimulus, and the impulse response is sampled at\n"
        "    :data:`~prfmodel.impulse.base.DEFAULT_RESOLUTION` (1 second) when the impulse model is used on its own.\n"
        "    See :meth:`~prfmodel.impulse.base.BaseImpulse.at_resolution`."
    ),
    "model_cf": ("cf_model : BaseTuning\n    A connective field tuning model instance."),
    "model_csf": ("cfs_model : BaseTuning\n    A contrast sensitivity function tuning model instance."),
    "model_encoding_cf": (
        "encoding_model : BaseStimulusEncoder or type, default=CFStimulusEncoder\n"
        "    An stimulus encoding model class or instance. Model classes will be instantiated during initialization.\n"
        "    The default creates a :class:`~prfmodel.models.cf.CFStimulusEncoder` instance."
    ),
    "model_encoding_prf": (
        "encoding_model : BaseStimulusEncoder or type, default=PRFStimulusEncoder\n"
        "    An stimulus encoding model class or instance. Model classes will be instantiated during initialization.\n"
        "    The default creates a :class:`~prfmodel.models.prf.PRFStimulusEncoder` instance."
    ),
    "model_fitter": (
        "model : BaseCanonical\n"
        "    A model instance that can be fit to data.\n"
        "    The model must implement the :meth:`~prfmodel.models.base.BaseCanonical.__call__` method to make\n"
        "    predictions that can be compared to data."
    ),
    "model_impulse": (
        "impulse_model : BaseImpulse or type or None, default=DerivativeTwoGammaImpulse, optional\n"
        "    An impulse model class or instance. Model classes will be instantiated during\n"
        "    initialization. The default creates a :class:`~prfmodel.impulse.DerivativeTwoGammaImpulse`\n"
        "    instance with default values. Impulse models without an explicit `resolution` are sampled at the\n"
        "    frame duration of the stimulus."
    ),
    "model_prf": ("prf_model : BaseTuning\n    A population receptive field tuning model instance."),
    "model_scaling": (
        "scaling_model : BaseScaling or type or None, default=BaselineAmplitude, optional\n"
        "    A scaling model class or instance. Model classes will be instantiated during initialization.\n"
        "    The default creates a :class:`~prfmodel.scaling.BaselineAmplitude` instance."
    ),
    "model_regressors": (
        "regressors_model : BaseRegressors or list of BaseRegressors or RegressorsList or None, default=None, optional\n"  # noqa: E501 (line too long)
        "    A regressor model instance, a list of regressor model instances, or `None`. When a list is\n"
        "    provided, it is wrapped in a :class:`~prfmodel.regressors.RegressorsList` and its contributions\n"
        "    are summed. The regressor contribution is added after the scaling model."
    ),
    "parameters": (
        "parameters : pandas.DataFrame\n"
        "    Dataframe with columns containing different model parameters and rows containing parameter values\n"
        "    for different units."
    ),
    "parameters_tensors": (
        "parameters : TensorFrame\n"
        "    Model parameters as tensors, supporting the same column selection as a :class:`pandas.DataFrame`."
    ),
    "predicted_response_2d": (
        "Tensor\n    The predicted model response with shape `(num_units, num_frames)` and dtype `dtype`."
    ),
    "predicted_response_2d_array": (
        "numpy.ndarray\n    The predicted model response with shape `(num_units, num_frames)` and dtype `dtype`."
    ),
    "raises_missing_parameters": ("ValueError\n    If `parameters` is missing one or more of :attr:`parameter_names`."),
    "regressors": (
        "regressors : pandas.DataFrame\n"
        "    Regressor design with shape `(num_frames, num_regressors)`. Must contain a column for each name in\n"
        "    :attr:`names`; extra columns are ignored."
    ),
    "regressors_tensors": (
        "regressors : TensorFrame or None, optional\n"
        "    Regressor design columns as tensors, supporting the same column selection as a :class:`pandas.DataFrame`."
    ),
    "regressors_canonical": (
        "regressors : pandas.DataFrame, optional\n"
        "    Regressor design data. Required when the canonical model has a regressors model configured.\n"
        "    A single data frame with shape `(num_frames, num_regressors)` whose columns cover the names\n"
        "    required by every configured regressor model. Extra columns are ignored."
    ),
    "stimulus": ("stimulus : Stimulus\n    Stimulus object."),
    "stimulus_tensors": (
        "stimulus : StimulusTensors\n    The tensor-holding stimulus object, from :meth:`~prfmodel.stimuli.Stimulus.to_tensors`."  # noqa: E501 (line too long)
    ),
    "stimulus_cf": ("stimulus : CFStimulus\n    Connective field stimulus object."),
    "stimulus_cf_tensors": (
        "stimulus : CFStimulusTensors\n"
        "    Tensor-holding connective field stimulus object,\n"
        "    from :meth:`~prfmodel.stimuli.CFStimulus.to_tensors`."
    ),
    "stimulus_csf": ("stimulus : CSFStimulus\n    Contrast sensitivity function stimulus object."),
    "stimulus_csf_tensors": (
        "stimulus : CSFStimulusTensors\n"
        "    Tensor-holding contrast sensitivity function stimulus object,\n"
        "    from :meth:`~prfmodel.stimuli.CSFStimulus.to_tensors`."
    ),
    "stimulus_prf": ("stimulus : PRFStimulus\n    Population receptive field stimulus object."),
    "stimulus_prf_tensors": (
        "stimulus : PRFStimulusTensors\n"
        "    Tensor-holding population receptive field stimulus object,\n"
        "    from :meth:`~prfmodel.stimuli.PRFStimulus.to_tensors`."
    ),
    "warns_invalid_units": (
        "InvalidUnitsWarning\n"
        "    When the time series of some units in `data` are constant or contain NaN or infinite values. These\n"
        "    units are excluded from the fit and their estimated parameters and losses are NaN."
    ),
    "warns_resolution_mismatch": (
        "ResolutionMismatchWarning\n"
        "    When an impulse submodel has an explicit resolution that differs from the frame duration of the\n"
        "    stimulus. See :meth:`~prfmodel.models.base.BaseCanonical.check_stimulus`."
    ),
}

doc = Substitution(**_PARAMS)
