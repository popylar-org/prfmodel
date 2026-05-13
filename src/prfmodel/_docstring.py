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
    "model_cf": ("cf_model : BasePopulationResponse\n    A connective field response model instance."),
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
        "    instance with default values."
    ),
    "model_prf": ("prf_model : BasePopulationResponse\n    A population receptive field response model instance."),
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
    "predicted_response_2d": (
        "Tensor\n    The predicted model response with shape `(num_units, num_frames)` and dtype `dtype`."
    ),
    "regressors": (
        "regressors : pandas.DataFrame\n"
        "    Regressor design with shape `(num_frames, num_regressors)`. Must contain a column for each name in\n"
        "    :attr:`names`; extra columns are ignored."
    ),
    "regressors_canonical": (
        "regressors : pandas.DataFrame, optional\n"
        "    Regressor design data. Required when the canonical model has a regressors model configured.\n"
        "    A single data frame with shape `(num_frames, num_regressors)` whose columns cover the names\n"
        "    required by every configured regressor model. Extra columns are ignored."
    ),
    "stimulus": ("stimulus : Stimulus\n    Stimulus object."),
    "stimulus_cf": ("stimulus : CFStimulus\n    Connective field stimulus object."),
    "stimulus_prf": ("stimulus : PRFStimulus\n    Population receptive field stimulus object."),
}

doc = Substitution(**_PARAMS)
