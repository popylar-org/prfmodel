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
    "model_cf": ("cf_model : BaseResponse\n    A connective field response model instance."),
    "model_encoding": (
        "encoding_model : BaseEncoder or type, default=PRFStimulusEncoder\n"
        "    An encoding model class or instance. Model classes will be instantiated during initialization. The\n"
        "    default creates a :class:`~prfmodel.models.encoding.PRFStimulusEncoder` instance."
    ),
    "model_fitter": (
        "model : BaseModel\n"
        "    A model instance that can be fit to data.\n"
        "    The model must implement the :meth:`__call__` method to make predictions that can be\n"
        "    compared to data."
    ),
    "model_impulse": (
        "impulse_model : BaseImpulse or type or None, default=DerivativeTwoGammaImpulse, optional\n"
        "    An impulse response model class or instance. Model classes will be instantiated during\n"
        "    initialization. The default creates a :class:`~prfmodel.models.impulse.DerivativeTwoGammaImpulse`\n"
        "    instance with default values."
    ),
    "model_prf": ("prf_model : BaseResponse\n    A population receptive field response model instance."),
    "model_temporal": (
        "temporal_model : BaseTemporal or type or None, default=BaselineAmplitude, optional\n"
        "    A temporal model class or instance. Model classes will be instantiated during initialization.\n"
        "    The default creates a :class:`~prfmodel.models.temporal.BaselineAmplitude` instance."
    ),
    "parameters": (
        "parameters : pandas.DataFrame\n"
        "    Dataframe with columns containing different model parameters and rows containing parameter values\n"
        "    for different units."
    ),
    "predicted_response_2d": (
        "Tensor\n    The predicted model response with shape `(num_units, num_frames)` and dtype `dtype`."
    ),
    "stimulus": ("stimulus : Stimulus\n    Stimulus object."),
    "stimulus_cf": ("stimulus : CFStimulus\n    Connective field stimulus object."),
    "stimulus_prf": ("stimulus : PRFStimulus\n    Population receptive field stimulus object."),
}

doc = Substitution(**_PARAMS)
