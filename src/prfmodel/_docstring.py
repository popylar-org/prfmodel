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
