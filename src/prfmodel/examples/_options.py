"""Validated options for loading an example dataset."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prfmodel.examples._registry import DatasetSpec

_HEMISPHERE_CODES = {
    "both": ("L", "R"),
    "left": ("L",),
    "right": ("R",),
}


@dataclass(frozen=True)
class Options:
    """
    Validated options that a dataset loader is called with.

    Parameters
    ----------
    hemisphere : str
        Hemisphere selection, one of `"both"`, `"left"`, or `"right"`.
    split : str or None
        Name of the requested data split, or `None` for datasets without splits.
    surface_type : str or None
        Requested surface type, or `None` for datasets without surfaces.

    """

    hemisphere: str = "both"
    split: str | None = None
    surface_type: str | None = None

    @property
    def hemisphere_codes(self) -> tuple[str, ...]:
        """Single-letter codes of the requested hemispheres, with the left hemisphere first."""
        return _HEMISPHERE_CODES[self.hemisphere]


def _check_supported(spec: DatasetSpec, name: str, value: object) -> None:
    if value is not None and name not in spec.options:
        msg = (
            f"Dataset '{spec.name}' does not accept the option '{name}'. It accepts "
            f"{tuple(sorted(spec.options)) if spec.options else 'no options'}."
        )
        raise ValueError(msg)


def _check_value(spec: DatasetSpec, name: str, value: str, valid: tuple[str, ...]) -> None:
    if value not in valid:
        msg = f"Argument '{name}' must be one of {valid} for dataset '{spec.name}' but was '{value}'"
        raise ValueError(msg)


def validate_options(
    spec: DatasetSpec,
    hemisphere: str | None = None,
    split: str | None = None,
    surface_type: str | None = None,
) -> Options:
    """
    Check the requested options against a dataset and fill in its defaults.

    Parameters
    ----------
    spec : prfmodel.examples._registry.DatasetSpec
        Specification of the dataset that the options are for.
    hemisphere : str, optional
        Requested hemisphere selection.
    split : str, optional
        Requested data split.
    surface_type : str, optional
        Requested surface type. The value `"pial"` is accepted as an alias for `"pia"`.

    Returns
    -------
    Options
        The validated options.

    Raises
    ------
    ValueError
        If an option is not accepted by the dataset, if its value is not valid, or if the dataset requires a
        split and none was requested.

    """
    _check_supported(spec, "hemisphere", hemisphere)
    _check_supported(spec, "split", split)
    _check_supported(spec, "surface_type", surface_type)

    resolved_hemisphere = hemisphere if hemisphere is not None else spec.default_hemisphere

    if "hemisphere" in spec.options:
        _check_value(spec, "hemisphere", resolved_hemisphere, spec.hemispheres)

    if spec.splits:
        if split is None:
            msg = f"Dataset '{spec.name}' requires the option 'split' to be one of {spec.splits}"
            raise ValueError(msg)

        _check_value(spec, "split", split, spec.splits)

    resolved_surface_type = surface_type

    if "surface_type" in spec.options:
        # The surfaces are stored under 'pia' but 'pial' is the more common name
        resolved_surface_type = "pia" if surface_type == "pial" else (surface_type or spec.surface_types[0])
        _check_value(spec, "surface_type", resolved_surface_type, spec.surface_types)

    return Options(
        hemisphere=resolved_hemisphere,
        split=split,
        surface_type=resolved_surface_type,
    )
