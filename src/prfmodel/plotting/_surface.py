"""Surface plotting functions."""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Literal
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn.plotting import plot_surf_stat_map
from nilearn.surface import InMemoryMesh
from nilearn.surface import PolyMesh

if TYPE_CHECKING:
    from prfmodel.examples import Dataset


def _rotate_xy(coords: np.ndarray, angle: float) -> np.ndarray:
    """Rotate the x-y coordinates of a flat patch counter-clockwise by `angle` degrees around their centre."""
    angle = np.deg2rad(angle)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = coords.copy()
    centre = coords[:, :2].mean(axis=0)
    rotated[:, :2] = (coords[:, :2] - centre) @ rotation.T
    return rotated

def plot_surface_stat_map(  # noqa: PLR0913
    mesh: PolyMesh,
    stat_map: np.ndarray,
    hemisphere: Literal["left", "right", "both"] = "both",
    view: str | tuple[float, float] = (90, 270),
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "inferno",
    cyclic: bool = False,
    title: str | None = None,
    **kwargs,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    r"""Plot a statistic (e.g., a model parameter or score) on a cortical surface mesh.

    Wraps :func:`nilearn.plotting.plot_surf_stat_map` and enlarges the surface to fill the space next to the colorbar.

    Parameters
    ----------
    mesh : nilearn.surface.PolyMesh
        Surface mesh to plot on (e.g., a flat or inflated surface).
    stat_map : numpy.ndarray
        Statistic for each vertex of the mesh with shape `(num_vertices,)`. If `hemi="both"`, the values of the left
        hemisphere come first. `NaN` values are not shown.
    hemisphere : {"left", "right", "both"}, optional
        Hemisphere(s) to plot.
    view : str or tuple[float, float], optional
        View of the surface passed to :func:`nilearn.plotting.plot_surf_stat_map`. The default `(90, 270)` (elevation,
        azimuth) shows a flat surface from above.
    vmin : float or None, optional
        Lower limit of the color scale.
    vmax : float or None, optional
        Upper limit of the color scale.
    cmap : str, optional
        Name of the colormap.
    cyclic : bool, optional
        Whether the statistic is an angle in radians (e.g., pRF polar angle). If `True`, the color scale ranges
        from :math:`-\pi` to :math:`\pi` with the cyclic colormap `"hsv"` so that both ends have the same color,
        overriding `vmin`, `vmax`, and `cmap`.
    title : str or None, optional
        Title of the plot.
    **kwargs
        Keyword arguments passed to :func:`matplotlib.pyplot.subplots`. Defaults to `figsize=(8, 6)`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Three-dimensional axes object.

    See Also
    --------
    prfmodel.utils.calculate_polar_angle : Polar angle of two-dimensional pRF centers.
    prfmodel.utils.calculate_eccentricity : Eccentricity of two-dimensional pRF centers.

    """
    if cyclic:
        vmin, vmax, cmap = -np.pi, np.pi, "hsv"

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, **({"figsize": (8, 6)} | kwargs))

    plot_surf_stat_map(
        mesh,
        stat_map,
        hemi=hemisphere,
        view=view,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        axes=ax,
        figure=fig,
        title=title,
    )

    # Expand the 3D axes to fill the space to the left of the colorbar
    surf_axes = [a for a in fig.axes if a.name == "3d"]
    cbar_axes = [a for a in fig.axes if a.name != "3d"]

    if cbar_axes:
        cbar_x0 = min(a.get_position().x0 for a in cbar_axes)
        for a in surf_axes:
            a.set_position((0.0, 0.0, cbar_x0 - 0.01, 0.97))

    return fig, ax


def make_flat_mesh(
    dataset: Dataset,
    rotation: dict[str, float] | None = None,
    gap: float = 20.0,
) -> tuple[PolyMesh, np.ndarray]:
    r"""Build a flat surface mesh from the flat patches of a dataset to plot values on a flatmap.

    Loads the flat patch of each hemisphere in `dataset`, rotates it in the plane, and places the hemispheres side
    by side. A flat patch covers fewer vertices than the full cortical surface (e.g., the medial wall is cut away),
    so the function also returns which row of `dataset.response` each vertex of the flat mesh belongs to. Values
    with one entry per response row can then be plotted on the flat mesh with
    :func:`~prfmodel.plotting.plot_surface_stat_map`.

    Parameters
    ----------
    dataset : prfmodel.examples.Dataset
        Dataset with flat patches, such as `"numerosity-four-conditions"`. Its `files` must contain the flat
        surface (e.g., `"lh_flat"`) and the indices of the flat vertices on the full surface (e.g.,
        `"lh_vertex_indices"`) of each hemisphere in `dataset.mesh`.
    rotation : dict of str to float, optional
        Counter-clockwise rotation in degrees of the flat patch of each hemisphere, keyed by `"left"` and
        `"right"`. The orientation of flat patches is arbitrary, so the rotation that shows a patch upright depends
        on the dataset. Hemispheres without an entry are not rotated.
    gap : float, optional
        Distance between the flat patches of the left and the right hemisphere, in the units of the surface
        coordinates (usually mm).

    Returns
    -------
    flat_mesh : nilearn.surface.PolyMesh
        Flat surface mesh with the hemispheres of `dataset`, the left hemisphere on the left.
    flat_indices : numpy.ndarray
        Row of `dataset.response` for each vertex of `flat_mesh` with shape `(num_flat_vertices,)`, with the
        vertices of the left hemisphere first.

    See Also
    --------
    prfmodel.plotting.plot_surface_stat_map : Plot a statistic on a cortical surface mesh.
    prfmodel.examples.load_dataset : Load an example dataset.

    Examples
    --------
    Plot the response at one time frame on the flatmaps of both hemispheres.

    >>> from prfmodel.examples import load_dataset
    >>> from prfmodel.plotting import make_flat_mesh, plot_surface_stat_map
    >>> dataset = load_dataset("numerosity-four-conditions", task="average")  # doctest: +SKIP
    >>> flat_mesh, flat_indices = make_flat_mesh(dataset, rotation={"left": -100, "right": -80})  # doctest: +SKIP
    >>> fig, ax = plot_surface_stat_map(  # doctest: +SKIP
    ...     flat_mesh, dataset.response[flat_indices, 0], vmin=-3, vmax=3, cmap="RdBu_r",
    ... )
    """
    if dataset.mesh is None:
        msg = "Dataset has no surface mesh"
        raise ValueError(msg)

    rotation = rotation or {}
    parts, indices = {}, []
    row_offset = 0

    for hemi, prefix in (("left", "lh"), ("right", "rh")):
        if hemi not in dataset.mesh.parts:
            continue
        flat = nib.load(dataset.files[f"{prefix}_flat"])
        if not isinstance(flat, nib.gifti.GiftiImage):
            msg = f"Flat patch of the {hemi} hemisphere must be a GIFTI file, got {type(flat).__name__}"
            raise TypeError(msg)
        coords = _rotate_xy(np.asarray(flat.darrays[0].data, dtype=np.float64), rotation.get(hemi, 0.0))
        if hemi == "left":
            coords[:, 0] -= coords[:, 0].max() + gap / 2
        else:
            coords[:, 0] -= coords[:, 0].min() - gap / 2
        parts[hemi] = InMemoryMesh(coords, np.asarray(flat.darrays[1].data, dtype=np.int64))
        indices.append(np.load(dataset.files[f"{prefix}_vertex_indices"]) + row_offset)
        row_offset += dataset.mesh.parts[hemi].n_vertices

    return PolyMesh(**parts), np.concatenate(indices)


