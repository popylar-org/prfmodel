"""Surface plotting functions."""

from typing import Literal
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from nilearn.plotting import plot_surf_stat_map
from nilearn.surface import PolyMesh


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
