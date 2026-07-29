"""2D density-slice visualization for Gaussian molecular shapes.

These helpers render a planar cross-section of one or two ``GaussianShape``
density fields. Comparing two shapes as overlaid contours gives an intuitive
picture of how well their volumes coincide.
"""

from __future__ import annotations

import numpy as np

from core.gaussian_shape import GaussianShape


def _make_grid(shapes, axis: str, offset: float, padding: float, resolution: int):
    """Build a 2D grid of query points on a plane orthogonal to ``axis``.

    Returns the meshgrid coordinates (for plotting) and the (M, 3) array of
    3D query points (for density evaluation).
    """
    all_centers = np.vstack([s.centers for s in shapes])
    max_sigma = max(float(s.sigmas.max()) for s in shapes)
    lo = all_centers.min(axis=0) - padding - 2.0 * max_sigma
    hi = all_centers.max(axis=0) + padding + 2.0 * max_sigma

    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    plane_axes = [i for i in range(3) if i != axis_index]
    a0, a1 = plane_axes

    u = np.linspace(lo[a0], hi[a0], resolution)
    v = np.linspace(lo[a1], hi[a1], resolution)
    grid_u, grid_v = np.meshgrid(u, v)

    points = np.zeros((grid_u.size, 3))
    points[:, a0] = grid_u.ravel()
    points[:, a1] = grid_v.ravel()
    points[:, axis_index] = offset
    return grid_u, grid_v, points, (a0, a1)


_AXIS_LABELS = {0: "x", 1: "y", 2: "z"}


def plot_density_slice(
    shape: GaussianShape,
    axis: str = "z",
    offset: float | None = None,
    padding: float = 2.0,
    resolution: int = 120,
    ax=None,
    title: str = "Gaussian shape density",
):
    """Plot a filled-contour density slice of a single shape.

    Parameters
    ----------
    shape : GaussianShape
    axis : {"x", "y", "z"}
        Normal direction of the slicing plane.
    offset : float, optional
        Coordinate of the plane along ``axis``. Defaults to the shape centroid.
    padding : float
        Extra margin (Angstrom) added around the atoms.
    resolution : int
        Number of grid points per axis.
    ax : matplotlib Axes, optional
    title : str
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    if offset is None:
        offset = float(shape.centroid()[{"x": 0, "y": 1, "z": 2}[axis]])

    grid_u, grid_v, points, (a0, a1) = _make_grid(
        [shape], axis, offset, padding, resolution
    )
    density = shape.density(points).reshape(grid_u.shape)

    contour = ax.contourf(grid_u, grid_v, density, levels=20, cmap="viridis")
    ax.figure.colorbar(contour, ax=ax, label="density")
    ax.set_xlabel(f"{_AXIS_LABELS[a0]} (Angstrom)")
    ax.set_ylabel(f"{_AXIS_LABELS[a1]} (Angstrom)")
    ax.set_title(title)
    ax.set_aspect("equal")
    return ax


def plot_shape_overlay(
    shape_a: GaussianShape,
    shape_b: GaussianShape,
    axis: str = "z",
    offset: float | None = None,
    padding: float = 2.0,
    resolution: int = 120,
    ax=None,
    labels=("A", "B"),
    title: str = "Shape overlay",
):
    """Overlay two shape densities as contour lines on the same plane.

    Shape A is drawn in reds, shape B in blues; regions where both are strong
    indicate good shape overlap.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    if offset is None:
        # Slice through the midpoint of the two centroids.
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        offset = float(0.5 * (shape_a.centroid()[idx] + shape_b.centroid()[idx]))

    grid_u, grid_v, points, (a0, a1) = _make_grid(
        [shape_a, shape_b], axis, offset, padding, resolution
    )
    dens_a = shape_a.density(points).reshape(grid_u.shape)
    dens_b = shape_b.density(points).reshape(grid_u.shape)

    ax.contour(grid_u, grid_v, dens_a, levels=6, cmap="Reds")
    ax.contour(grid_u, grid_v, dens_b, levels=6, cmap="Blues")

    # Proxy artists for the legend (ContourSet has no legend handle by itself).
    from matplotlib.lines import Line2D

    proxies = [
        Line2D([0], [0], color="tab:red", label=labels[0]),
        Line2D([0], [0], color="tab:blue", label=labels[1]),
    ]

    ax.set_xlabel(f"{_AXIS_LABELS[a0]} (Angstrom)")
    ax.set_ylabel(f"{_AXIS_LABELS[a1]} (Angstrom)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(handles=proxies, loc="upper right")
    return ax
