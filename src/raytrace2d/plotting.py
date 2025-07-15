from typing import Optional, Protocol

import matplotlib.pyplot as plt
import numpy as np


class Ray(Protocol):
    def plot(self): ...


class RayTrace(Protocol):
    def plot(self): ...


def plot_bathymetry(
    distance: np.ndarray,
    water_depth: np.ndarray,
    ax: Optional[plt.Axes] = None,
    *args,
    **kwargs
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    ax.plot(distance, water_depth, *args, **kwargs)
    return ax


def plot_ray(
    dep_var: np.ndarray,
    ind_var: np.ndarray,
    ax: Optional[plt.Axes] = None,
    *args,
    **kwargs
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    ax.plot(dep_var, ind_var, *args, **kwargs)
    return ax


def plot_ray_trace(raytrace: RayTrace) -> plt.Figure:
    FIG_KW = {
        "figsize": (12, 4),
        "nrows": 1,
        "ncols": 2,
        "gridspec_kw": {"width_ratios": [1, 4], "wspace": 0.0},
    }
    SSP_KW = {
        "c": "tab:blue",
    }
    RAY_KW = {
        "c": "tab:blue",
    }
    SRC_KW = {
        "c": "tab:red",
        "marker": "*",
    }
    RX_KW = {
        "c": "tab:red",
        "marker": "o",
    }
    BATHY_KW = {
        "c": "k",
    }
    max_depth = raytrace.bathymetry.water_depth.max()
    zvec = np.linspace(0, max_depth, 1001)

    fig, axs = plt.subplots(**FIG_KW)

    ax = axs[0]
    ax = plot_ssp(1 / raytrace.profile(zvec), zvec, ax=ax, **SSP_KW)
    ax.set_ylim(0, max_depth)
    ax.invert_yaxis()
    ax.set_xlabel("Speed [m/s]")
    ax.set_ylabel("Depth [m]")
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    ax = axs[1]
    ax = plot_rays(raytrace.rays, ax=ax, **RAY_KW, zorder=1)
    ax.plot(raytrace.source.distance, raytrace.source.depth, **SRC_KW)
    if raytrace.eigenrays:
        ax = plot_rays(raytrace.eigenrays, c="r", ax=ax, zorder=2)
        ax.plot(raytrace.receiver.distance, raytrace.receiver.depth, **RX_KW)

    ax = plot_bathymetry(
        raytrace.bathymetry.distance, raytrace.bathymetry.water_depth, ax=ax, **BATHY_KW
    )
    ax.set_xlim(-10, max(raytrace.bathymetry.distance) + 10)
    ax.set_ylim(0, max_depth)
    ax.invert_yaxis()
    ax.set_yticklabels([])
    ax.set_xlabel("Range [m]")

    return fig


def plot_rays(
    rays: list[Ray], ax: Optional[plt.Axes] = None, *args, **kwargs
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    for ray in rays:
        ax = ray.plot(ax=ax, *args, **kwargs)
    return ax


def plot_slowness(
    depth: np.ndarray,
    speed: np.ndarray,
    slowness: np.ndarray,
    dsdz: np.ndarray,
    dsdx: np.ndarray,
) -> plt.Figure:

    fig, axs = plt.subplots(
        figsize=(8, 6), nrows=1, ncols=4, gridspec_kw={"wspace": 0.2}
    )

    ax = axs[0]
    ax.plot(speed, depth)
    ax.set_ylim(0, depth.max())
    ax.invert_yaxis()
    ax.set_ylabel("Depth [m]")
    ax.set_title("Sound Speed [m/s]")

    ax = axs[1]
    ax.plot(slowness, depth)
    ax.set_ylim(0, depth.max())
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_title("Slowness [s/m]")

    ax = axs[2]
    ax.plot(dsdz, depth)
    ax.set_ylim(0, depth.max())
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_title("$ds/dz$ [s/m^2]")

    ax = axs[3]
    ax.plot(dsdx, depth)
    ax.set_ylim(0, depth.max())
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_title("$ds/dx$ [s/m^2]")

    return fig


def plot_ssp(
    speed: np.ndarray, depth: np.ndarray, ax: Optional[plt.Axes] = None, *args, **kwargs
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    ax.plot(speed, depth, *args, **kwargs)
    return ax
