# -*- coding: utf-8 -*-

from typing import Optional, Protocol

import matplotlib.pyplot as plt
import numpy as np


class Ray(Protocol):
    def plot(self): ...


class RayTrace(Protocol):
    def plot(self): ...


class SoundSpeedProfile(Protocol):
    def plot(self): ...


def plot_ray(
    distance: np.ndarray,
    depth: np.ndarray,
    ax: Optional[plt.Axes] = None,
    *args,
    **kwargs
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    ax.plot(distance, depth, *args, **kwargs)
    return ax


def plot_rays(
    rays: list[Ray], ax: Optional[plt.Axes] = None, *args, **kwargs
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    for ray in rays:
        ax = ray.plot(ax=ax, *args, **kwargs)
    return ax


def plot_ssp(
    speed: np.ndarray, depth: np.ndarray, ax: Optional[plt.Axes] = None, *args, **kwargs
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    ax.plot(speed, depth, *args, **kwargs)
    return ax


# TODO: Plot all of the above in a single figure
