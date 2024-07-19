# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from enum import Enum
from functools import partial, update_wrapper

import numpy as np

import raytrace2d.env as env


class Events(Enum):
    SURFACE_REFLECTION = 0
    BOTTOM_REFLECTION = 1
    MAX_BOUNDS_REACHED = 2
    ZERO_DEPTH_REACHED = 3
    MAX_TIME_REACHED = 4


class Event(ABC):
    """Abstract base class for events that terminate the ray trace. Requires
    the `__call__` method to be implemented with signature including `s`
    and `y`. The `s` parameter is the current value of the independent
    variable, and `y` is the current state of the system. Additional
    parameters can be passed in as `*args` and `**kwargs`.
    """

    @abstractmethod
    def __call__(self, s, y, z_ref, x_ref, *args, **kwargs): ...


class BottomReflection(Event):
    def __init__(self) -> None:
        self.terminal = True
        self.direction = -1

    def __call__(
        self, s, y, z_ref: float, x_ref: float, bathymetry: env.Bathymetry
    ) -> float:
        return self._bottom_reflection(y, z_ref, x_ref, bathymetry)

    @staticmethod
    def _bottom_reflection(
        y, z_ref: float, x_ref: float, bathymetry: env.Bathymetry
    ) -> float:
        return bathymetry(y[0] + x_ref) - (y[2] + z_ref)


class MaxBoundsReached(Event):
    def __init__(self) -> None:
        self.terminal = True
        self.direction = -1

    def __call__(
        self, s, y, z_ref: float, x_ref: float, bathymetry: env.Bathymetry
    ) -> float:
        return self._max_bounds_reached(y, x_ref, bathymetry)

    @staticmethod
    def _max_bounds_reached(y, x_ref: float, bathymetry: env.Bathymetry) -> float:
        return np.min(
            [
                y[0] + x_ref - bathymetry.distance[0],
                bathymetry.distance[-1] - (y[0] + x_ref),
            ]
        )


class MaxTimeReached(Event):
    def __init__(self) -> None:
        self.terminal = True
        self.direction = -1

    def __call__(
        self, s, y, z_ref: float, x_ref: float, tau_ref: float, t_max: float = 10.0
    ) -> float:
        return self._max_time_reached(y, tau_ref, t_max)

    @staticmethod
    def _max_time_reached(y, tau_ref: float, t_max: float = 1e10) -> float:
        return t_max - (y[4] + tau_ref)


class SurfaceReflection(Event):
    def __init__(self) -> None:
        self.terminal = True
        self.direction = -1

    def __call__(self, s, y, z_ref: float, x_ref: float) -> float:
        return self._surface_reflection(y, z_ref)

    @staticmethod
    def _surface_reflection(y, z_ref: float) -> float:
        return y[2] + z_ref - env.SeaSurface.LEVEL


class ZeroDepthReached(Event):
    def __init__(self) -> None:
        self.terminal = True
        self.direction = -1

    def __call__(
        self, s, y, z_ref: float, x_ref: float, bathymetry: env.Bathymetry
    ) -> float:
        return self._zero_depth(y, x_ref, bathymetry)

    @staticmethod
    def _zero_depth(y, x_ref: float, bathymetry: env.Bathymetry) -> float:
        return bathymetry(y[0] + x_ref)


class PartialWithAttributes(partial):
    def __new__(cls, func, *args, **kwargs):
        obj = super().__new__(cls, func, *args, **kwargs)
        update_wrapper(obj, func)
        obj.__dict__.update(func.__dict__)
        return obj
