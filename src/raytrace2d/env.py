#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import raytrace2d.plotting as rplt


class DimensionMismatchException(Exception):
    pass


class Interpolator:
    def __init__(
        self, points: np.ndarray, values: np.ndarray, method: str = "linear"
    ) -> None:
        self.interp = RegularGridInterpolator(
            points, values, method=method, bounds_error=False, fill_value=np.nan
        )
        self.nearest = RegularGridInterpolator(
            points, values, method="nearest", bounds_error=False, fill_value=None
        )

    def __call__(
        self,
        zi: Union[float, np.ndarray],
        xi: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        if xi is not None:
            zi = np.atleast_1d(zi)
            xi = np.atleast_1d(xi)
            Z, X = np.meshgrid(zi, xi, indexing="ij")
            vals = self.interp((Z, X))
            idxs = np.isnan(vals)
            vals[idxs] = self.nearest((Z[idxs], X[idxs]))
            return vals.squeeze()

        zi = np.atleast_1d(zi)
        vals = self.interp(zi)
        idxs = np.isnan(vals)
        vals[idxs] = self.nearest(zi[idxs])
        return vals.squeeze()


@dataclass
class Bathymetry:
    water_depth: Union[float, np.ndarray]  # Water depth [m]
    distance: Union[float, np.ndarray] = 0.0  # Range [m]

    def __call__(self, distance: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self._query(distance)

    def __post_init__(self):
        self.water_depth = np.atleast_1d(self.water_depth)
        self._distance_full_like_depth()
        self._set_interpolator()
        self._compute_gradient()

    def _distance_full_like_depth(self):
        if type(self.distance) is float or len(self.distance) == 1:
            self.distance = np.full_like(self.water_depth, self.distance)

    def _compute_gradient(self):
        if self.water_depth.shape[0] == 1:
            self.dwdx = np.zeros_like(self.water_depth)
        else:
            self.dwdx = np.gradient(self.water_depth, self.distance)
        self.dwdx_interpolator = Interpolator(points=(self.distance,), values=self.dwdx)

    def normal_vector(
        self, distance: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        dwdx = self.dwdx_interpolator(distance)
        grad_unit_vec = np.array([1.0, dwdx])
        grad_unit_vec /= np.linalg.norm(grad_unit_vec)
        if dwdx == 0.0:
            return np.array([0.0, -1.0])
        if dwdx < 0.0:
            rotation_matrix = np.array([[0, -1], [1, 0]])
        if dwdx > 0.0:
            rotation_matrix = np.array([[0, 1], [-1, 0]])
        return np.dot(rotation_matrix, grad_unit_vec)

    def _set_interpolator(self):
        self._interpolator = Interpolator(
            points=(self.distance,), values=self.water_depth
        )

    def _query(self, distance: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self._interpolator(distance)


@dataclass(frozen=True)
class SeaSurface:
    LEVEL: float = 0.0  # Sea surface level [m]
    NORMAL: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0])
    )  # Unit vector normal to the sea surface

    def __call__(self):
        return self.LEVEL, self.NORMAL


@dataclass
class SoundSpeedProfile:
    depth: np.ndarray  # Depth [m]
    speed: Union[float, np.ndarray]  # Sound speed [m/s]
    resample: bool = True  # Resample the profile
    dz: float = 1.0  # Depth resolution [m]
    distance: Union[float, np.ndarray] = 0.0  # Range [m]

    def __call__(
        self, depth: Union[float, np.ndarray], distance: Union[float, np.ndarray] = 0.0
    ) -> np.ndarray:
        return self._query(depth, distance)

    def __post_init__(self) -> None:
        self._speed_full_like_depth()
        self._check_input_dims()
        if self.resample:
            self._resample_depth()
        self.speed = np.atleast_2d(self.speed).T
        self.distance = np.atleast_1d(self.distance)
        self._speed_full_like_distance()
        self._compute_slowness()
        self._set_interpolator()
        self._compute_slowness_gradient()

    def _check_input_dims(self) -> None:
        # TODO: This needs to check additional cases.
        if len(self.depth) != len(self.speed):
            raise DimensionMismatchException(
                f"Depth and speed must have the same number of elements in the `z` "
                f"direction. len(depth)={len(self.depth)}, len(speed)={len(self.speed)}"
            )
        # TODO: Make a case to check distance dims.

    def _compute_slowness(self) -> None:
        self.slowness = 1 / self.speed

    def _compute_slowness_gradient(self) -> None:
        self.dsdz = np.gradient(self.slowness, self.depth, axis=0)

        if self.slowness.shape[1] == 1:
            self.dsdx = np.zeros_like(self.slowness)
        else:
            self.dsdx = np.gradient(self.slowness, self.distance, axis=1)

        self.dsdz_interpolator = Interpolator(
            points=(self.depth, self.distance), values=self.dsdz
        )
        self.dsdx_interpolator = Interpolator(
            points=(self.depth, self.distance), values=self.dsdx
        )

    def plot(self, ax: Optional[plt.Axes] = None, *args, **kwargs) -> plt.Axes:
        return rplt.plot_ssp(self.speed, self.depth, ax=ax, *args, **kwargs)

    def _set_interpolator(self) -> None:
        self._interpolator = Interpolator(
            points=(
                self.depth,
                self.distance,
            ),
            values=self.slowness,
        )

    def slowness_gradient(
        self, depth: Union[float, np.ndarray], distance: Union[float, np.ndarray]
    ) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        return self.dsdz_interpolator(depth, distance), self.dsdx_interpolator(
            depth, distance
        )

    def _speed_full_like_depth(self) -> None:
        if type(self.speed) is float or len(self.speed) == 1:
            self.speed = np.full_like(self.depth, self.speed)

    def _speed_full_like_distance(self) -> None:
        if len(self.distance) > 1 and self.depth.ndim == 1:
            self.speed = np.tile(self.speed, len(self.distance))

    def _query(
        self, depth: Union[float, np.ndarray], distance: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        return self._interpolator(depth, distance)

    def _resample_depth(self) -> None:
        new_depth = np.arange(0.0, self.depth.max() + self.dz, self.dz)
        new_speed = np.interp(new_depth, self.depth, self.speed)
        self.depth = new_depth
        self.speed = new_speed


@dataclass
class Source:
    depth: float  # Depth [m]
    distance: float = 0.0  # Range [m]


@dataclass
class Receiver:
    depth: float  # Depth [m]
    distance: float  # Range [m]
