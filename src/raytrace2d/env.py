"""Module provides classes for the environment in which the ray trace occurs.
The environment includes the bathymetry, sea surface, sound speed profile,
source, and receiver.
"""

from dataclasses import dataclass, field
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import raytrace2d.plotting as rplt


class DimensionMismatchException(Exception):
    """Exception raised when the dimensions of the input data do not match."""

    pass


class Interpolator:
    """Interpolator class for 1D and 2D data.

    This class combines two `RegularGridInterpolator` objects, `self.interp`
    and `self.nearest`, to provide interpolation and nearest neighbor
    interpolation, respectively. This is to ensure that query points within
    the interpolation bounds are interpolated, while points outside the bounds
    are assigned the nearest neighbor value. If this is undesired behavior,
    more fully specified input data should be provided.

    Attributes:
        interp: Interpolator for the input data.
        nearest: Interpolator for the nearest neighbor data.
    """

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
        """Interpolate the input data at the given points.

        Args:
            zi: The z-coordinates of the points to interpolate.
            xi: The x-coordinates of the points to interpolate.

        Returns:
            The interpolated values at the given points.
        """
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
    """Bathymetry class for the ray trace simulation.

    Attributes:
        water_depth: Water depth [m].
        distance: Range [m].
    """

    water_depth: Union[float, np.ndarray]
    distance: Union[float, np.ndarray] = 0.0

    def __call__(self, distance: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Return the water depth at the given distance.

        Args:
            distance: Range [m].

        Returns:
            Water depth at the given distance [m].
        """
        return self._query(distance)

    def __post_init__(self):
        self.water_depth = np.atleast_1d(self.water_depth)
        self._distance_full_like_depth()
        self._set_interpolator()
        self._compute_gradient()

    def _distance_full_like_depth(self):
        """Ensure that the distance is full like the depth if different."""
        if type(self.distance) is float or len(self.distance) == 1:
            self.distance = np.full_like(self.water_depth, self.distance)

    def _compute_gradient(self):
        """Compute the gradient of the water depth."""
        if self.water_depth.shape[0] == 1:
            self.dwdx = np.zeros_like(self.water_depth)
        else:
            self.dwdx = np.gradient(self.water_depth, self.distance)
        self.dwdx_interpolator = Interpolator(points=(self.distance,), values=self.dwdx)

    def normal_vector(
        self, distance: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Return the normal vector to the bathymetry at the given distance.

        Args:
            distance: Range [m].

        Returns:
            Normal vector to the bathymetry at the given distance.
        """
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
        """Set the interpolator for the bathymetry."""
        self._interpolator = Interpolator(
            points=(self.distance,), values=self.water_depth
        )

    def _query(self, distance: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Return the water depth at the given distance.

        Args:
            distance: Range [m].

        Returns:
            Water depth at the given distance [m].
        """
        return self._interpolator(distance)


@dataclass
class Receiver:
    """Receiver class for the ray trace simulation.

    Attributes:
        depth: Receiver depth [m].
        distance: Receiver range [m].
    """

    depth: float
    distance: float


@dataclass(frozen=True)
class SeaSurface:
    """Sea surface class for the ray trace simulation.

    The attributes of the sea surface are immutable.

    Attributes:
        LEVEL: Sea surface level [m].
        NORMAL: Unit vector normal to the sea surface.
    """

    LEVEL: float = 0.0
    NORMAL: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))

    def __call__(self):
        """Return the sea surface level and normal vector.

        Returns:
            Sea surface level [m].
            Unit vector normal to the sea surface
        """
        return self.LEVEL, self.NORMAL


@dataclass
class SoundSpeedProfile:
    """Sound speed profile class for the ray trace simulation.

    As the ray tracing code uses slowness (inverse of sound speed), most
    of this class' methods are related to the computation of slowness and
    its gradient.

    Attributes:
        depth: Depth [m].
        speed: Sound speed [m/s].
        resample: Whether to resample the profile to a regular grid with
            resolution `dz` m.
        dz: Depth resolution [m].
        distance: Range [m].
    """

    depth: np.ndarray
    speed: Union[float, np.ndarray]
    resample: bool = True
    dz: float = 1.0
    distance: Union[float, np.ndarray] = 0.0

    def __call__(
        self, depth: Union[float, np.ndarray], distance: Union[float, np.ndarray] = 0.0
    ) -> np.ndarray:
        """Return the slowness at the given depth and distance.

        Args:
            depth: Depth [m].
            distance: Range [m].

        Returns:
            Slowness at the given depth and distance [s/m].
        """
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
        """Check the dimensions of the input data.

        Raises:
            DimensionMismatchException: If the dimensions of the input data do not match.
        """
        # TODO: This needs to check additional cases.
        if len(self.depth) != len(self.speed):
            raise DimensionMismatchException(
                f"Depth and speed must have the same number of elements in the `z` "
                f"direction. len(depth)={len(self.depth)}, len(speed)={len(self.speed)}"
            )
        # TODO: Make a case to check distance dims.

    def _compute_slowness(self) -> None:
        """Compute the slowness from the sound speed."""
        self.slowness = 1 / self.speed

    def _compute_slowness_gradient(self) -> None:
        """Compute the gradient of the slowness."""
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
        """Plot the sound speed profile.

        Args:
            ax: Matplotlib axes object.
            *args: Additional arguments to pass to the plot
            **kwargs: Additional keyword arguments to pass to the plot.

        Returns:
            Matplotlib axes object.
        """
        return rplt.plot_ssp(self.speed, self.depth, ax=ax, *args, **kwargs)
    
    def plot_slowness(self) -> plt.Figure:
        """Plot the slowness profile.

        Returns:
            Matplotlib figure.
        """
        # TODO: Handle multiple distances.
        return rplt.plot_slowness(
            depth=self.depth,
            speed=self.speed,
            slowness=self.slowness,
            dsdz=self.dsdz,
            dsdx=self.dsdx,
        )

    def _set_interpolator(self) -> None:
        """Set the interpolator for the slowness profile."""
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
        """Return the gradient of the slowness at the given depth and distance.

        Args:
            depth: Depth [m].
            distance: Range [m].

        Returns:
            The gradient of the slowness at the given depth and distance [s/m/m].
        """
        return self.dsdz_interpolator(depth, distance), self.dsdx_interpolator(
            depth, distance
        )

    def _speed_full_like_depth(self) -> None:
        """Ensure that the speed is full like the depth if different."""
        if type(self.speed) is float or len(self.speed) == 1:
            self.speed = np.full_like(self.depth, self.speed)

    def _speed_full_like_distance(self) -> None:
        """Ensure that the speed is full like the distance if different."""
        if len(self.distance) > 1 and self.depth.ndim == 1:
            self.speed = np.tile(self.speed, len(self.distance))

    def _query(
        self, depth: Union[float, np.ndarray], distance: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Return the slowness at the given depth and distance.

        Args:
            depth: Depth [m].
            distance: Range [m].

        Returns:
            Slowness at the given depth and distance [s/m].
        """
        return self._interpolator(depth, distance)

    def _resample_depth(self) -> None:
        """Resample the depth to a regular grid with resolution `dz`."""
        new_depth = np.arange(0.0, self.depth.max() + self.dz, self.dz)
        new_speed = np.interp(new_depth, self.depth, self.speed)
        self.depth = new_depth
        self.speed = new_speed


@dataclass
class Source:
    """Source class for the ray trace simulation.

    Attributes:
        depth: Source depth [m].
        distance: Source range [m].
    """

    depth: float
    distance: float = 0.0
