"""Module for 2D ray tracing."""

from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
import logging
from typing import Protocol

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from tqdm import tqdm

import raytrace2d.env as env
import raytrace2d.events as events
from raytrace2d.events import PartialWithAttributes as partial
import raytrace2d.plotting as rplt

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


class PathPhase(StrEnum):
    D = "d"
    B = "b"
    S = "s"
    BS = "bs"
    SB = "sb"
    BB = "bb"
    SS = "ss"


@dataclass
class SoundSpeedProfile(Protocol):
    def query(self, position: float | np.ndarray) -> float | np.ndarray: ...


@dataclass
class Reflection:
    depth: float
    distance: float
    slowness: float
    angle_i: float
    angle_r: float
    interface: str


@dataclass
class Ray:
    x: np.ndarray
    z: np.ndarray
    s: np.ndarray
    tau: np.ndarray
    tang: np.ndarray
    slw: np.ndarray
    dslwdx: np.ndarray
    dslwdz: np.ndarray
    launch_angle: float
    reflections: list[Reflection] | None = None

    def plot(self, ax: Axes | None = None, *args, **kwargs) -> Axes:
        return rplt.plot_ray(self.x, self.z, ax=ax, *args, **kwargs)

    def plot_slowness(self, ax: Axes | None = None, *args, **kwargs) -> Axes:
        return rplt.plot_ray(self.x, self.slw, ax=ax, *args, **kwargs)

    def plot_slowness_xgrad(self, ax: Axes | None = None, *args, **kwargs) -> Axes:
        return rplt.plot_ray(self.x, self.dslwdx, ax=ax, *args, **kwargs)

    def plot_slowness_zgrad(self, ax: Axes | None = None, *args, **kwargs) -> Axes:
        return rplt.plot_ray(self.x, self.dslwdz, ax=ax, *args, **kwargs)

    @property
    def num_reflections(self) -> int:
        return len(self.reflections) if self.reflections is not None else 0

    @property
    def path_phase(self) -> str:
        if self.num_reflections == 0:
            return PathPhase.D
        phase_path = ""
        for reflection in self.reflections:
            phase_path += reflection.interface
        return phase_path


@dataclass(kw_only=True)
class Eigenray(Ray):
    depth_error: float


class RayTrace:

    def __init__(
        self,
        profile: SoundSpeedProfile,
        bathymetry: env.Bathymetry,
        source: env.Source,
        receiver: env.Receiver | None = None,
    ):
        self.profile: SoundSpeedProfile = profile
        self.bathymetry: env.Bathymetry = bathymetry
        self.source: env.Source = source
        self.receiver: env.Receiver = receiver
        self.rays: list[Ray] = []
        self.eigenrays: list[Eigenray] = []

    @staticmethod
    def _depth_difference(ray: Ray, receiver: env.Receiver) -> float:
        ind = np.argmin(np.abs(ray.x - receiver.distance))
        return ray.z[ind] - receiver.depth

    def _eikonal(self, s, y, z_ref, x_ref):
        # recall: y = [x, dxi/ds, z, dzeta/ds, tau]
        y = np.array(y).squeeze()
        tmp_x = y[0] + x_ref
        tmp_z = y[2] + z_ref
        slw = self.profile(tmp_z, tmp_x)
        dslwdz, dslwdx = self.profile.slowness_gradient(tmp_z, tmp_x)
        return np.array([y[1] / slw, dslwdx, y[3] / slw, dslwdz, slw])

    @staticmethod
    def _filter_rays(rays: list[Ray], ptype: str) -> list[Ray]:
        return [ray for ray in rays if ray.path_phase == ptype]

    def _intersect_and_reorder_rays(
        self, rays: list[Ray]
    ) -> tuple[list[Ray], int | None, int | None]:
        sorted_ind, last_negative_index, first_positive_index = self._get_intersection(
            rays, self.receiver
        )
        rays = [rays[i] for i in sorted_ind]
        return rays, last_negative_index, first_positive_index

    def find_eigenrays(
        self,
        xtol: float = 2e-12,
        rtol: float = np.float64(8.881784197001252e-16),
        maxiter: int = 100,
    ) -> list[Eigenray]:
        pbar_kw = {
            "total": len(PathPhase),
            "desc": "Finding eigenrays",
            "unit": "ray",
            "bar_format": "{l_bar}{bar:20}{r_bar}{bar:-20b}",
        }
        with ProcessPoolExecutor(max_workers=7) as executor:
            eigenrays = list(
                tqdm(
                    executor.map(
                        self._find_eigenrays_by_ptype,
                        [p for p in PathPhase],
                        [xtol] * len(PathPhase),
                        [rtol] * len(PathPhase),
                        [maxiter] * len(PathPhase),
                    ),
                    **pbar_kw,
                )
            )
        # eigenrays = [
        #     self._find_eigenrays_by_ptype(
        #         ptype.value, xtol=xtol, rtol=rtol, maxiter=maxiter
        #     )
        #     for ptype in tqdm(PathPhase, **pbar_kw)
        # ]

        if all(ray is None for ray in eigenrays):
            logging.debug("Eigenrays not found.")
            return None
        return [ray for ray in eigenrays if ray is not None]

    def _find_eigenrays_by_ptype(
        self,
        ptype: str,
        xtol: float = 2e-12,
        rtol: float = np.float64(8.881784197001252e-16),
        maxiter: int = 100,
    ) -> Eigenray | None:
        logging.debug(
            "=" * 80 + f"\nSearching for eigenray with path type `{ptype}`..."
        )
        rays = self._filter_rays(self.rays, ptype)
        if len(rays) == 0:
            return
        num_rays = 8
        launch_angle = rays[0].launch_angle
        lower_angle_limit = -10.0
        upper_angle_limit = 10.0
        failure = False
        attempt = 0
        while len(rays) <= 1 and not failure:
            logging.debug(f"{ptype}: Need to shoot more rays...")
            new_angles = np.linspace(
                max(launch_angle + lower_angle_limit, -85.0),
                min(launch_angle + upper_angle_limit, 85.0),
                num_rays,
            )
            logging.debug(f"New angles: {new_angles}")
            rays = self._filter_rays(self.ray_trace(new_angles, num_workers=8), ptype)
            num_rays *= 2
            lower_angle_limit *= 1.5
            upper_angle_limit *= 1.5

            attempt += 1
            if attempt > 2:
                failure = True

        if failure:
            logging.debug(f"Eigenray not found for path type `{ptype}`.")
            return

        rays, last_negative_index, first_positive_index = (
            self._intersect_and_reorder_rays(rays)
        )

        logging.debug(
            f"Last (-) index: {last_negative_index} | First (+) index: {first_positive_index}"
        )
        new_angles = np.array([0, 0])
        num_rays = 8
        if last_negative_index is None:
            angle_lim = rays[first_positive_index].launch_angle
            logging.debug(f"All rays are below the receiver. Angle = {angle_lim}")
        if first_positive_index is None:
            angle_lim = rays[last_negative_index].launch_angle
            logging.debug(f"All rays are above the receiver. Angle = {angle_lim}")
        attempt = 0
        while last_negative_index is None:
            if angle_lim <= -85.0:
                break
            new_angles = np.linspace(
                max(angle_lim - 15.0, -85.0),
                angle_lim,
                num_rays,
            )
            logging.debug(f"New angles: {new_angles[0]}, {new_angles[-1]}")
            rays = self._filter_rays(self.ray_trace(new_angles, num_workers=8), ptype)
            if len(rays) == 0:
                logging.debug(f"No rays found. Terminating search.")
                break
            if len(rays) == 1 and attempt < 2:
                logging.debug(f"{len(rays)} found; incrementing angles.")
                angle_lim = new_angles[0]
                attempt += 1
                continue
            rays, last_negative_index, first_positive_index = (
                self._intersect_and_reorder_rays(rays)
            )
            angle_lim = rays[first_positive_index].launch_angle
            logging.debug(
                f"First (+) index: {first_positive_index} | Last (-) index: {last_negative_index}"
            )
            if last_negative_index is None:
                logging.debug(f"All rays are below the receiver. Angle = {angle_lim}")

        while first_positive_index is None:
            if angle_lim >= 85.0:
                break
            new_angles = np.linspace(
                angle_lim,
                min(angle_lim + 15.0, 85.0),
                num_rays,
            )
            logging.debug(f"New angles: {new_angles[0]}, {new_angles[-1]}")
            rays = self._filter_rays(self.ray_trace(new_angles, num_workers=8), ptype)
            if len(rays) == 0:
                logging.debug(f"No rays found. Terminating search.")
                break
            if len(rays) == 1 and attempt < 2:
                logging.debug(f"{len(rays)} found; incrementing angles.")
                angle_lim = new_angles[-1]
                attempt += 1
                continue
            rays, last_negative_index, first_positive_index = (
                self._intersect_and_reorder_rays(rays)
            )
            angle_lim = rays[last_negative_index].launch_angle
            logging.debug(
                f"First (+) index: {first_positive_index} | Last (-) index: {last_negative_index}"
            )
            if first_positive_index is None:
                logging.debug(f"All rays are above the receiver. Angle = {angle_lim}")

        if last_negative_index is None or first_positive_index is None:
            logging.debug(f"Eigenray not found for path type `{ptype}`.")
            return

        lower_angle = rays[last_negative_index].launch_angle
        upper_angle = rays[first_positive_index].launch_angle
        logging.debug(
            f"Last (-) index: {last_negative_index} | First (+) index: {first_positive_index}"
        )
        logging.debug(f"Launch angle bounds: {lower_angle}, {upper_angle}")

        er_launch = self._find_roots(
            (lower_angle, upper_angle), xtol=xtol, rtol=rtol, maxiter=maxiter
        )
        ray = self.trace_ray(er_launch)
        depth_diff = self._depth_difference(ray, self.receiver)
        logging.debug(f"Eigenray found: {er_launch}")
        return Eigenray(
            x=ray.x,
            z=ray.z,
            s=ray.s,
            tau=ray.tau,
            tang=ray.tang,
            slw=ray.slw,
            dslwdx=ray.dslwdx,
            dslwdz=ray.dslwdz,
            launch_angle=er_launch,
            reflections=ray.reflections,
            depth_error=depth_diff,
        )

    @staticmethod
    def _find_event(t_events: list[np.ndarray | None]) -> tuple[int, float]:
        min_t = 0.0
        for aid, t_event in enumerate(t_events):
            if len(t_event) > 0 and t_event[0] > min_t:
                event_id = aid
                min_t = t_event[0]
        return event_id, min_t

    @staticmethod
    def _find_indices(array: np.ndarray) -> tuple[int | None, int | None]:
        # Ensure the array is a NumPy array
        array = np.array(array)

        # Find the index of the first positive number
        first_positive_index = np.searchsorted(array, 0, side="right")

        # Find the index of the last negative number
        last_negative_index = first_positive_index - 1

        # If the array has no negative numbers, handle the edge case
        if last_negative_index < 0 or array[last_negative_index] >= 0:
            last_negative_index = None

        # If the array has no positive numbers, handle the edge case
        if first_positive_index >= len(array) or array[first_positive_index] <= 0:
            first_positive_index = None

        return last_negative_index, first_positive_index

    def _find_roots(
        self,
        angle_bounds: tuple[float],
        xtol: float = 2e-12,
        rtol: float = np.float64(8.881784197001252e-16),
        maxiter: int = 100,
    ) -> float:
        def f(angle: float) -> float:
            logging.debug(f"Trying angle {angle}...")
            ray = self.trace_ray(angle)
            return self._depth_difference(ray, self.receiver)

        logging.debug(
            f"Objective function output: {f(angle_bounds[0])}, {f(angle_bounds[1])}"
        )
        return brentq(f, *angle_bounds, xtol=xtol, rtol=rtol, maxiter=maxiter)

    def _get_intersection(
        self, rays: list[Ray], receiver: env.Receiver
    ) -> tuple[int, int]:
        zind = np.empty((len(rays),), dtype=int)
        ray_depth = np.empty((len(rays),), dtype=float)
        for i, ray in enumerate(rays):
            ind = np.argmin(np.abs(ray.x - receiver.distance))
            zind[i] = ind
            ray_depth[i] = ray.z[ind]
            logging.debug(
                f"Ray depth = {ray_depth[i]}, launch angle = {ray.launch_angle}"
            )

        sorted_ind = np.argsort(ray_depth)
        sorted_ray_depth = ray_depth[sorted_ind]
        logging.debug(
            f"Ray depth: {sorted_ray_depth}, Receiver depth: {receiver.depth}"
        )
        last_negative_index, first_positive_index = self._find_indices(
            sorted_ray_depth - receiver.depth
        )
        return sorted_ind, last_negative_index, first_positive_index

    @staticmethod
    def _get_reflection(
        z_ref, x_ref, s_ref, min_t, event_y, normal_vec: np.ndarray, interface: str
    ) -> Reflection:
        return Reflection(
            depth=float(event_y[2] + z_ref),
            distance=float(event_y[0] + x_ref),
            slowness=float(min_t + s_ref),
            angle_i=float(np.arctan2(event_y[3], event_y[1])),
            angle_r=float(np.arctan2(normal_vec[1], normal_vec[0])),
            interface=interface,
        )

    def _initialize_events(self, tau_ref: float = 0.0) -> list[events.Event]:
        return [
            events.SurfaceReflection(),
            partial(events.BottomReflection(), bathymetry=self.bathymetry),
            partial(events.MaxBoundsReached(), bathymetry=self.bathymetry),
            partial(events.ZeroDepthReached(), bathymetry=self.bathymetry),
            partial(events.MaxTimeReached(), tau_ref=tau_ref, t_max=1e10),
        ]

    def _initialize_launch_vector(self, angle: float) -> tuple[float, float]:
        # Slowness at source
        slw0 = self.profile(self.source.depth, self.source.distance)
        # Change in x in the direction of the ray
        dxds0 = np.cos(angle * DEG2RAD)
        # Change in z in the direction of the ray
        dzds0 = np.sin(angle * DEG2RAD)
        xi0 = dxds0 * slw0
        zeta0 = dzds0 * slw0
        return xi0, zeta0

    def _initialize_references(self) -> tuple[float, float, float, float]:
        return self.source.depth, self.source.distance, 0.0, 0.0

    def plot(self, only_eig: bool = False) -> Figure:
        return rplt.plot_ray_trace(self, only_eig)

    def plot_eigenrays(self, ax: Axes | None = None, *args, **kwargs) -> Axes:
        ax = rplt.plot_rays(self.eigenrays, ax=ax, *args, **kwargs)
        ax.plot(self.receiver.distance, self.receiver.depth, "rx")
        return ax

    def plot_rays(self, ax: Axes | None = None, *args, **kwargs) -> Axes:
        return rplt.plot_rays(self.rays, ax=ax, *args, **kwargs)

    def ray_trace(
        self,
        angles: float | Iterable[float],
        ds: float = 10.0,
        max_bottom_bounce: int = 9999,
        num_workers: int = 16,
        display: bool = False,
    ) -> list[Ray]:
        pbar_kw = {
            "total": len(angles),
            "desc": "Tracing rays",
            "unit": "ray",
            "bar_format": "{l_bar}{bar:20}{r_bar}{bar:-20b}",
        }
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            return list(
                tqdm(
                    executor.map(
                        self.trace_ray,
                        angles,
                        [ds] * len(angles),
                        [max_bottom_bounce] * len(angles),
                    ),
                    disable=not display,
                    **pbar_kw,
                )
            )

    @staticmethod
    def _reflect(ei: np.ndarray, normal: np.ndarray) -> np.ndarray:
        return ei - 2 * np.dot(ei, normal) * normal

    def run(
        self,
        angles: float | Iterable[float] = np.linspace(-80, 80, 41),
        ds: float = 10.0,
        max_bottom_bounce: int = 9999,
        num_workers: int = 16,
        eigenrays: bool = False,
        xtol: float = 2e-12,
        rtol: float = np.float64(8.881784197001252e-16),
        maxiter: int = 100,
    ) -> tuple[list[Ray], list[Eigenray] | None]:
        if type(angles) is float:
            angles = [angles]

        self.rays = self.ray_trace(
            angles, ds, max_bottom_bounce, num_workers, display=True
        )
        if eigenrays and self.receiver is None:
            raise ValueError("Receiver must be defined to find eigenrays.")
        if eigenrays:
            self.eigenrays = self.find_eigenrays(xtol=xtol, rtol=rtol, maxiter=maxiter)
            return self.rays, self.eigenrays
        return self.rays, None

    def trace_ray(
        self,
        angle: float,
        abstol: float = 1e-9,
        reltol: float = 1e-6,
        ds: float = 10.0,
        max_bottom_bounce: int = 9999,
    ) -> Ray:

        xi0, zeta0 = self._initialize_launch_vector(angle)
        z_ref, x_ref, tau_ref, s_ref = self._initialize_references()
        ivp_events = self._initialize_events(tau_ref)

        z0 = z_ref
        x0 = x_ref
        tau0 = tau_ref
        s0 = s_ref

        x = np.array([])  # x-pos
        z = np.array([])  # z-pos
        tau = np.array([])  # travel time
        s = np.array([])  # arc length
        tang = np.array([])  # ray tangent angle
        num_btm_bnc = 0

        reflections = []
        while True:
            # recall: y = [x, dxi/ds, z, dzeta/ds, tau]
            y0 = np.array([x0 - x_ref, xi0, z0 - z_ref, zeta0, tau0 - tau_ref]).T

            sol = solve_ivp(
                lambda s, y, z_ref, x_ref: self._eikonal(s, y, z_ref, x_ref),
                [s0 - s_ref, np.inf],
                y0,
                method="RK45",
                dense_output=True,
                events=ivp_events,
                args=(z_ref, x_ref),
                rtol=reltol,
                atol=abstol,
                max_step=ds,
            )

            event_id, min_t = self._find_event(sol.t_events)

            t_arr = np.arange(0.0, sol.t[-1] + ds, ds)
            t_arr = t_arr[t_arr < min_t]  # Exclude termination point
            y = sol.sol(t_arr).T

            x = np.append(x, y[:, 0] + x_ref)
            z = np.append(z, y[:, 2] + z_ref)
            tau = np.append(tau, y[:, 4] + tau_ref)
            s = np.append(s, t_arr + s_ref)
            tang = np.append(tang, np.arctan2(y[:, 3], y[:, 1]))

            event_y = sol.y_events[event_id].squeeze()

            if event_id == events.Events.SURFACE_REFLECTION.value:
                normal = env.SeaSurface().NORMAL
                reflections.append(
                    self._get_reflection(
                        z_ref, x_ref, s_ref, min_t, event_y, normal, PathPhase.S
                    )
                )
            if event_id == events.Events.BOTTOM_REFLECTION.value:
                num_btm_bnc += 1
                normal = self.bathymetry.normal_vector(distance=x_ref)
                reflections.append(
                    self._get_reflection(
                        z_ref,
                        x_ref,
                        s_ref,
                        min_t,
                        event_y,
                        normal,
                        PathPhase.B,
                    )
                )
                if num_btm_bnc >= max_bottom_bounce:
                    break
            if (
                event_id == events.Events.MAX_BOUNDS_REACHED.value
                or event_id == events.Events.ZERO_DEPTH_REACHED.value
                or event_id == events.Events.MAX_TIME_REACHED.value
            ):

                x = np.append(x, event_y[0] + x_ref)
                z = np.append(z, event_y[2] + z_ref)
                s = np.append(s, min_t + s_ref)
                tau = np.append(tau, event_y[4] + tau_ref)
                tang = np.append(tang, np.arctan2(event_y[3], event_y[1]))

                break

            x0 = event_y[0] + x_ref
            xi0 = event_y[1]
            z0 = event_y[2] + z_ref
            zeta0 = event_y[3]
            s0 = min_t + s_ref
            tau0 = event_y[4] + tau_ref

            ei = np.array([xi0, zeta0])
            er = self._reflect(ei, normal)

            xi0 = er[0]
            zeta0 = er[1]

            x_ref = x0
            z_ref = z0
            s_ref = s0
            tau_ref = tau0

        return Ray(
            x=x,
            z=z,
            s=s,
            tau=tau,
            tang=tang,
            slw=self.profile(x, z),
            dslwdx=np.array(
                [self.profile.slowness_gradient(ze, xe)[1] for ze, xe in zip(z, x)]
            ),
            dslwdz=np.array(
                [self.profile.slowness_gradient(ze, xe)[0] for ze, xe in zip(z, x)]
            ),
            launch_angle=angle,
            reflections=reflections,
        )
