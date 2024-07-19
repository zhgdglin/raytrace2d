# -*- coding: utf-8 -*-

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Optional, Protocol, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import raytrace2d.env as env
import raytrace2d.events as events
from raytrace2d.events import PartialWithAttributes as partial
import raytrace2d.plotting as rplt

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


@dataclass
class SoundSpeedProfile(Protocol):
    def query(self, position: Union[float, np.ndarray]) -> Union[float, np.ndarray]: ...


@dataclass
class Reflection:
    depth: float
    distance: float
    slowness: float
    angle_i: float
    angle_r: float


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
    reflections: Optional[list[Reflection]] = None

    def plot(self, ax: Optional[plt.Axes] = None, *args, **kwargs) -> plt.Axes:
        return rplt.plot_ray(self.x, self.z, ax=ax, *args, **kwargs)


class RayTrace:

    def __init__(
        self,
        profile: SoundSpeedProfile,
        bathymetry: env.Bathymetry,
        source: env.Source,
        receiver: env.Receiver = Optional[None],
    ):
        self.profile: SoundSpeedProfile = profile
        self.bathymetry: env.Bathymetry = bathymetry
        self.source: env.Source = source
        self.receiver: env.Receiver = receiver
        self.reflections: list[Union[None, Reflection]] = []
        self.rays: list[Ray] = []

    def _eikonal(self, s, y, z_ref, x_ref):
        # recall: y = [x, dxi/ds, z, dzeta/ds, tau]
        y = np.array(y).squeeze()
        tmp_x = y[0] + x_ref
        tmp_z = y[2] + z_ref
        slw = self.profile(tmp_x, tmp_z)
        dslwdz, dslwdx = self.profile.slowness_gradient(tmp_x, tmp_z)
        return np.array([y[1] / slw, dslwdx, y[3] / slw, dslwdz, slw]).T

    def find_eigenrays(self):
        return

    @staticmethod
    def _find_event(t_events: list[Union[None, np.ndarray]]) -> tuple[int, float]:
        min_t = 0.0
        for aid, t_event in enumerate(t_events):
            if len(t_event) > 0 and t_event[0] > min_t:
                event_id = aid
                min_t = t_event[0]
        return event_id, min_t

    @staticmethod
    def _get_reflection(
        z_ref, x_ref, s_ref, min_t, event_y, normal_vec: np.ndarray
    ) -> Reflection:
        return Reflection(
            depth=float(event_y[2] + z_ref),
            distance=float(event_y[0] + x_ref),
            slowness=float(min_t + s_ref),
            angle_i=float(np.arctan2(event_y[3], event_y[1])),
            angle_r=float(np.arctan2(normal_vec[1], normal_vec[0])),
        )

    def _initialize_reference(self) -> tuple[float, float, float, float]:
        return self.source.depth, self.source.distance, 0.0, 0.0

    def plot(self) -> plt.Axes:
        return rplt.plot_ray_trace(self)
    
    def plot_rays(self, ax: Optional[plt.Axes] = None, *args, **kwargs) -> plt.Axes:
        return rplt.plot_rays(self.rays, ax=ax, *args, **kwargs)

    @staticmethod
    def _reflect(ei: np.ndarray, normal: np.ndarray) -> np.ndarray:
        return ei - 2 * np.dot(ei, normal) * normal

    def run(
        self,
        angles: Union[float, Iterable[float]],
        ds: float = 10.0,
        max_bottom_bounce: int = 9999,
        num_workers: int = 16,
    ) -> list[Ray]:
        if type(angles) is float:
            angles = [angles]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            self.rays = list(
                executor.map(
                    self.trace_ray,
                    angles,
                    [ds] * len(angles),
                    [max_bottom_bounce] * len(angles),
                )
            )
        return self.rays

    def trace_ray(
        self,
        angle: float,
        abstol: float = 1e-9,
        reltol: float = 1e-6,
        ds: float = 10.0,
        max_bottom_bounce: int = 9999,
    ) -> Ray:
        slw0 = self.profile(self.source.depth, self.source.distance)
        dxds0 = np.cos(angle * DEG2RAD)
        dzds0 = np.sin(angle * DEG2RAD)
        xi0 = dxds0 * slw0
        zeta0 = dzds0 * slw0

        # Initialize events
        z_ref, x_ref, tau_ref, s_ref = self._initialize_reference()

        ivp_events = [
            events.SurfaceReflection(),
            partial(events.BottomReflection(), bathymetry=self.bathymetry),
            partial(events.MaxBoundsReached(), bathymetry=self.bathymetry),
            partial(events.ZeroDepthReached(), bathymetry=self.bathymetry),
            partial(events.MaxTimeReached(), tau_ref=tau_ref, t_max=1e10),
        ]

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
        t_events = []
        yevents = []

        while True:
            # recall: y = [x, dxi/ds, z, dzeta/ds, tau]
            y0 = np.array([x0 - x_ref, xi0, z0 - z_ref, zeta0, tau0 - tau_ref])
            yevents.append(y0)

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

            t_events.append(min_t)
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
                self.reflections.append(
                    self._get_reflection(z_ref, x_ref, s_ref, min_t, event_y, normal)
                )
            if event_id == events.Events.BOTTOM_REFLECTION.value:
                num_btm_bnc += 1
                normal = self.bathymetry.normal_vector(distance=x_ref)
                self.reflections.append(
                    self._get_reflection(
                        z_ref,
                        x_ref,
                        s_ref,
                        min_t,
                        event_y,
                        normal,
                    )
                )
                if num_btm_bnc >= max_bottom_bounce:
                    break
            if (
                event_id == events.Events.MAX_BOUNDS_REACHED.value
                or event_id == events.Events.ZERO_DEPTH_REACHED.value
                or event_id == events.Events.MAX_TIME_REACHED.value
            ):

                z = np.append(z, event_y[2] + z_ref)
                x = np.append(x, event_y[0] + x_ref)
                s = np.append(s, min_t + s_ref)
                tau = np.append(tau, event_y[4] + tau_ref)
                tang = np.append(tang, np.arctan2(event_y[3], event_y[1]))

                break

            z0 = event_y[2] + z_ref
            zeta0 = event_y[3]
            x0 = event_y[0] + x_ref
            xi0 = event_y[1]
            s0 = min_t + s_ref
            tau0 = event_y[4] + tau_ref

            ei = np.array([xi0, zeta0])
            er = self._reflect(ei, normal)

            xi0 = er[0]
            zeta0 = er[1]

            z_ref = z0
            x_ref = x0
            s_ref = s0
            tau_ref = tau0

        return Ray(
            x=x,
            z=z,
            s=s,
            tau=tau,
            tang=tang,
            slw=self.profile(x, z),
            dslwdx=self.profile.slowness_gradient(x, z)[1],
            dslwdz=self.profile.slowness_gradient(x, z)[0],
            reflections=self.reflections,
        )
