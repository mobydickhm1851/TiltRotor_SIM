"""Deterministic urban-wind scenarios and passenger-comfort proxy metrics.

The models in this module are deliberately transparent and inexpensive. They
support repeatable controller experiments; they are not certified atmospheric
or human-vibration standards.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import numpy as np


WIND_SCENARIO_CODES = {
    "off": 0,
    "steady": 1,
    "discrete_gust": 2,
    "wind_shear": 3,
    "urban_wake": 4,
}


@dataclass
class WindScenarioConfig:
    enabled: bool = False
    mode: str = "steady"
    base_wind_mps: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    disturbance_amplitude_mps: float = 5.0
    disturbance_direction_deg: float = 90.0
    start_time_s: float = 20.0
    duration_s: float = 8.0
    shear_gradient_per_m: float = 0.08
    reference_altitude_m: float = 30.0
    wake_frequency_hz: float = 0.35

    def normalized(self) -> "WindScenarioConfig":
        mode = self.mode if self.mode in WIND_SCENARIO_CODES else "steady"
        return WindScenarioConfig(
            enabled=bool(self.enabled),
            mode=mode,
            base_wind_mps=np.asarray(
                self.base_wind_mps, dtype=float
            ).reshape(3),
            disturbance_amplitude_mps=max(
                0.0, float(self.disturbance_amplitude_mps)
            ),
            disturbance_direction_deg=float(
                self.disturbance_direction_deg
            ),
            start_time_s=max(0.0, float(self.start_time_s)),
            duration_s=max(0.05, float(self.duration_s)),
            shear_gradient_per_m=max(
                0.0, float(self.shear_gradient_per_m)
            ),
            reference_altitude_m=float(self.reference_altitude_m),
            wake_frequency_hz=max(0.02, float(self.wake_frequency_hz)),
        )


@dataclass(frozen=True)
class WindSample:
    vector_mps: np.ndarray
    base_mps: np.ndarray
    disturbance_mps: np.ndarray
    scenario_code: int
    active: bool

    def as_record(self) -> dict:
        return {
            "wind_x_mps": float(self.vector_mps[0]),
            "wind_y_mps": float(self.vector_mps[1]),
            "wind_z_mps": float(self.vector_mps[2]),
            "wind_speed_mps": float(np.linalg.norm(self.vector_mps)),
            "gust_component_mps": float(
                np.linalg.norm(self.disturbance_mps)
            ),
            "wind_scenario_code": float(self.scenario_code),
            "gust_active": float(self.active),
        }


class UrbanWindModel:
    """Repeatable wind/gust generator for dashboard experiments."""

    def __init__(self, config: WindScenarioConfig | None = None):
        self.config = (
            WindScenarioConfig() if config is None else config.normalized()
        )
        self.last_sample = WindSample(
            np.zeros(3), np.zeros(3), np.zeros(3), 0, False
        )

    def configure(self, config: WindScenarioConfig) -> None:
        self.config = config.normalized()

    def reset(self) -> None:
        self.last_sample = WindSample(
            np.zeros(3), np.zeros(3), np.zeros(3), 0, False
        )

    @staticmethod
    def _direction_vectors(direction_deg: float) -> tuple[np.ndarray, np.ndarray]:
        angle = np.deg2rad(direction_deg)
        along = np.array([np.cos(angle), np.sin(angle), 0.0])
        cross = np.array([-np.sin(angle), np.cos(angle), 0.0])
        return along, cross

    @staticmethod
    def _pulse(t: float, start: float, duration: float) -> tuple[float, float]:
        if t < start or t > start + duration:
            return 0.0, 0.0
        phase = np.clip((t - start) / duration, 0.0, 1.0)
        return float(np.sin(np.pi * phase) ** 2), float(phase)

    def sample(self, t: float, state: dict) -> WindSample:
        cfg = self.config
        if not cfg.enabled:
            sample = WindSample(
                np.zeros(3), np.zeros(3), np.zeros(3), 0, False
            )
            self.last_sample = sample
            return sample

        base = np.asarray(cfg.base_wind_mps, dtype=float).copy()
        disturbance = np.zeros(3)
        along, cross = self._direction_vectors(
            cfg.disturbance_direction_deg
        )
        active = False

        if cfg.mode == "steady":
            pass

        elif cfg.mode == "discrete_gust":
            envelope, phase = self._pulse(
                float(t), cfg.start_time_s, cfg.duration_s
            )
            active = envelope > 1e-9
            disturbance = (
                cfg.disturbance_amplitude_mps * envelope * along
                + 0.20
                * cfg.disturbance_amplitude_mps
                * envelope
                * np.sin(2.0 * np.pi * phase)
                * np.array([0.0, 0.0, 1.0])
            )

        elif cfg.mode == "wind_shear":
            altitude = float(np.asarray(state["x"])[2])
            delta_h = altitude - cfg.reference_altitude_m
            shear_speed = np.clip(
                cfg.shear_gradient_per_m * delta_h,
                -cfg.disturbance_amplitude_mps,
                cfg.disturbance_amplitude_mps,
            )
            disturbance = shear_speed * along
            active = abs(shear_speed) > 1e-9

        elif cfg.mode == "urban_wake":
            envelope, _ = self._pulse(
                float(t), cfg.start_time_s, cfg.duration_s
            )
            active = envelope > 1e-9
            omega_t = 2.0 * np.pi * cfg.wake_frequency_hz * (
                float(t) - cfg.start_time_s
            )
            disturbance = cfg.disturbance_amplitude_mps * envelope * (
                0.55 * np.sin(omega_t) * along
                + 0.75 * np.sin(1.73 * omega_t + 0.65) * cross
                + 0.30
                * np.sin(0.61 * omega_t + 1.10)
                * np.array([0.0, 0.0, 1.0])
            )

        vector = base + disturbance
        sample = WindSample(
            vector_mps=vector,
            base_mps=base,
            disturbance_mps=disturbance,
            scenario_code=WIND_SCENARIO_CODES[cfg.mode],
            active=active,
        )
        self.last_sample = sample
        return sample


@dataclass
class ComfortLimits:
    acceleration_mps2: float = 2.0
    jerk_mps3: float = 1.5
    angular_rate_deg_s: float = 10.0
    rms_window_s: float = 5.0

    def normalized(self) -> "ComfortLimits":
        return ComfortLimits(
            acceleration_mps2=max(0.05, float(self.acceleration_mps2)),
            jerk_mps3=max(0.05, float(self.jerk_mps3)),
            angular_rate_deg_s=max(
                0.1, float(self.angular_rate_deg_s)
            ),
            rms_window_s=max(0.25, float(self.rms_window_s)),
        )


class PassengerComfortMonitor:
    """Rolling comfort indicators for comparative controller experiments.

    Metrics are engineering proxies. They intentionally avoid claiming
    compliance with ISO 2631 or any certification standard.
    """

    def __init__(
        self,
        limits: ComfortLimits | None = None,
        enabled: bool = True,
    ):
        self.enabled = bool(enabled)
        self.limits = (
            ComfortLimits() if limits is None else limits.normalized()
        )
        self.reset()

    def configure(
        self,
        *,
        enabled: bool | None = None,
        limits: ComfortLimits | None = None,
    ) -> None:
        if enabled is not None:
            self.enabled = bool(enabled)
        if limits is not None:
            self.limits = limits.normalized()

    def reset(self) -> None:
        self._history: deque[tuple[float, float, float, float]] = deque()
        self._last_time: float | None = None
        self.peak_acceleration_mps2 = 0.0
        self.peak_jerk_mps3 = 0.0
        self.peak_angular_rate_deg_s = 0.0
        self.exceedance_time_s = 0.0

    @staticmethod
    def _rms(values: list[float]) -> float:
        if not values:
            return 0.0
        arr = np.asarray(values, dtype=float)
        return float(np.sqrt(np.mean(arr * arr)))

    def update(
        self,
        t: float,
        accel_world: np.ndarray,
        jerk_world: np.ndarray,
        body_rates_rad_s: np.ndarray,
    ) -> dict:
        limits = self.limits
        accel = float(np.linalg.norm(np.asarray(accel_world, dtype=float)))
        jerk = float(np.linalg.norm(np.asarray(jerk_world, dtype=float)))
        rate = float(
            np.linalg.norm(
                np.rad2deg(np.asarray(body_rates_rad_s, dtype=float))
            )
        )

        if not self.enabled:
            self._last_time = float(t)
            return {
                "comfort_enabled": 0.0,
                "comfort_accel_mps2": accel,
                "comfort_accel_rms_mps2": 0.0,
                "comfort_jerk_mps3": jerk,
                "comfort_jerk_rms_mps3": 0.0,
                "comfort_rate_deg_s": rate,
                "comfort_rate_rms_deg_s": 0.0,
                "comfort_index": 0.0,
                "comfort_rms_index": 0.0,
                "comfort_alert": 0.0,
                "comfort_exceedance_time_s": self.exceedance_time_s,
                "comfort_peak_accel_mps2": self.peak_acceleration_mps2,
                "comfort_peak_jerk_mps3": self.peak_jerk_mps3,
                "comfort_peak_rate_deg_s": self.peak_angular_rate_deg_s,
                "comfort_accel_limit_mps2": limits.acceleration_mps2,
                "comfort_jerk_limit_mps3": limits.jerk_mps3,
                "comfort_rate_limit_deg_s": limits.angular_rate_deg_s,
            }

        now = float(t)
        dt = (
            0.0
            if self._last_time is None
            else max(0.0, now - self._last_time)
        )
        self._last_time = now
        self._history.append((now, accel, jerk, rate))
        cutoff = now - limits.rms_window_s
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        a_values = [item[1] for item in self._history]
        j_values = [item[2] for item in self._history]
        r_values = [item[3] for item in self._history]
        a_rms = self._rms(a_values)
        j_rms = self._rms(j_values)
        r_rms = self._rms(r_values)

        self.peak_acceleration_mps2 = max(
            self.peak_acceleration_mps2, accel
        )
        self.peak_jerk_mps3 = max(self.peak_jerk_mps3, jerk)
        self.peak_angular_rate_deg_s = max(
            self.peak_angular_rate_deg_s, rate
        )

        ratios = np.array([
            accel / limits.acceleration_mps2,
            jerk / limits.jerk_mps3,
            rate / limits.angular_rate_deg_s,
        ])
        rms_ratios = np.array([
            a_rms / limits.acceleration_mps2,
            j_rms / limits.jerk_mps3,
            r_rms / limits.angular_rate_deg_s,
        ])
        index = float(np.max(ratios))
        rms_index = float(np.max(rms_ratios))
        alert = index > 1.0
        if alert:
            self.exceedance_time_s += dt

        return {
            "comfort_enabled": 1.0,
            "comfort_accel_mps2": accel,
            "comfort_accel_rms_mps2": a_rms,
            "comfort_jerk_mps3": jerk,
            "comfort_jerk_rms_mps3": j_rms,
            "comfort_rate_deg_s": rate,
            "comfort_rate_rms_deg_s": r_rms,
            "comfort_index": index,
            "comfort_rms_index": rms_index,
            "comfort_alert": float(alert),
            "comfort_exceedance_time_s": self.exceedance_time_s,
            "comfort_peak_accel_mps2": self.peak_acceleration_mps2,
            "comfort_peak_jerk_mps3": self.peak_jerk_mps3,
            "comfort_peak_rate_deg_s": self.peak_angular_rate_deg_s,
            "comfort_accel_limit_mps2": limits.acceleration_mps2,
            "comfort_jerk_limit_mps3": limits.jerk_mps3,
            "comfort_rate_limit_deg_s": limits.angular_rate_deg_s,
        }
