"""Urban-wind scenarios and passenger-comfort metrics.

The disturbance models are deliberately reduced-order and repeatable.  The
ISO filter implementation is an ISO-2631-style engineering approximation for
comparative simulation work, not a certified vibration measurement chain.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import numpy as np
from scipy import signal


WIND_SCENARIO_CODES = {
    "off": 0,
    "steady": 1,
    "discrete_gust": 2,
    "repeated_gusts": 3,
    "continuous_turbulence": 4,
    "wind_shear": 5,
    "urban_wake": 6,
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
    # In v0.4.1 this is a delay after the scenario is enabled, not absolute
    # simulation time.  Keeping the field name preserves dashboard/API
    # compatibility with v0.4.0.
    start_time_s: float = 2.0
    duration_s: float = 8.0
    shear_gradient_per_m: float = 0.08
    reference_altitude_m: float = 30.0
    # Used as wake frequency, repeated-gust occurrence frequency, or the
    # characteristic frequency of the continuous-turbulence proxy.
    wake_frequency_hz: float = 0.35
    random_seed: int = 7

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
            random_seed=int(self.random_seed),
        )


@dataclass(frozen=True)
class WindSample:
    vector_mps: np.ndarray
    base_mps: np.ndarray
    disturbance_mps: np.ndarray
    scenario_code: int
    active: bool
    scenario_elapsed_s: float = 0.0
    gust_event_index: int = -1

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
            "scenario_elapsed_s": float(self.scenario_elapsed_s),
            "gust_event_index": float(self.gust_event_index),
        }


class UrbanWindModel:
    """Repeatable wind/gust generator for dashboard experiments.

    A scenario clock begins when a disturbance is enabled (or when the mode or
    schedule parameters are changed).  This removes the v0.4.0 ambiguity where
    a gust could silently expire before the user enabled it.
    """

    def __init__(self, config: WindScenarioConfig | None = None):
        self.config = (
            WindScenarioConfig() if config is None else config.normalized()
        )
        self._activation_time_s: float | None = None
        self._schedule_signature = self._signature(self.config)
        self._rng = np.random.default_rng(self.config.random_seed)
        self._turbulence_state = np.zeros(3)
        self._last_turbulence_time: float | None = None
        self.last_sample = WindSample(
            np.zeros(3), np.zeros(3), np.zeros(3), 0, False
        )

    @staticmethod
    def _signature(cfg: WindScenarioConfig) -> tuple:
        return (
            bool(cfg.enabled),
            str(cfg.mode),
            round(float(cfg.disturbance_amplitude_mps), 9),
            round(float(cfg.disturbance_direction_deg), 9),
            round(float(cfg.start_time_s), 9),
            round(float(cfg.duration_s), 9),
            round(float(cfg.shear_gradient_per_m), 9),
            round(float(cfg.reference_altitude_m), 9),
            round(float(cfg.wake_frequency_hz), 9),
            int(cfg.random_seed),
        )

    def configure(self, config: WindScenarioConfig) -> None:
        new_cfg = config.normalized()
        signature = self._signature(new_cfg)
        restart = signature != self._schedule_signature
        self.config = new_cfg
        self._schedule_signature = signature
        if not new_cfg.enabled:
            self._activation_time_s = None
            self._last_turbulence_time = None
            self._turbulence_state[:] = 0.0
        elif restart:
            # The actual activation time is captured by the next sample(t, ...)
            # so dashboard changes are referenced to the current simulation time.
            self._activation_time_s = None
            self._rng = np.random.default_rng(new_cfg.random_seed)
            self._last_turbulence_time = None
            self._turbulence_state[:] = 0.0

    def reset(self) -> None:
        self._activation_time_s = None
        self._rng = np.random.default_rng(self.config.random_seed)
        self._last_turbulence_time = None
        self._turbulence_state[:] = 0.0
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
        """Smooth finite pulse with zero value/slope at both ends."""
        if t < start or t > start + duration:
            return 0.0, 0.0
        phase = np.clip((t - start) / duration, 0.0, 1.0)
        return float(np.sin(np.pi * phase) ** 2), float(phase)

    def _relative_time(self, t: float) -> float:
        if self._activation_time_s is None:
            self._activation_time_s = float(t)
        return max(0.0, float(t) - self._activation_time_s)

    def _repeated_gusts(
        self,
        elapsed: float,
        cfg: WindScenarioConfig,
    ) -> tuple[np.ndarray, bool, int]:
        if elapsed < cfg.start_time_s:
            return np.zeros(3), False, -1
        frequency = max(0.02, cfg.wake_frequency_hz)
        interval = 1.0 / frequency
        # Keep repeated gusts visibly separated even when the generic duration
        # control is longer than the recurrence interval.
        pulse_duration = min(cfg.duration_s, 0.55 * interval)
        nominal_index = int(max(0.0, (elapsed - cfg.start_time_s) / interval))
        disturbance = np.zeros(3)
        active = False
        active_index = -1

        # Only nearby deterministic events can overlap the current time.
        for k in range(max(0, nominal_index - 3), nominal_index + 3):
            jitter = 0.18 * interval * np.sin(1.73 * k + 0.31)
            event_start = cfg.start_time_s + k * interval + jitter
            envelope, phase = self._pulse(
                elapsed, event_start, pulse_duration
            )
            if envelope <= 0.0:
                continue
            active = True
            active_index = k
            amplitude_scale = 0.65 + 0.35 * (
                0.5 + 0.5 * np.sin(2.11 * k + 1.17)
            )
            direction = (
                cfg.disturbance_direction_deg
                + 25.0 * np.sin(1.37 * k + 0.52)
            )
            along, _ = self._direction_vectors(direction)
            disturbance += (
                cfg.disturbance_amplitude_mps
                * amplitude_scale
                * envelope
                * along
            )
            disturbance += (
                0.15
                * cfg.disturbance_amplitude_mps
                * envelope
                * np.sin(2.0 * np.pi * phase + 0.4 * k)
                * np.array([0.0, 0.0, 1.0])
            )
        return disturbance, active, active_index

    def _continuous_turbulence(
        self,
        t: float,
        elapsed: float,
        cfg: WindScenarioConfig,
    ) -> tuple[np.ndarray, bool]:
        if elapsed < cfg.start_time_s:
            self._last_turbulence_time = float(t)
            self._turbulence_state[:] = 0.0
            return np.zeros(3), False

        dt = (
            0.0
            if self._last_turbulence_time is None
            else max(0.0, float(t) - self._last_turbulence_time)
        )
        self._last_turbulence_time = float(t)
        if dt <= 0.0:
            return self._turbulence_state.copy(), True

        # Ornstein-Uhlenbeck filtered stochastic proxy.  It is intentionally
        # simpler than a certified Dryden/Von-Karman implementation but gives
        # continuously varying, correlated 3-D disturbances.
        tau = max(0.18, 1.0 / (2.0 * np.pi * cfg.wake_frequency_hz))
        axis_tau = np.array([1.0, 0.75, 0.55]) * tau
        alpha = np.exp(-dt / axis_tau)
        sigma = cfg.disturbance_amplitude_mps * np.array([0.65, 0.85, 0.35])
        innovation = self._rng.normal(size=3)
        local = (
            alpha * self._turbulence_state
            + sigma * np.sqrt(np.maximum(0.0, 1.0 - alpha * alpha))
            * innovation
        )
        along, cross = self._direction_vectors(
            cfg.disturbance_direction_deg
        )
        # local[0/1/2] are along/cross/vertical components.
        self._turbulence_state = (
            local[0] * along
            + local[1] * cross
            + local[2] * np.array([0.0, 0.0, 1.0])
        )
        return self._turbulence_state.copy(), True

    def sample(self, t: float, state: dict) -> WindSample:
        cfg = self.config
        if not cfg.enabled:
            sample = WindSample(
                np.zeros(3), np.zeros(3), np.zeros(3), 0, False
            )
            self.last_sample = sample
            return sample

        elapsed = self._relative_time(float(t))
        base = np.asarray(cfg.base_wind_mps, dtype=float).copy()
        disturbance = np.zeros(3)
        along, cross = self._direction_vectors(
            cfg.disturbance_direction_deg
        )
        active = False
        event_index = -1

        if cfg.mode == "steady":
            pass

        elif cfg.mode == "discrete_gust":
            envelope, phase = self._pulse(
                elapsed, cfg.start_time_s, cfg.duration_s
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
            event_index = 0 if active else -1

        elif cfg.mode == "repeated_gusts":
            disturbance, active, event_index = self._repeated_gusts(
                elapsed, cfg
            )

        elif cfg.mode == "continuous_turbulence":
            disturbance, active = self._continuous_turbulence(
                float(t), elapsed, cfg
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
                elapsed, cfg.start_time_s, cfg.duration_s
            )
            active = envelope > 1e-9
            omega_t = 2.0 * np.pi * cfg.wake_frequency_hz * (
                elapsed - cfg.start_time_s
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
            scenario_elapsed_s=elapsed,
            gust_event_index=event_index,
        )
        self.last_sample = sample
        return sample


@dataclass
class ComfortLimits:
    # NASA-informed engineering targets for short transient UAM motions.
    # They are not certification limits.
    acceleration_mps2: float = 0.50
    jerk_mps3: float = 1.50
    angular_rate_deg_s: float = 10.0
    rms_window_s: float = 5.0
    # ISO 2631-1 comfort-band boundary: below 0.315 m/s^2 is commonly
    # described as "not uncomfortable" for frequency-weighted RMS vibration.
    iso_weighted_rms_mps2: float = 0.315

    def normalized(self) -> "ComfortLimits":
        return ComfortLimits(
            acceleration_mps2=max(0.05, float(self.acceleration_mps2)),
            jerk_mps3=max(0.05, float(self.jerk_mps3)),
            angular_rate_deg_s=max(
                0.1, float(self.angular_rate_deg_s)
            ),
            rms_window_s=max(0.25, float(self.rms_window_s)),
            iso_weighted_rms_mps2=max(
                0.02, float(self.iso_weighted_rms_mps2)
            ),
        )


def _cascade_tf(*sections: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    b = np.array([1.0])
    a = np.array([1.0])
    for b_section, a_section in sections:
        b = np.polymul(b, np.asarray(b_section, dtype=float))
        a = np.polymul(a, np.asarray(a_section, dtype=float))
    return b, a


def _iso_2631_analog_tf(kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Return an ISO-2631-style analogue Wd or Wk weighting transfer function.

    The factorization follows the published ISO 2631-1 weighting structure:
    0.4-Hz high-pass and 100-Hz low-pass band limits, an acceleration/velocity
    transition, and (for Wk) the additional rising-step section.  The simulator
    runs at 100 Hz, so this is used as a low-frequency engineering
    approximation rather than a compliance implementation extending to 80 Hz.
    """
    if kind not in {"Wd", "Wk"}:
        raise ValueError("kind must be 'Wd' or 'Wk'")

    w1 = 2.0 * np.pi * 0.4
    w2 = 2.0 * np.pi * 100.0
    high_pass = (
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, np.sqrt(2.0) * w1, w1 * w1]),
    )
    low_pass = (
        np.array([w2 * w2]),
        np.array([1.0, np.sqrt(2.0) * w2, w2 * w2]),
    )

    if kind == "Wd":
        f3 = f4 = 2.0
    else:
        f3 = f4 = 12.5
    q4 = 0.63
    w3 = 2.0 * np.pi * f3
    w4 = 2.0 * np.pi * f4
    transition = (
        np.array([w4 * w4 / w3, w4 * w4]),
        np.array([1.0, w4 / q4, w4 * w4]),
    )

    if kind == "Wd":
        return _cascade_tf(high_pass, low_pass, transition)

    f5, q5 = 2.37, 0.91
    f6, q6 = 3.35, 0.91
    w5 = 2.0 * np.pi * f5
    w6 = 2.0 * np.pi * f6
    rising_step = (
        np.array([1.0, w5 / q5, w5 * w5]),
        np.array([1.0, w6 / q6, w6 * w6]),
    )
    return _cascade_tf(high_pass, low_pass, transition, rising_step)


class ISO2631StyleWeighting:
    """Causal digital Wd/Wk approximation for a fixed simulation sample rate."""

    def __init__(self, sample_period_s: float = 0.01):
        self.sample_period_s = max(1e-4, float(sample_period_s))
        fs = 1.0 / self.sample_period_s
        self._sos = {}
        self._zi = {}
        for kind in ("Wd", "Wk"):
            b_a, a_a = _iso_2631_analog_tf(kind)
            b_z, a_z = signal.bilinear(b_a, a_a, fs=fs)
            sos = signal.tf2sos(b_z, a_z)
            self._sos[kind] = sos
            self._zi[kind] = np.zeros((sos.shape[0], 2), dtype=float)
        self.reset()

    def reset(self) -> None:
        for kind, sos in self._sos.items():
            self._zi[kind] = np.zeros((sos.shape[0], 2), dtype=float)

    def _filter_one(self, kind: str, value: float) -> float:
        y, zf = signal.sosfilt(
            self._sos[kind],
            np.array([float(value)]),
            zi=self._zi[kind],
        )
        self._zi[kind] = zf
        return float(y[-1])

    def update(self, accel_xyz: np.ndarray) -> np.ndarray:
        accel = np.asarray(accel_xyz, dtype=float).reshape(3)
        return np.array([
            self._filter_one("Wd", accel[0]),
            self._filter_one("Wd", accel[1]),
            self._filter_one("Wk", accel[2]),
        ])


class PassengerComfortMonitor:
    """Rolling comfort indicators for comparative controller experiments.

    Two different concepts are intentionally kept separate:

    * ``comfort_index``: transient-motion proxy based on peak translational
      acceleration, jerk and a user-defined angular-rate study threshold.
    * ``comfort_rms_index`` / ``comfort_iso_index``: ISO-2631-style
      frequency-weighted RMS vibration compared with 0.315 m/s^2.

    The first is useful for eVTOL maneuver jolts; the second is useful for
    sustained vibration/turbulence.  Neither is claimed as certification.
    """

    def __init__(
        self,
        limits: ComfortLimits | None = None,
        enabled: bool = True,
        sample_period_s: float = 0.01,
    ):
        self.enabled = bool(enabled)
        self.limits = (
            ComfortLimits() if limits is None else limits.normalized()
        )
        self.sample_period_s = max(1e-4, float(sample_period_s))
        self.weighting = ISO2631StyleWeighting(self.sample_period_s)
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
        self._history: deque[tuple[float, float, float, float, float, float, float]] = deque()
        self._last_time: float | None = None
        self.peak_acceleration_mps2 = 0.0
        self.peak_jerk_mps3 = 0.0
        self.peak_angular_rate_deg_s = 0.0
        self.exceedance_time_s = 0.0
        self.weighting.reset()

    @staticmethod
    def _rms(values: list[float]) -> float:
        if not values:
            return 0.0
        arr = np.asarray(values, dtype=float)
        return float(np.sqrt(np.mean(arr * arr)))

    def _record(
        self,
        *,
        enabled: float,
        accel: float,
        jerk: float,
        rate: float,
        a_rms: float,
        j_rms: float,
        r_rms: float,
        weighted_xyz: np.ndarray,
        iso_rms: float,
        transient_index: float,
        iso_index: float,
        alert: bool,
    ) -> dict:
        limits = self.limits
        return {
            "comfort_enabled": enabled,
            "comfort_accel_mps2": accel,
            "comfort_accel_rms_mps2": a_rms,
            "comfort_jerk_mps3": jerk,
            "comfort_jerk_rms_mps3": j_rms,
            "comfort_rate_deg_s": rate,
            "comfort_rate_rms_deg_s": r_rms,
            "comfort_index": transient_index,
            "comfort_rms_index": iso_index,
            "comfort_iso_index": iso_index,
            "comfort_alert": float(alert),
            "comfort_exceedance_time_s": self.exceedance_time_s,
            "comfort_peak_accel_mps2": self.peak_acceleration_mps2,
            "comfort_peak_jerk_mps3": self.peak_jerk_mps3,
            "comfort_peak_rate_deg_s": self.peak_angular_rate_deg_s,
            "comfort_accel_limit_mps2": limits.acceleration_mps2,
            "comfort_jerk_limit_mps3": limits.jerk_mps3,
            "comfort_rate_limit_deg_s": limits.angular_rate_deg_s,
            "comfort_iso_awx_mps2": float(weighted_xyz[0]),
            "comfort_iso_awy_mps2": float(weighted_xyz[1]),
            "comfort_iso_awz_mps2": float(weighted_xyz[2]),
            "comfort_iso_weighted_rms_mps2": float(iso_rms),
            "comfort_iso_limit_mps2": limits.iso_weighted_rms_mps2,
        }

    def update(
        self,
        t: float,
        accel_world: np.ndarray,
        jerk_world: np.ndarray,
        body_rates_rad_s: np.ndarray,
    ) -> dict:
        limits = self.limits
        accel_vec = np.asarray(accel_world, dtype=float).reshape(3)
        jerk_vec = np.asarray(jerk_world, dtype=float).reshape(3)
        accel = float(np.linalg.norm(accel_vec))
        jerk = float(np.linalg.norm(jerk_vec))
        rate = float(
            np.linalg.norm(
                np.rad2deg(np.asarray(body_rates_rad_s, dtype=float))
            )
        )

        if not self.enabled:
            self._last_time = float(t)
            return self._record(
                enabled=0.0,
                accel=accel,
                jerk=jerk,
                rate=rate,
                a_rms=0.0,
                j_rms=0.0,
                r_rms=0.0,
                weighted_xyz=np.zeros(3),
                iso_rms=0.0,
                transient_index=0.0,
                iso_index=0.0,
                alert=False,
            )

        now = float(t)
        dt = (
            0.0
            if self._last_time is None
            else max(0.0, now - self._last_time)
        )
        self._last_time = now
        weighted_xyz = self.weighting.update(accel_vec)

        self._history.append((
            now,
            accel,
            jerk,
            rate,
            float(weighted_xyz[0]),
            float(weighted_xyz[1]),
            float(weighted_xyz[2]),
        ))
        cutoff = now - limits.rms_window_s
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        a_values = [item[1] for item in self._history]
        j_values = [item[2] for item in self._history]
        r_values = [item[3] for item in self._history]
        awx_values = [item[4] for item in self._history]
        awy_values = [item[5] for item in self._history]
        awz_values = [item[6] for item in self._history]
        a_rms = self._rms(a_values)
        j_rms = self._rms(j_values)
        r_rms = self._rms(r_values)
        awx_rms = self._rms(awx_values)
        awy_rms = self._rms(awy_values)
        awz_rms = self._rms(awz_values)
        # For seated comfort ISO 2631-1 applies k=1 to Wd x/y and Wk z.
        iso_rms = float(np.sqrt(
            awx_rms * awx_rms
            + awy_rms * awy_rms
            + awz_rms * awz_rms
        ))

        self.peak_acceleration_mps2 = max(
            self.peak_acceleration_mps2, accel
        )
        self.peak_jerk_mps3 = max(self.peak_jerk_mps3, jerk)
        self.peak_angular_rate_deg_s = max(
            self.peak_angular_rate_deg_s, rate
        )

        transient_index = float(np.max(np.array([
            accel / limits.acceleration_mps2,
            jerk / limits.jerk_mps3,
            rate / limits.angular_rate_deg_s,
        ])))
        iso_index = float(
            iso_rms / limits.iso_weighted_rms_mps2
        )
        alert = max(transient_index, iso_index) > 1.0
        if alert:
            self.exceedance_time_s += dt

        return self._record(
            enabled=1.0,
            accel=accel,
            jerk=jerk,
            rate=rate,
            a_rms=a_rms,
            j_rms=j_rms,
            r_rms=r_rms,
            weighted_xyz=weighted_xyz,
            iso_rms=iso_rms,
            transient_index=transient_index,
            iso_index=iso_index,
            alert=alert,
        )
