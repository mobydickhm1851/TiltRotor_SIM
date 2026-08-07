"""Local small-signal Bode analysis for the implemented tiltrotor controller.

The nonlinear simulator contains mode switching, saturation, comfort envelopes,
safety overrides, wind scenarios and gain scheduling.  A single global LTI
model is therefore not physically meaningful.  This module derives transparent
SISO reduced-order loop models directly from the gains, actuator time constants
and rigid-body inertias already used by the simulator at fixed operating points.

The models are intended for Q4 frequency-domain verification:

* Hover altitude: cascaded position/vertical-speed loop + rotor lag.
* Cruise airspeed: forward-speed P loop + rotor lag.
* Pitch attitude (hover/cruise): implemented attitude PID-equivalent loop with
  the dominant rotor or elevator actuator lag and Iyy rigid-body dynamics.
* Cruise cross-track: implemented cross-track outer loop wrapped around the
  closed roll-attitude loop and lateral kinematics.

Comfort Guard, actuator saturation and altitude-safety override are nonlinear
supervisory elements.  Around a sufficiently small perturbation they are inactive,
so they are intentionally excluded from the nominal local Bode model and must be
verified using the existing nonlinear time-domain regressions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import control as ct
import numpy as np

from .controller import TiltrotorController
from .parameters import default_tiltrotor_params
from .vehicle import Tiltrotor


DEFAULT_OMEGA_RAD_S = np.logspace(-2.0, np.log10(50.0), 700)


@dataclass(frozen=True)
class LoopDefinition:
    key: str
    label: str
    operating_point: str
    input_name: str
    output_name: str
    derivation: str
    open_loop: ct.TransferFunction


@dataclass(frozen=True)
class BodeResult:
    definition: LoopDefinition
    omega_rad_s: np.ndarray
    open_mag_db: np.ndarray
    open_phase_deg: np.ndarray
    closed_mag_db: np.ndarray
    closed_phase_deg: np.ndarray
    phase_margin_deg: float
    gain_crossover_rad_s: float
    gain_margin_db: float
    phase_crossover_rad_s: float
    closed_loop_bandwidth_rad_s: float
    closed_loop_peak_db: float
    closed_loop_poles: np.ndarray
    stable: bool


def _pid_equivalent(kd: float, kp: float, ki: float) -> ct.TransferFunction:
    """Return Kd*s + Kp + Ki/s as one transfer function."""
    return ct.tf([float(kd), float(kp), float(ki)], [1.0, 0.0])


def _motor_acceleration_plant(tau_m: float) -> ct.TransferFunction:
    # acceleration command -> position, with first-order rotor/thrust response
    return ct.tf([1.0], [float(tau_m), 1.0, 0.0, 0.0])


def _speed_plant(tau_m: float) -> ct.TransferFunction:
    # acceleration command -> forward speed
    return ct.tf([1.0], [float(tau_m), 1.0, 0.0])


def _attitude_plant(inertia: float, actuator_tau: float) -> ct.TransferFunction:
    # moment command -> attitude angle: actuator lag + rigid-body double integral
    return ct.tf(
        [1.0],
        [float(inertia * actuator_tau), float(inertia), 0.0, 0.0],
    )


def build_loop_definitions() -> Dict[str, LoopDefinition]:
    """Build local loop-transfer functions from the current implemented gains."""
    vehicle = Tiltrotor()
    controller = TiltrotorController(vehicle)
    p = default_tiltrotor_params()

    # Hover altitude controller:
    # a_z = K_v (K_x e_h - h_dot) + K_i integral(e_h)
    # For a fixed reference, -h_dot = d(e_h)/dt, so the local loop is a
    # PID-equivalent position controller with Kd=K_v, Kp=K_v*K_x, Ki=K_i.
    hover_kd = float(controller.kp_vel_hover[2])
    hover_kp = float(controller.kp_vel_hover[2] * controller.k_pos_to_vel_hover[2])
    hover_ki = float(controller.ki_pos_hover[2])
    hover_controller = _pid_equivalent(hover_kd, hover_kp, hover_ki)
    hover_plant = _motor_acceleration_plant(float(p["tau_m"]))
    hover_altitude = LoopDefinition(
        key="hover_altitude",
        label="Hover altitude",
        operating_point="Hover, nacelle = 0 deg",
        input_name="Altitude error",
        output_name="Altitude",
        derivation=(
            "C_h(s)=(K_v s^2 + K_v K_x s + K_i)/s; "
            "P_h(s)=1/[s^2(τ_m s+1)]."
        ),
        open_loop=hover_controller * hover_plant,
    )

    # Cruise speed loop: a_x = K_V (V_ref - V), followed by rotor/thrust lag
    # and one integration from acceleration to speed.
    speed_controller = ct.tf([float(controller.k_speed)], [1.0])
    cruise_speed = LoopDefinition(
        key="cruise_airspeed",
        label="Cruise airspeed",
        operating_point="Cruise, nacelle = 90 deg",
        input_name="Forward-speed error",
        output_name="Forward speed",
        derivation="C_V(s)=K_V; P_V(s)=1/[s(τ_m s+1)].",
        open_loop=speed_controller * _speed_plant(float(p["tau_m"])),
    )

    # Pitch-attitude inner loop.  The implemented controller is PID-equivalent
    # in angle error.  Hover is rotor-moment dominated; cruise is elevator/
    # aerodynamic-moment dominated, so use the matching actuator lag at each end.
    pitch_controller = _pid_equivalent(
        float(controller.k_rate[1]),
        float(controller.k_att[1]),
        float(controller.ki_att[1]),
    )
    iyy = float(np.asarray(p["inertia"])[1, 1])
    pitch_hover = LoopDefinition(
        key="pitch_hover",
        label="Pitch attitude — hover",
        operating_point="Hover, rotor-moment dominated",
        input_name="Pitch-angle error",
        output_name="Pitch angle",
        derivation=(
            "C_θ(s)=(K_d s^2 + K_p s + K_i)/s; "
            "P_θ(s)=1/[Iyy s^2(τ_m s+1)]."
        ),
        open_loop=pitch_controller * _attitude_plant(iyy, float(p["tau_m"])),
    )
    pitch_cruise = LoopDefinition(
        key="pitch_cruise",
        label="Pitch attitude — cruise",
        operating_point="Cruise, elevator/aerodynamic-moment dominated",
        input_name="Pitch-angle error",
        output_name="Pitch angle",
        derivation=(
            "C_θ(s)=(K_d s^2 + K_p s + K_i)/s; "
            "P_θ(s)=1/[Iyy s^2(τ_e s+1)]."
        ),
        open_loop=pitch_controller * _attitude_plant(
            iyy, float(np.asarray(p["tau_surface"])[1])
        ),
    )

    # Cruise cross-track loop.  Around a straight route and small bank angle,
    # roll-command geometry cancels g between a_y≈-g*phi and phi_cmd≈-a_y/g.
    # The outer position/velocity controller therefore sees the closed inner roll
    # loop followed by two kinematic integrations.
    roll_controller = _pid_equivalent(
        float(controller.k_rate[0]),
        float(controller.k_att[0]),
        float(controller.ki_att[0]),
    )
    ixx = float(np.asarray(p["inertia"])[0, 0])
    roll_plant = _attitude_plant(
        ixx, float(np.asarray(p["tau_surface"])[0])
    )
    roll_closed = ct.feedback(roll_controller * roll_plant, 1.0)
    lateral_controller = _pid_equivalent(
        float(controller.kp_lateral_speed),
        float(controller.kp_lateral_speed * controller.k_cross_track_to_speed),
        float(controller.ki_cross_track),
    )
    lateral_kinematics = ct.tf([1.0], [1.0, 0.0, 0.0])
    cross_track = LoopDefinition(
        key="cruise_cross_track",
        label="Cruise cross-track",
        operating_point="Straight, wing-borne cruise",
        input_name="Cross-track error",
        output_name="Cross-track position",
        derivation=(
            "C_y(s)=(K_vy s^2 + K_vy K_y s + K_iy)/s; "
            "P_y(s)=T_roll(s)/s^2, where T_roll is the closed roll-attitude loop."
        ),
        open_loop=lateral_controller * roll_closed * lateral_kinematics,
    )

    return {
        loop.key: loop
        for loop in (
            hover_altitude,
            cruise_speed,
            pitch_hover,
            pitch_cruise,
            cross_track,
        )
    }


def _frequency_complex(system: ct.TransferFunction, omega: np.ndarray) -> np.ndarray:
    values = []
    for w in np.asarray(omega, dtype=float):
        value = np.asarray(ct.evalfr(system, 1j * float(w))).squeeze()
        values.append(complex(value))
    return np.asarray(values, dtype=complex)


def _phase_deg(values: np.ndarray) -> np.ndarray:
    phase = np.unwrap(np.angle(values))
    deg = np.rad2deg(phase)
    # Multi-integrator loops can start at +90 deg in principal-value form even
    # though the conventional Bode branch is -270 deg.  Shift the entire branch
    # down by 360 deg until the low-frequency phase is non-positive.
    while len(deg) and deg[0] > 0.0:
        deg = deg - 360.0
    return deg


def analyze_loop(
    definition: LoopDefinition,
    omega_rad_s: np.ndarray | None = None,
) -> BodeResult:
    omega = (
        DEFAULT_OMEGA_RAD_S.copy()
        if omega_rad_s is None
        else np.asarray(omega_rad_s, dtype=float)
    )
    closed = ct.feedback(definition.open_loop, 1.0)
    open_response = _frequency_complex(definition.open_loop, omega)
    closed_response = _frequency_complex(closed, omega)

    gm, pm, wcg, wcp = ct.margin(definition.open_loop)
    gm_db = (
        float(20.0 * np.log10(gm))
        if np.isfinite(gm) and gm > 0.0
        else float("inf")
    )
    try:
        bandwidth = float(ct.bandwidth(closed))
    except Exception:
        bandwidth = float("nan")

    poles = np.asarray(ct.poles(closed), dtype=complex)
    stable = bool(np.all(np.real(poles) < -1e-8))
    closed_mag_db = 20.0 * np.log10(
        np.maximum(np.abs(closed_response), 1e-14)
    )

    return BodeResult(
        definition=definition,
        omega_rad_s=omega,
        open_mag_db=20.0 * np.log10(
            np.maximum(np.abs(open_response), 1e-14)
        ),
        open_phase_deg=_phase_deg(open_response),
        closed_mag_db=closed_mag_db,
        closed_phase_deg=_phase_deg(closed_response),
        phase_margin_deg=float(pm),
        gain_crossover_rad_s=float(wcp),
        gain_margin_db=gm_db,
        phase_crossover_rad_s=float(wcg),
        closed_loop_bandwidth_rad_s=bandwidth,
        closed_loop_peak_db=float(np.max(closed_mag_db)),
        closed_loop_poles=poles,
        stable=stable,
    )


def analyze_all_loops() -> Dict[str, BodeResult]:
    return {
        key: analyze_loop(definition)
        for key, definition in build_loop_definitions().items()
    }


def phase_margin_grade(pm_deg: float, stable: bool) -> str:
    """Engineering interpretation, not a certification criterion."""
    if not stable or not np.isfinite(pm_deg) or pm_deg <= 0.0:
        return "unstable / unacceptable"
    if pm_deg >= 60.0:
        return "strong"
    if pm_deg >= 45.0:
        return "good"
    if pm_deg >= 30.0:
        return "moderate"
    return "low"


def format_poles(poles: np.ndarray, max_count: int = 8) -> str:
    ordered = sorted(
        np.asarray(poles, dtype=complex),
        key=lambda value: float(np.real(value)),
        reverse=True,
    )
    parts = []
    for pole in ordered[:max_count]:
        if abs(pole.imag) < 1e-7:
            parts.append(f"{pole.real:.3f}")
        else:
            sign = "+" if pole.imag >= 0 else "-"
            parts.append(f"{pole.real:.3f}{sign}{abs(pole.imag):.3f}j")
    return ", ".join(parts)


def summary_rows() -> list[dict]:
    rows = []
    for result in analyze_all_loops().values():
        rows.append({
            "key": result.definition.key,
            "loop": result.definition.label,
            "stable": result.stable,
            "phase_margin_deg": result.phase_margin_deg,
            "gain_crossover_rad_s": result.gain_crossover_rad_s,
            "bandwidth_rad_s": result.closed_loop_bandwidth_rad_s,
            "closed_loop_peak_db": result.closed_loop_peak_db,
            "grade": phase_margin_grade(
                result.phase_margin_deg, result.stable
            ),
            "poles": format_poles(result.closed_loop_poles),
        })
    return rows
