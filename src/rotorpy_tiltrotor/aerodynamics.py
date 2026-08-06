"""Reduced-order wing/tail aerodynamics for the tiltrotor prototype."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AeroResult:
    force_body: np.ndarray
    moment_body: np.ndarray
    airspeed: float
    alpha: float
    beta: float
    qbar: float
    wing_blend: float
    coefficients: dict


def _smoothstep01(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def aerodynamic_wrench(
    body_airspeed: np.ndarray,
    body_rates: np.ndarray,
    surfaces: np.ndarray,
    params: dict,
    tilt_angle: float = 0.0,
) -> AeroResult:
    """Compute a blended quasi-steady aerodynamic wrench.

    The conventional wing/tail coefficient model is only valid when there is
    meaningful forward flow over the wing. Earlier versions used total airspeed
    directly, so a vertical climb produced ``alpha ~= -90 deg`` and artificial
    lift/pitching moments. This implementation always retains simple fuselage
    drag, but smoothly activates the fixed-wing model as forward speed and
    nacelle tilt increase.

    Body/world convention is FLU/z-up to remain compatible with RotorPy. This
    is a transparent reduced-order blend, not a post-stall or rotor-wing
    interference model.
    """
    body_airspeed = np.asarray(body_airspeed, dtype=float)
    u, v, w = body_airspeed
    p, q, r = np.asarray(body_rates, dtype=float)
    da, de, dr = np.asarray(surfaces, dtype=float)
    V = float(np.linalg.norm(body_airspeed))

    if V > 1e-9:
        body_drag = -V * params["body_drag"] @ body_airspeed
    else:
        body_drag = np.zeros(3)

    zeros = {
        "CL": 0.0,
        "CD": 0.0,
        "CY": 0.0,
        "Cl": 0.0,
        "Cm": 0.0,
        "Cn": 0.0,
    }
    if V < 0.25:
        return AeroResult(
            body_drag,
            np.zeros(3),
            V,
            0.0,
            0.0,
            0.0,
            0.0,
            zeros,
        )

    rho, S, b, c = params["rho"], params["S"], params["b"], params["c"]

    forward_speed = max(float(u), 0.0)
    v_start = float(params.get("wing_aero_start_mps", 2.0))
    v_full = float(params.get("wing_aero_full_mps", 7.0))
    speed_blend = _smoothstep01(
        (forward_speed - v_start) / max(v_full - v_start, 1e-6)
    )

    tilt_start = float(params.get("wing_tilt_start", np.deg2rad(15.0)))
    tilt_full = float(params.get("wing_tilt_full", np.deg2rad(70.0)))
    tilt_blend = _smoothstep01(
        (float(tilt_angle) - tilt_start) / max(tilt_full - tilt_start, 1e-6)
    )
    early_fraction = float(
        np.clip(params.get("wing_early_transition_fraction", 0.35), 0.0, 1.0)
    )
    wing_blend = float(
        speed_blend * (early_fraction + (1.0 - early_fraction) * tilt_blend)
    )

    if wing_blend < 1e-8:
        return AeroResult(
            body_drag,
            np.zeros(3),
            V,
            0.0,
            0.0,
            0.0,
            0.0,
            zeros,
        )

    alpha = float(np.arctan2(-w, max(forward_speed, 1e-6)))
    beta = float(np.arctan2(v, max(forward_speed, 1e-6)))
    qbar_raw = 0.5 * rho * V * V
    qbar = qbar_raw * wing_blend

    a_stall = params["alpha_stall"]
    alpha_eff = a_stall * np.tanh(alpha / max(a_stall, 1e-6))
    p_hat = p * b / (2.0 * V)
    q_hat = q * c / (2.0 * V)
    r_hat = r * b / (2.0 * V)

    CL = params["CL0"] + params["CL_alpha"] * alpha_eff + params["CL_de"] * de
    CD = params["CD0"] + params["CD_k"] * CL * CL
    CY = (
        params["CY_beta"] * beta
        + params["CY_p"] * p_hat
        + params["CY_r"] * r_hat
        + params["CY_dr"] * dr
    )
    Cl = (
        params["Cl_beta"] * beta
        - params["Cl_p_damp"] * p_hat
        + params["Cl_r"] * r_hat
        + params["Cl_da"] * da
        + params["Cl_dr"] * dr
    )
    Cm = (
        params["Cm0"]
        + params["Cm_alpha"] * alpha_eff
        - params["Cm_q_damp"] * q_hat
        + params["Cm_de"] * de
    )
    Cn = (
        params["Cn_beta"] * beta
        + params["Cn_p"] * p_hat
        - params["Cn_r_damp"] * r_hat
        + params["Cn_da"] * da
        + params["Cn_dr"] * dr
    )

    lift = qbar * S * CL
    drag = qbar * S * CD
    side = qbar * S * CY

    ca, sa = np.cos(alpha), np.sin(alpha)
    wing_force_body = np.array([
        -drag * ca + lift * sa,
        side,
        drag * sa + lift * ca,
    ])
    force_body = wing_force_body + body_drag
    moment_body = qbar * S * np.array([b * Cl, c * Cm, b * Cn])

    effective_coefficients = {
        "CL": wing_blend * CL,
        "CD": wing_blend * CD,
        "CY": wing_blend * CY,
        "Cl": wing_blend * Cl,
        "Cm": wing_blend * Cm,
        "Cn": wing_blend * Cn,
    }
    return AeroResult(
        force_body=force_body,
        moment_body=moment_body,
        airspeed=V,
        alpha=alpha,
        beta=beta,
        qbar=qbar,
        wing_blend=wing_blend,
        coefficients=effective_coefficients,
    )
