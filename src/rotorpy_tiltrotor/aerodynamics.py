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
    coefficients: dict


def aerodynamic_wrench(body_airspeed: np.ndarray, body_rates: np.ndarray, surfaces: np.ndarray, params: dict) -> AeroResult:
    u, v, w = np.asarray(body_airspeed, dtype=float)
    p, q, r = np.asarray(body_rates, dtype=float)
    da, de, dr = np.asarray(surfaces, dtype=float)
    V = float(np.linalg.norm(body_airspeed))
    if V < 0.25:
        zeros = np.zeros(3)
        return AeroResult(zeros, zeros, V, 0.0, 0.0, 0.0, {"CL":0.0,"CD":0.0,"CY":0.0,"Cl":0.0,"Cm":0.0,"Cn":0.0})
    rho, S, b, c = params["rho"], params["S"], params["b"], params["c"]
    alpha = float(np.arctan2(-w, max(abs(u), 1e-6) * np.sign(u if u != 0 else 1.0)))
    beta = float(np.arcsin(np.clip(v / V, -1.0, 1.0)))
    qbar = 0.5 * rho * V * V
    a_stall = params["alpha_stall"]
    alpha_eff = a_stall * np.tanh(alpha / max(a_stall, 1e-6))
    p_hat, q_hat, r_hat = p*b/(2*V), q*c/(2*V), r*b/(2*V)
    CL = params["CL0"] + params["CL_alpha"]*alpha_eff + params["CL_de"]*de
    CD = params["CD0"] + params["CD_k"]*CL*CL
    CY = params["CY_beta"]*beta + params["CY_p"]*p_hat + params["CY_r"]*r_hat + params["CY_dr"]*dr
    Cl = params["Cl_beta"]*beta - params["Cl_p_damp"]*p_hat + params["Cl_r"]*r_hat + params["Cl_da"]*da + params["Cl_dr"]*dr
    Cm = params["Cm0"] + params["Cm_alpha"]*alpha_eff - params["Cm_q_damp"]*q_hat + params["Cm_de"]*de
    Cn = params["Cn_beta"]*beta + params["Cn_p"]*p_hat - params["Cn_r_damp"]*r_hat + params["Cn_da"]*da + params["Cn_dr"]*dr
    lift, drag, side = qbar*S*CL, qbar*S*CD, qbar*S*CY
    ca, sa = np.cos(alpha), np.sin(alpha)
    force_body = np.array([-drag*ca + lift*sa, side, drag*sa + lift*ca])
    force_body += -np.linalg.norm(body_airspeed) * params["body_drag"] @ body_airspeed
    moment_body = qbar * S * np.array([b*Cl, c*Cm, b*Cn])
    return AeroResult(force_body, moment_body, V, alpha, beta, qbar, {"CL":CL,"CD":CD,"CY":CY,"Cl":Cl,"Cm":Cm,"Cn":Cn})
