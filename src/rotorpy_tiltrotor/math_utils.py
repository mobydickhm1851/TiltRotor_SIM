"""Small math helpers shared by dynamics, control, and dashboard code."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from rotorpy.vehicles.multirotor import quat_dot as rotorpy_quat_dot
except Exception:
    rotorpy_quat_dot = None


def hat(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def quat_dot(quat: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Quaternion derivative in scipy/RotorPy [x,y,z,w] ordering."""
    if rotorpy_quat_dot is not None:
        return rotorpy_quat_dot(quat, omega)
    qx, qy, qz, qw = np.asarray(quat, dtype=float)
    p, q, r = np.asarray(omega, dtype=float)
    G = np.array([[qw, qz, -qy, -qx], [-qz, qw, qx, -qy], [qy, -qx, qw, -qz]])
    return 0.5 * G.T @ np.array([p, q, r])


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=float)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def rotation_matrix(quat: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(normalize_quat(quat)).as_matrix()


def euler_deg(quat: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(normalize_quat(quat)).as_euler("xyz", degrees=True)


def smoothstep01(x: float | np.ndarray) -> float | np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def clamp_norm(v: np.ndarray, maximum: float) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v if n <= maximum or n < 1e-12 else v * (maximum / n)


def finite_difference_jacobian(fun, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(fun(x), dtype=float)
    jac = np.zeros((f0.size, x.size))
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps * max(1.0, abs(x[i]))
        jac[:, i] = (np.asarray(fun(x + dx)) - np.asarray(fun(x - dx))) / (2.0 * dx[i])
    return jac
