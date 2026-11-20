"""Spherical pendulum - numerical solution (pure Python / NumPy, no GUI).

Convention
    theta : polar angle measured from the HANGING position  (theta = 0  ->  bob at (0, 0, 0))
    phi   : azimuth in the xy-plane (phi0 = pi/2  ->  the initial displacement lies in the yz-plane)
    pivot : (0, 0, R)

Energy equation:
    1/2 m R^2 theta_dot^2 + 1/2 p_phi^2 / (m R^2 sin^2 theta) - m g R cos(theta) = E
    =>  theta_dot = +/- sqrt( 2/(m R^2) * ( E + m g R cos(theta) - p_phi^2 / (2 m R^2 sin^2 theta) ) )

Why the square root alone is not enough: the sign of theta_dot flips at the turning points and, with
floating-point error, the radicand can become slightly negative -> NaN (this happens for high energy).
So we integrate the equation of motion (obtained by differentiating the energy equation)
    theta_ddot = p_phi^2 cos(theta) / (m^2 R^4 sin^3 theta) - (g/R) sin(theta)
    phi_dot    = p_phi / (m R^2 sin^2 theta)
with RK4, and for every frame we also evaluate theta_dot from the energy equation to compare.
"""

import io
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Params:
    m: float = 1.0  # mass [kg]
    R: float = 1.0  # length [m]
    g: float = 9.81  # gravity [m/s^2]
    theta0: float = 0.8  # initial theta [rad], from the hanging position
    theta_dot0: float = 0.0  # initial theta_dot [rad/s]
    phi0: float = math.pi / 2  # initial azimuth: yz-plane
    phi_dot0: float = 1.5  # initial phi_dot [rad/s]

    @property
    def p_phi(self):  # conserved angular momentum about the z axis
        return self.m * self.R**2 * math.sin(self.theta0) ** 2 * self.phi_dot0

    @property
    def energy(self):  # conserved total energy E
        s = math.sin(self.theta0)
        centrifugal = (
            0.5 * self.p_phi**2 / (self.m * self.R**2 * s**2) if s > 1e-9 else 0.0
        )
        return (
            0.5 * self.m * self.R**2 * self.theta_dot0**2
            + centrifugal
            - self.m * self.g * self.R * math.cos(self.theta0)
        )


def theta_dot_from_energy(p, theta, sign=1.0, E=None, p_phi=None):
    """theta_dot taken directly from the energy equation. Returns (value, radicand); value is NaN if radicand < 0."""
    E = p.energy if E is None else E
    p_phi = p.p_phi if p_phi is None else p_phi
    s = math.sin(theta)
    centrifugal = p_phi**2 / (2 * p.m * p.R**2 * s**2) if abs(s) > 1e-9 else 0.0
    radicand = (
        2 / (p.m * p.R**2) * (E + p.m * p.g * p.R * math.cos(theta) - centrifugal)
    )
    return (sign * math.sqrt(radicand) if radicand >= 0 else math.nan), radicand


def _deriv(p, th, w, p_phi):
    s = math.sin(th)
    if abs(p_phi) > 1e-12 and abs(s) > 1e-9:
        acc = p_phi**2 * math.cos(th) / (p.m**2 * p.R**4 * s**3) - p.g / p.R * s
        phi_dot = p_phi / (p.m * p.R**2 * s * s)
    else:  # planar pendulum (p_phi = 0)
        acc, phi_dot = -p.g / p.R * s, 0.0
    return w, acc, phi_dot


def _rk4(p, y, h, p_phi):
    k1 = _deriv(p, y[0], y[1], p_phi)
    k2 = _deriv(p, y[0] + h / 2 * k1[0], y[1] + h / 2 * k1[1], p_phi)
    k3 = _deriv(p, y[0] + h / 2 * k2[0], y[1] + h / 2 * k2[1], p_phi)
    k4 = _deriv(p, y[0] + h * k3[0], y[1] + h * k3[1], p_phi)
    return tuple(
        y[i] + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(3)
    )


def g_force(p, th, w, phi_dot):
    """Tension in units of weight, T/(m g):  T = m v^2 / R + m g cos(theta)."""
    v2 = p.R**2 * (w**2 + math.sin(th) ** 2 * phi_dot**2)
    return (p.m * v2 / p.R + p.m * p.g * math.cos(th)) / (p.m * p.g)


def simulate(p, t_max=30.0, fps=60, substeps=20):
    """Integrate and return one row per animation frame (dict of NumPy arrays)."""
    n, h = int(t_max * fps), 1.0 / (fps * substeps)
    E, p_phi = p.energy, p.p_phi
    y = (p.theta0, p.theta_dot0, p.phi0)
    keys = [
        "t",
        "theta",
        "theta_dot",
        "theta_dot_eq",
        "radicand",
        "phi",
        "x",
        "y",
        "z",
        "g_force",
        "E_num",
    ]
    out = {k: np.empty(n) for k in keys}
    for i in range(n):
        th, w, ph = y
        phi_dot = _deriv(p, th, w, p_phi)[2]
        eq, rad = theta_dot_from_energy(
            p, th, math.copysign(1.0, w) if w else 1.0, E, p_phi
        )
        s = math.sin(th)
        cen = 0.5 * p_phi**2 / (p.m * p.R**2 * s**2) if abs(s) > 1e-9 else 0.0
        row = (
            i / fps,
            th,
            w,
            eq,
            rad,
            ph,
            p.R * s * math.cos(ph),
            p.R * s * math.sin(ph),
            p.R * (1 - math.cos(th)),
            g_force(p, th, w, phi_dot),
            0.5 * p.m * p.R**2 * w**2 + cen - p.m * p.g * p.R * math.cos(th),
        )
        for k, v in zip(keys, row):
            out[k][i] = v
        for _ in range(substeps):
            y = _rk4(p, y, h, p_phi)
    return out


def to_csv(sim):
    """theta_dot of every frame as CSV text."""
    cols = [
        "t",
        "theta",
        "theta_dot",
        "theta_dot_eq",
        "radicand",
        "x",
        "y",
        "z",
        "g_force",
    ]
    buf = io.StringIO()
    buf.write("frame," + ",".join(cols) + "\n")
    for i in range(len(sim["t"])):
        buf.write(str(i) + "," + ",".join(f"{sim[c][i]:.6f}" for c in cols) + "\n")
    return buf.getvalue()


if __name__ == "__main__":
    # Quick command-line check: python pendulum.py
    for label, p in [
        ("low energy ", Params(theta0=0.8, theta_dot0=0.0)),
        ("high energy", Params(theta0=0.8, theta_dot0=9.0)),
    ]:
        sim = simulate(p)
        bad = int(np.isnan(sim["theta_dot_eq"]).sum())
        print(
            f"{label}: E={p.energy:8.3f}  max|E_num-E|={np.abs(sim['E_num'] - p.energy).max():.2e}  "
            f"frames where the sqrt formula fails: {bad}/{len(sim['t'])}"
        )
