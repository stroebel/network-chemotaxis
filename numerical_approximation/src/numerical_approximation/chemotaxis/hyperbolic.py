import math
from typing import NamedTuple

import numpy as np


class ArcParams(NamedTuple):
    lam: float  # transport speed lambda_i
    h: float  # space step h_i
    k: float  # time step (same for all arcs, CFL: h = 2*k*lam)
    chi: float  # chemotactic sensitivity


def _closure_root(alpha: float, beta: float, gamma: float, known: float) -> float:
    b = beta + 2 * alpha * known
    c = gamma + known * beta + alpha * known**2
    radicand = b**2 - 4 * alpha * c
    return (-b + math.sqrt(radicand)) / (2 * alpha)


def interior_step(
    u_minus: np.ndarray, u_plus: np.ndarray, phi_x: np.ndarray, params: ArcParams
) -> tuple[np.ndarray, np.ndarray]:
    lam, h, k, chi = params.lam, params.h, params.k, params.chi
    M = len(u_minus) - 2  # grid indices run 0..M+1

    u = u_minus + u_plus  # density, length M+2
    f = chi * u * phi_x  # forcing term, length M+2

    u_minus_next = np.full(M + 2, np.nan)
    u_plus_next = np.full(M + 2, np.nan)

    # Eq. (14): u_minus at j = 0, ..., M
    j, jp1 = slice(0, M + 1), slice(1, M + 2)
    u_minus_next[j] = (
        (1 - lam * k / h - k / 4) * u_minus[j]
        + (lam * k / h - k / 4) * u_minus[jp1]
        + k / 4 * (u_plus[j] + u_plus[jp1])
        - k / (4 * lam) * (f[jp1] + f[j])
    )

    # Eq. (15): u_plus at j = 1, ..., M+1
    j, jm1 = slice(1, M + 2), slice(0, M + 1)
    u_plus_next[j] = (
        (1 - lam * k / h - k / 4) * u_plus[j]
        + (lam * k / h - k / 4) * u_plus[jm1]
        + k / 4 * (u_minus[j] + u_minus[jm1])
        + k / (4 * lam) * (f[jm1] + f[j])
    )

    return u_minus_next, u_plus_next


def outgoing_left(
    u_minus_0: float,
    u_plus_0: float,
    u_minus_1: float,
    u_plus_1: float,
    phi_x_0: float,
    phi_0: float,
    phi_2: float,
    params: ArcParams,
) -> float:
    """Eq. (29)'s explicit LxF advance of the outgoing characteristic u_minus at j=0"""
    lam, h, k, chi = params.lam, params.h, params.k, params.chi
    phi_x_1 = (phi_2 - phi_0) / (2 * h)
    return (
        u_minus_0 * (1 - lam * k / h - k / 4 - k / (4 * lam) * phi_x_0 * chi)
        + u_plus_0 * (k / 4 - k / (4 * lam) * phi_x_0 * chi)
        + u_minus_1 * (lam * k / h - k / 4 - k * chi / (4 * lam) * phi_x_1)
        + u_plus_1 * (k / 4 - k * chi / (4 * lam) * phi_x_1)
    )


def outgoing_right(
    u_plus_Mp1: float,
    u_minus_Mp1: float,
    u_plus_M: float,
    u_minus_M: float,
    phi_x_Mp1: float,
    phi_Mp1: float,
    phi_Mm1: float,
    params: ArcParams,
) -> float:
    """Eq. (30)'s explicit LxF advance of the outgoing characteristic u_plus at j=M+1"""
    lam, h, k, chi = params.lam, params.h, params.k, params.chi
    phi_x_M = (phi_Mp1 - phi_Mm1) / (2 * h)
    return (
        u_plus_Mp1 * (1 - lam * k / h - k / 4 + k / (4 * lam) * phi_x_Mp1 * chi)
        + u_minus_Mp1 * (k / 4 + k / (4 * lam) * phi_x_Mp1 * chi)
        + u_minus_M * (k / 4 + k * chi / (4 * lam) * phi_x_M)
        + u_plus_M * (lam * k / h - k / 4 + k * chi / (4 * lam) * phi_x_M)
    )


def source_left_boundary_step(
    u_minus_0: float,
    u_plus_0: float,  # j=0, time n  (u_plus_0 = g1^{n-1}(u_minus_0), already stored)
    u_minus_1: float,
    u_plus_1: float,  # j=1, time n
    phi_x_0: float,  # boundary derivative at j=0 (Eq. 41)
    phi_0: float,
    phi_2: float,  # for the central-difference phi_x at j=1
    params: ArcParams,
) -> tuple[float, float, float]:
    """
    One time-step update at a source (left, outflow-from-node) endpoint.
    Returns (u_minus_next_0, u_plus_next_0, v_next_0), where v_next_0 is
    the physical inflow flux 2/(1+u) actually imposed at t^{n+1}.
    """
    lam, h, k, chi = params.lam, params.h, params.k, params.chi
    alpha1 = h / k

    # densities and forcing term f = chi * u * phi_x
    u0 = u_minus_0 + u_plus_0
    u1 = u_minus_1 + u_plus_1
    phi_x_1 = (phi_2 - phi_0) / (2 * h)
    f0 = chi * u0 * phi_x_0
    f1 = chi * u1 * phi_x_1

    # Eq. (29): explicit advance of the outgoing characteristic u_minus
    u_minus_next_0 = outgoing_left(
        u_minus_0, u_plus_0, u_minus_1, u_plus_1, phi_x_0, phi_0, phi_2, params
    )

    # Eq. (25): A^n, then beta1^n, gamma1^n
    A_n = -2.0 / (1.0 + u0) + h * (
        -1.0 / k * u0
        + (2 * lam / h - 0.5) * u_plus_0
        + 0.5 * u_minus_0
        - 0.5 * u_plus_1
        - (2 * lam / h - 0.5) * u_minus_1
        + 1.0 / lam * (0.5 * f1 + 0.5 * f0)
    )
    beta1 = alpha1 + A_n
    gamma1 = A_n - 2.0

    # Eq. (28): closure g1^n(u_minus_next_0) -> u_plus_next_0
    u_plus_next_0 = _closure_root(alpha1, beta1, gamma1, u_minus_next_0)

    v_next_0 = 2.0 / (1.0 + u_plus_next_0 + u_minus_next_0)
    return u_minus_next_0, u_plus_next_0, v_next_0


def sink_right_boundary_step(
    u_plus_Mp1: float,
    u_minus_Mp1: float,  # j=M+1, time n (u_minus_Mp1 = g2^{n-1}(u_plus_Mp1), already stored)
    u_plus_M: float,
    u_minus_M: float,  # j=M, time n
    phi_x_Mp1: float,  # boundary derivative at j=M+1 (Eq. 41)
    phi_Mp1: float,
    phi_Mm1: float,  # for the central-difference phi_x at j=M
    params: ArcParams,
) -> tuple[float, float, float]:
    """
    One time-step update at a sink (right, inflow-to-node) endpoint.
    """
    lam, h, k, chi = params.lam, params.h, params.k, params.chi
    alpha2 = h / k

    u_end = u_plus_Mp1 + u_minus_Mp1
    u_M = u_plus_M + u_minus_M
    phi_x_M = (phi_Mp1 - phi_Mm1) / (2 * h)
    f_end = chi * u_end * phi_x_Mp1
    f_M = chi * u_M * phi_x_M

    # Eq. (30): explicit advance of the outgoing characteristic u_plus
    u_plus_next_Mp1 = outgoing_right(
        u_plus_Mp1,
        u_minus_Mp1,
        u_plus_M,
        u_minus_M,
        phi_x_Mp1,
        phi_Mp1,
        phi_Mm1,
        params,
    )

    # Mirrored Eq. (25)/(22): A^n_2, then beta2^n, gamma2^n
    z_n = -2.0 / (1.0 + u_end)
    A_n2 = z_n + h * (
        -1.0 / k * u_end
        + 0.5 * u_plus_Mp1
        + (2 * lam / h - 0.5) * u_minus_Mp1
        - (2 * lam / h - 0.5) * u_plus_M
        - 0.5 * u_minus_M
        - 1.0 / lam * (0.5 * f_end + 0.5 * f_M)
    )

    beta2 = alpha2 + A_n2
    gamma2 = A_n2 - 2.0

    # closure g2^n(u_plus_next_Mp1) -> u_minus_next_Mp1
    u_minus_next_Mp1 = _closure_root(alpha2, beta2, gamma2, u_plus_next_Mp1)

    v_next_Mp1 = -2.0 / (1.0 + u_plus_next_Mp1 + u_minus_next_Mp1)
    return u_plus_next_Mp1, u_minus_next_Mp1, v_next_Mp1


def neumann_left_boundary_step(
    u_minus_0: float,
    u_plus_0: float,
    u_minus_1: float,
    u_plus_1: float,
    phi_x_0: float,
    phi_0: float,
    phi_2: float,
    params: ArcParams,
) -> tuple[float, float]:
    """No-flux (v = 0, i.e. u_plus = u_minus) closure at a left endpoint with no food"""
    u_minus_next_0 = outgoing_left(
        u_minus_0, u_plus_0, u_minus_1, u_plus_1, phi_x_0, phi_0, phi_2, params
    )
    return u_minus_next_0, u_minus_next_0


def neumann_right_boundary_step(
    u_plus_Mp1: float,
    u_minus_Mp1: float,
    u_plus_M: float,
    u_minus_M: float,
    phi_x_Mp1: float,
    phi_Mp1: float,
    phi_Mm1: float,
    params: ArcParams,
) -> tuple[float, float]:
    """No-flux closure at a right endpoint with no food"""
    u_plus_next_Mp1 = outgoing_right(
        u_plus_Mp1,
        u_minus_Mp1,
        u_plus_M,
        u_minus_M,
        phi_x_Mp1,
        phi_Mp1,
        phi_Mm1,
        params,
    )
    return u_plus_next_Mp1, u_plus_next_Mp1
