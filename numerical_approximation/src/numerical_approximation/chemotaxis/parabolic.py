from typing import Literal, NamedTuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

Side = Literal["left", "right"]  # left = j=0, right = j=M+1


class ArcPhiParams(NamedTuple):
    D: float  # diffusion coefficient
    a: float  # production rate
    b: float  # degradation rate
    h: float  # space step
    k: float  # time step
    M: int  # interior points; grid is j = 0, ..., M+1


class Endpoint(NamedTuple):
    arc: int
    side: Side


class ExternalBC(NamedTuple):
    """One outer node. kind='flux' is Eq. (41) (beta=0, given phibar);
    kind='robin1' is Eq. (42) (beta=1, phibar=0, phibar field ignored)."""

    endpoint: Endpoint
    kind: Literal["flux", "robin1"]
    phibar: float = 0.0  # phibar_i^{n+1}, used only when kind == "flux"


class InternalNode(NamedTuple):
    """One internal node p, keyed by the arcs meeting at it.
    kappa[(i, j)], i != j: phi flux coupling; must equal kappa[(j, i)] for
    flux conservation at the node.
    xi[(i, j)], including i == j: share of arc j's outgoing characteristic
    sent into arc i's incoming one; sum_i xi[(i, j)] = 1 conserves mass.
    """

    endpoints: list[Endpoint]
    kappa: dict[tuple[int, int], float]
    xi: dict[tuple[int, int], float]


def _add_row(rows, cols, vals, row_idx, col_idxs, coeffs):
    rows.extend([row_idx] * len(col_idxs))
    cols.extend(col_idxs)
    vals.extend(coeffs)


def assemble_phi_system(
    arcs: dict[int, ArcPhiParams],
    external_bcs: list[ExternalBC],
    internal_nodes: list[InternalNode],
    phi_n: dict[int, np.ndarray],  # phi at time n, per arc, length M+2
    u_next: dict[int, np.ndarray],  # u = u+ + u- at time n+1, per arc
    u_n: dict[int, np.ndarray],  # u = u+ + u- at time n, per arc
) -> dict[int, np.ndarray]:
    """One Crank-Nicolson step for phi across the whole network.
    Returns phi^{n+1} per arc."""

    offsets: dict[int, int] = {}
    total = 0
    for i, arc in arcs.items():
        offsets[i] = total
        total += arc.M + 2

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    c = np.zeros(total)

    for i, (D, a, b, h, k, M) in arcs.items():
        off = offsets[i]
        diag = 1 + b * k / 2 + D * k / h**2
        offdiag = -D * k / (2 * h**2)
        for j in range(1, M + 1):
            _add_row(
                rows,
                cols,
                vals,
                off + j,
                [off + j - 1, off + j, off + j + 1],
                [offdiag, diag, offdiag],
            )
            c[off + j] = (
                (1 - D * k / h**2) * phi_n[i][j]
                - D * k / (2 * h**2) * (-phi_n[i][j + 1] - phi_n[i][j - 1])
                + a * k / 2 * (u_next[i][j] + u_n[i][j])
                - b * k / 2 * phi_n[i][j]
            )

    for bc in external_bcs:
        i = bc.endpoint.arc
        h = arcs[i].h
        off = offsets[i]
        if bc.endpoint.side == "left":
            j0, j1, j2 = off, off + 1, off + 2
            if bc.kind == "flux":
                _add_row(rows, cols, vals, j0, [j0, j1, j2], [1.0, -4 / 3, 1 / 3])
                c[j0] = -2 * h / 3 * bc.phibar  # Eq. (41), as printed
            else:
                _add_row(
                    rows, cols, vals, j0, [j0, j1, j2], [1 + 2 * h / 3, -4 / 3, 1 / 3]
                )
                c[j0] = 0.0  # Eq. (42)
        else:
            M = arcs[i].M
            jE, jM, jMm1 = off + M + 1, off + M, off + M - 1
            if bc.kind == "flux":
                _add_row(rows, cols, vals, jE, [jMm1, jM, jE], [1 / 3, -4 / 3, 1.0])
                c[jE] = 2 * h / 3 * bc.phibar  # Eq. (41)
            else:
                _add_row(
                    rows, cols, vals, jE, [jMm1, jM, jE], [1 / 3, -4 / 3, 1 + 2 * h / 3]
                )
                c[jE] = 0.0  # Eq. (42)

    for node in internal_nodes:
        for ep in node.endpoints:
            i, D_i, h_i, off = ep.arc, arcs[ep.arc].D, arcs[ep.arc].h, offsets[ep.arc]
            M = arcs[i].M
            kappa_sum = sum(
                node.kappa[(i, other.arc)] for other in node.endpoints if other.arc != i
            )
            eta = 1 + (2 / 3) * (h_i / D_i) * kappa_sum

            if ep.side == "left":
                own_idx = off
                _add_row(
                    rows,
                    cols,
                    vals,
                    own_idx,
                    [off, off + 1, off + 2],
                    [eta, -4 / 3, 1 / 3],
                )
            else:
                own_idx = off + M + 1
                _add_row(
                    rows,
                    cols,
                    vals,
                    own_idx,
                    [off + M - 1, off + M, own_idx],
                    [1 / 3, -4 / 3, eta],
                )

            for other in node.endpoints:
                if other.arc == i:
                    continue
                other_off = offsets[other.arc]
                other_idx = (
                    other_off
                    if other.side == "left"
                    else other_off + arcs[other.arc].M + 1
                )
                coeff = -(2 * h_i) / (3 * D_i) * node.kappa[(i, other.arc)]
                rows.append(own_idx)
                cols.append(other_idx)
                vals.append(coeff)

            c[own_idx] = 0.0  # no RHS given for transmission rows -> homogeneous

    M_global = sp.coo_matrix((vals, (rows, cols)), shape=(total, total)).tocsc()
    phi_next_flat = spla.spsolve(M_global, c)

    return {i: phi_next_flat[offsets[i] : offsets[i] + arcs[i].M + 2] for i in arcs}
