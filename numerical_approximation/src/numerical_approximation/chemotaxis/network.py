from typing import Literal

import numpy as np

from .hyperbolic import (
    ArcParams,
    interior_step,
    neumann_left_boundary_step,
    neumann_right_boundary_step,
    outgoing_left,
    outgoing_right,
    sink_right_boundary_step,
    source_left_boundary_step,
)
from .parabolic import (
    ArcPhiParams,
    ExternalBC,
    InternalNode,
    Side,
    assemble_phi_system,
)

BoundaryKind = Literal["food_source", "food_sink", "neumann", "node"]


def boundary_gradient(phi_i_n: np.ndarray, h: float) -> np.ndarray:
    """phi_x at every grid point of one arc: central differences in the
    interior, one-sided differences at both ends.
    """
    px = np.empty_like(phi_i_n)
    px[1:-1] = (phi_i_n[2:] - phi_i_n[:-2]) / (2 * h)
    px[0] = (-3 * phi_i_n[0] + 4 * phi_i_n[1] - phi_i_n[2]) / (2 * h)
    px[-1] = (phi_i_n[-3] - 4 * phi_i_n[-2] + 3 * phi_i_n[-1]) / (2 * h)
    return px


def node_outgoing_characteristics(
    node_edges: list[tuple[int, Side]],
    arc_params: dict[int, ArcParams],
    u_minus: dict[int, np.ndarray],
    u_plus: dict[int, np.ndarray],
    phi: dict[int, np.ndarray],
    n: int,
) -> dict[int, float]:
    """Each node-facing edge's own outgoing characteristic"""
    out = {}
    for arc, side in node_edges:
        params = arc_params[arc]
        phi_x_n = boundary_gradient(phi[arc][n], params.h)
        M = phi[arc].shape[1] - 2
        if side == "right":
            out[arc] = outgoing_right(
                u_plus[arc][n, M + 1],
                u_minus[arc][n, M + 1],
                u_plus[arc][n, M],
                u_minus[arc][n, M],
                phi_x_n[-1],
                phi[arc][n, M + 1],
                phi[arc][n, M - 1],
                params,
            )
        else:
            out[arc] = outgoing_left(
                u_minus[arc][n, 0],
                u_plus[arc][n, 0],
                u_minus[arc][n, 1],
                u_plus[arc][n, 1],
                phi_x_n[0],
                phi[arc][n, 0],
                phi[arc][n, 2],
                params,
            )
    return out


def node_incoming_characteristics(
    node_edges: list[tuple[int, Side]],
    xi: dict[tuple[int, int], float],
    out: dict[int, float],
) -> dict[int, float]:
    """xi transmission law"""
    return {
        arc: sum(xi[(arc, j)] * out[j] for j, _ in node_edges) for arc, _ in node_edges
    }


def step_arc_uv(
    u_minus: dict[int, np.ndarray],
    u_plus: dict[int, np.ndarray],
    phi: dict[int, np.ndarray],
    arc: int,
    n: int,
    arc_params: dict[int, ArcParams],
    bc_type: dict[tuple[int, Side], BoundaryKind],
    node_out: dict[int, float] | None = None,
    node_in: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one arc's (u_minus, u_plus) from time row n to n+1"""
    params = arc_params[arc]
    phi_x_n = boundary_gradient(phi[arc][n], params.h)
    um_mid, up_mid = interior_step(u_minus[arc][n], u_plus[arc][n], phi_x_n, params)
    M = phi[arc].shape[1] - 2

    left_kind = bc_type[(arc, "left")]
    if left_kind == "food_source":
        um0, up0, _ = source_left_boundary_step(
            u_minus[arc][n, 0],
            u_plus[arc][n, 0],
            u_minus[arc][n, 1],
            u_plus[arc][n, 1],
            phi_x_n[0],
            phi[arc][n, 0],
            phi[arc][n, 2],
            params,
        )
    elif left_kind == "node":
        um0, up0 = node_out[arc], node_in[arc]
    else:
        um0, up0 = neumann_left_boundary_step(
            u_minus[arc][n, 0],
            u_plus[arc][n, 0],
            u_minus[arc][n, 1],
            u_plus[arc][n, 1],
            phi_x_n[0],
            phi[arc][n, 0],
            phi[arc][n, 2],
            params,
        )
    um_mid[0], up_mid[0] = um0, up0

    right_kind = bc_type[(arc, "right")]
    if right_kind == "food_sink":
        upE, umE, _ = sink_right_boundary_step(
            u_plus[arc][n, M + 1],
            u_minus[arc][n, M + 1],
            u_plus[arc][n, M],
            u_minus[arc][n, M],
            phi_x_n[-1],
            phi[arc][n, M + 1],
            phi[arc][n, M - 1],
            params,
        )
    elif right_kind == "node":
        upE, umE = node_out[arc], node_in[arc]
    else:
        upE, umE = neumann_right_boundary_step(
            u_plus[arc][n, M + 1],
            u_minus[arc][n, M + 1],
            u_plus[arc][n, M],
            u_minus[arc][n, M],
            phi_x_n[-1],
            phi[arc][n, M + 1],
            phi[arc][n, M - 1],
            params,
        )
    um_mid[-1], up_mid[-1] = umE, upE

    return um_mid, up_mid


def step_network(
    u_minus: dict[int, np.ndarray],
    u_plus: dict[int, np.ndarray],
    phi: dict[int, np.ndarray],
    n: int,
    arc_params: dict[int, ArcParams],
    phi_arc_params: dict[int, ArcPhiParams],
    bc_type: dict[tuple[int, Side], BoundaryKind],
    node_edges: list[tuple[int, Side]],
    xi: dict[tuple[int, int], float],
    external_bcs: list[ExternalBC],
    internal_nodes: list[InternalNode],
) -> None:
    """Advance every arc's u/v and the whole-network phi from time row n to
    n+1"""
    node_out = node_outgoing_characteristics(
        node_edges, arc_params, u_minus, u_plus, phi, n
    )
    node_in = node_incoming_characteristics(node_edges, xi, node_out)
    for arc in arc_params:
        um_next, up_next = step_arc_uv(
            u_minus, u_plus, phi, arc, n, arc_params, bc_type, node_out, node_in
        )
        u_minus[arc][n + 1] = um_next
        u_plus[arc][n + 1] = up_next

    u_next = {arc: u_minus[arc][n + 1] + u_plus[arc][n + 1] for arc in arc_params}
    u_n = {arc: u_minus[arc][n] + u_plus[arc][n] for arc in arc_params}
    phi_n = {arc: phi[arc][n] for arc in arc_params}
    phi_next = assemble_phi_system(
        phi_arc_params, external_bcs, internal_nodes, phi_n, u_next, u_n
    )
    for arc in arc_params:
        phi[arc][n + 1] = phi_next[arc]


def arc_mass(u_minus: np.ndarray, u_plus: np.ndarray, h: float) -> float:
    """Total organism mass on one arc at one time row."""
    return float((u_minus + u_plus).sum() * h)
