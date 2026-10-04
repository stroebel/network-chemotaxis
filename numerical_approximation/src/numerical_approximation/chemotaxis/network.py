import warnings
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
    Endpoint,
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


def _outgoing_at(
    ep: Endpoint,
    arc_params: dict[int, ArcParams],
    u_minus: dict[int, np.ndarray],
    u_plus: dict[int, np.ndarray],
    phi: dict[int, np.ndarray],
    n: int,
) -> float:
    """Outgoing characteristic at one arc end: u_plus at the right end,
    u_minus at the left end"""
    arc = ep.arc
    params = arc_params[arc]
    phi_x_n = boundary_gradient(phi[arc][n], params.h)
    M = phi[arc].shape[1] - 2
    if ep.side == "right":
        return outgoing_right(
            u_plus[arc][n, M + 1],
            u_minus[arc][n, M + 1],
            u_plus[arc][n, M],
            u_minus[arc][n, M],
            phi_x_n[-1],
            phi[arc][n, M + 1],
            phi[arc][n, M - 1],
            params,
        )
    return outgoing_left(
        u_minus[arc][n, 0],
        u_plus[arc][n, 0],
        u_minus[arc][n, 1],
        u_plus[arc][n, 1],
        phi_x_n[0],
        phi[arc][n, 0],
        phi[arc][n, 2],
        params,
    )


def node_outgoing_characteristics(
    internal_nodes: list[InternalNode],
    arc_params: dict[int, ArcParams],
    u_minus: dict[int, np.ndarray],
    u_plus: dict[int, np.ndarray],
    phi: dict[int, np.ndarray],
    n: int,
) -> dict[Endpoint, float]:
    """Outgoing characteristic at every arc end that meets an internal node"""
    return {
        ep: _outgoing_at(ep, arc_params, u_minus, u_plus, phi, n)
        for node in internal_nodes
        for ep in node.endpoints
    }


def node_incoming_characteristics(
    internal_nodes: list[InternalNode],
    out: dict[Endpoint, float],
) -> dict[Endpoint, float]:
    """xi transmission law, applied separately at each internal node"""
    return {
        ep: sum(node.xi[(ep.arc, other.arc)] * out[other] for other in node.endpoints)
        for node in internal_nodes
        for ep in node.endpoints
    }


def step_arc_uv(
    u_minus: dict[int, np.ndarray],
    u_plus: dict[int, np.ndarray],
    phi: dict[int, np.ndarray],
    arc: int,
    n: int,
    arc_params: dict[int, ArcParams],
    bc_type: dict[tuple[int, Side], BoundaryKind],
    node_out: dict[Endpoint, float] | None = None,
    node_in: dict[Endpoint, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one arc's (u_minus, u_plus) from time row n to n+1.
    Ends present in node_out are internal-node ends; every other end uses
    bc_type[(arc, side)]"""
    node_out = node_out or {}
    node_in = node_in or {}
    left, right = Endpoint(arc, "left"), Endpoint(arc, "right")
    params = arc_params[arc]
    phi_x_n = boundary_gradient(phi[arc][n], params.h)
    um_mid, up_mid = interior_step(u_minus[arc][n], u_plus[arc][n], phi_x_n, params)
    M = phi[arc].shape[1] - 2

    left_kind = "node" if left in node_out else bc_type[left]
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
        um0, up0 = node_out[left], node_in[left]
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

    right_kind = "node" if right in node_out else bc_type[right]
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
        upE, umE = node_out[right], node_in[right]
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
    external_bcs: list[ExternalBC],
    internal_nodes: list[InternalNode],
) -> None:
    """Advance every arc's u/v and the whole-network phi from time row n to
    n+1. bc_type only needs the external (non-node) arc ends."""
    node_out = node_outgoing_characteristics(
        internal_nodes, arc_params, u_minus, u_plus, phi, n
    )
    node_in = node_incoming_characteristics(internal_nodes, node_out)
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


def validate_network(
    arc_params: dict[int, ArcParams],
    bc_type: dict[tuple[int, Side], BoundaryKind],
    external_bcs: list[ExternalBC],
    internal_nodes: list[InternalNode],
) -> None:
    """Check the network description once, before time stepping. Every arc
    end must belong to exactly one internal node or be external, with both a
    u/v kind (bc_type) and a phi condition (ExternalBC)."""
    arcs = set(arc_params)
    owner: dict[Endpoint, str] = {}

    for arc, p in arc_params.items():
        if not np.isclose(p.h, 2 * p.k * p.lam):
            warnings.warn(
                f"arc {arc}: h = {p.h:g} but 2*k*lam = {2 * p.k * p.lam:g}; the scheme "
                "assumes the CFL condition with equality on every arc"
            )

    def claim(ep: Endpoint, who: str) -> None:
        if ep.arc not in arcs:
            raise ValueError(f"{who} refers to arc {ep.arc}, not in arcs {sorted(arcs)}")
        if ep in owner:
            raise ValueError(f"{tuple(ep)} is claimed by both {owner[ep]} and {who}")
        owner[ep] = who

    for p, node in enumerate(internal_nodes):
        who = f"internal node {p}"
        node_arcs = [ep.arc for ep in node.endpoints]
        if len(node.endpoints) < 2:
            raise ValueError(f"{who} has fewer than 2 endpoints")
        if len(set(node_arcs)) != len(node_arcs):
            raise ValueError(f"{who} contains the same arc twice (self-loops are not supported)")
        for ep in node.endpoints:
            claim(Endpoint(*ep), who)
        for i in node_arcs:
            for j in node_arcs:
                if (i, j) not in node.xi:
                    raise ValueError(f"{who} is missing xi[{(i, j)}]")
                if i != j:
                    if (i, j) not in node.kappa:
                        raise ValueError(f"{who} is missing kappa[{(i, j)}]")
                    if node.kappa[(i, j)] != node.kappa.get((j, i)):
                        raise ValueError(f"{who} has kappa[{(i, j)}] != kappa[{(j, i)}]")
        for j in node_arcs:
            col = sum(node.xi[(i, j)] for i in node_arcs)
            if not np.isclose(col, 1.0):
                warnings.warn(
                    f"{who}: sum_i xi[(i, {j})] = {col:g} != 1, mass is not conserved"
                )

    phi_bcs: dict[Endpoint, int] = {}
    for bc in external_bcs:
        ep = Endpoint(*bc.endpoint)
        if ep in owner:
            raise ValueError(f"ExternalBC at {tuple(ep)} is on an {owner[ep]} end")
        phi_bcs[ep] = phi_bcs.get(ep, 0) + 1

    for (arc, side), kind in bc_type.items():
        ep = Endpoint(arc, side)
        if kind == "node":
            if ep not in owner:
                raise ValueError(f"bc_type{tuple(ep)} is 'node' but no internal node has it")
            continue
        claim(ep, "bc_type")
        if kind == "food_source" and side != "left":
            raise ValueError(f"food_source at {tuple(ep)}: only allowed on a left end")
        if kind == "food_sink" and side != "right":
            raise ValueError(f"food_sink at {tuple(ep)}: only allowed on a right end")
        if phi_bcs.get(ep, 0) != 1:
            raise ValueError(
                f"external end {tuple(ep)} needs exactly one ExternalBC, has {phi_bcs.get(ep, 0)}"
            )

    for ep in phi_bcs:
        if ep not in owner:
            raise ValueError(f"ExternalBC at {tuple(ep)} has no bc_type entry")

    for arc in sorted(arcs):
        for side in ("left", "right"):
            if Endpoint(arc, side) not in owner:
                raise ValueError(
                    f"arc end {(arc, side)} is in no internal node and has no bc_type"
                )


def arc_mass(u_minus: np.ndarray, u_plus: np.ndarray, h: float) -> float:
    """Total organism mass on one arc at one time row (trapezoidal rule over
    the grid j = 0, ..., M+1)."""
    u = u_minus + u_plus
    return float(h * (u.sum() - (u[0] + u[-1]) / 2))
