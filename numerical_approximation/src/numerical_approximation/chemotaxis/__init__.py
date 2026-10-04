"""Chemotaxis on a metric graph: hyperbolic transport for the organism
density (u, v), Crank-Nicolson diffusion for the chemoattractant (phi), and
the per-time-step orchestration coupling them across a network of arcs.

Experiment notebooks should only need this package's public names plus
their own topology/parameters/initial conditions/plots.
"""

from .hyperbolic import (
    ArcParams,
    BlowUpError,
    interior_step,
    neumann_left_boundary_step,
    neumann_right_boundary_step,
    outgoing_left,
    outgoing_right,
    sink_monotone,
    sink_right_boundary_step,
    source_left_boundary_step,
    source_monotone,
)
from .network import (
    BoundaryKind,
    arc_mass,
    boundary_gradient,
    node_incoming_characteristics,
    node_outgoing_characteristics,
    step_arc_uv,
    step_network,
    validate_network,
)
from .parabolic import (
    ArcPhiParams,
    Endpoint,
    ExternalBC,
    InternalNode,
    Side,
    assemble_phi_system,
)

__all__ = [
    "ArcParams",
    "BlowUpError",
    "interior_step",
    "neumann_left_boundary_step",
    "neumann_right_boundary_step",
    "outgoing_left",
    "outgoing_right",
    "sink_monotone",
    "sink_right_boundary_step",
    "source_left_boundary_step",
    "source_monotone",
    "BoundaryKind",
    "arc_mass",
    "boundary_gradient",
    "node_incoming_characteristics",
    "node_outgoing_characteristics",
    "step_arc_uv",
    "step_network",
    "validate_network",
    "ArcPhiParams",
    "Endpoint",
    "ExternalBC",
    "InternalNode",
    "Side",
    "assemble_phi_system",
]
