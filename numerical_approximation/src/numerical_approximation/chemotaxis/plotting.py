"""
Claude figured out the plotting.
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from .parabolic import ExternalBC, InternalNode, Side

NodeKey = int | tuple[int, Side]


def _arc_positions(
    start: tuple[float, float], end: tuple[float, float], n: int, curvature: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """n points along a straight line (curvature=0) or a quadratic-Bezier bow
    (curvature = perpendicular control-point offset, as a fraction of arc length)."""
    p0, p1 = np.array(start), np.array(end)
    t = np.linspace(0, 1, n)
    if curvature == 0:
        pts = p0 + t[:, None] * (p1 - p0)
        return pts[:, 0], pts[:, 1]

    direction = p1 - p0
    length = np.linalg.norm(direction)
    perp = np.array([-direction[1], direction[0]]) / length if length else np.zeros(2)
    control = (p0 + p1) / 2 + perp * curvature * length
    pts = (
        (1 - t)[:, None] ** 2 * p0
        + 2 * (1 - t)[:, None] * t[:, None] * control
        + t[:, None] ** 2 * p1
    )
    return pts[:, 0], pts[:, 1]


def arc_endpoints(
    internal_nodes: list[InternalNode], external_bcs: list[ExternalBC]
) -> dict[int, tuple[NodeKey, NodeKey]]:
    """arc -> (node at its j=0 end, node at its j=M+1 end), so u along the
    arc can be drawn in array order"""
    node_of: dict[tuple[int, Side], NodeKey] = {}
    for p, node in enumerate(internal_nodes):
        for ep in node.endpoints:
            node_of[tuple(ep)] = p
    for bc in external_bcs:
        node_of[tuple(bc.endpoint)] = tuple(bc.endpoint)

    arcs: dict[int, tuple[NodeKey, NodeKey]] = {}
    for arc in sorted({arc for arc, _ in node_of}):
        left, right = (arc, "left"), (arc, "right")
        missing = [end for end in (left, right) if end not in node_of]
        if missing:
            raise ValueError(f"arc {arc}: end(s) {missing} are in no node or ExternalBC")
        arcs[arc] = (node_of[left], node_of[right])
    return arcs


def _default_labels(
    internal_nodes: list[InternalNode], external_bcs: list[ExternalBC]
) -> dict[NodeKey, str]:
    labels: dict[NodeKey, str] = {p: f"N{p}" for p in range(len(internal_nodes))}
    for bc in external_bcs:
        labels[tuple(bc.endpoint)] = (
            f"flux $\\bar\\phi$={bc.phibar:g}" if bc.kind == "flux" else bc.kind
        )
    return labels


def _arc_curvatures(
    arcs: dict[int, tuple[NodeKey, NodeKey]], curvature: dict[int, float] | None
) -> dict[int, float]:
    """Explicit curvature wins; parallel arcs between the same pair of nodes
    otherwise get bows 0, +0.25, -0.25, +0.5, ... so they don't overlap."""
    curvature = curvature or {}
    by_pair: dict[frozenset, list[int]] = defaultdict(list)
    for arc, (n0, n1) in arcs.items():
        by_pair[frozenset((n0, n1))].append(arc)

    result = {}
    for group in by_pair.values():
        for rank, arc in enumerate(group):
            if arc in curvature:
                result[arc] = curvature[arc]
                continue
            bow = 0.25 * ((rank + 1) // 2) * (1 if rank % 2 else -1)
            # keep the bow on the same side regardless of the arc's direction
            if arcs[arc] != arcs[group[0]]:
                bow = -bow
            result[arc] = bow
    return result


def plot_network_density(
    node_pos: dict[NodeKey, tuple[float, float]],
    internal_nodes: list[InternalNode],
    external_bcs: list[ExternalBC],
    u_values: dict[int, np.ndarray],  # arc_id -> density along the arc, j=0..M+1
    *,
    curvature: dict[int, float] | None = None,
    node_labels: dict[NodeKey, str] | None = None,
    cmap: str = "jet",
    vmin: float | None = None,
    vmax: float | None = None,
    lw_range: tuple[float, float] = (1.5, 8.0),
    title: str | None = None,
    ax: plt.Axes | None = None,
    colorbar: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """One frame of the density plot (call once per time step you want to show)."""
    arcs = arc_endpoints(internal_nodes, external_bcs)
    missing = sorted(
        {key for ends in arcs.values() for key in ends if key not in node_pos}, key=str
    )
    if missing:
        raise KeyError(f"node_pos is missing positions for nodes {missing}")

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()

    if vmin is None or vmax is None:
        all_vals = np.concatenate([np.asarray(u_values[arc]) for arc in arcs])
        vmin = float(all_vals.min()) if vmin is None else vmin
        vmax = float(all_vals.max()) if vmax is None else vmax
    norm = plt.Normalize(vmin, vmax)
    lw_min, lw_max = lw_range
    curv = _arc_curvatures(arcs, curvature)

    segments, colors, widths = [], [], []
    for arc_id, (n0, n1) in arcs.items():
        u = np.asarray(u_values[arc_id])
        x, y = _arc_positions(node_pos[n0], node_pos[n1], len(u), curv[arc_id])
        pts = np.column_stack([x, y]).reshape(-1, 1, 2)
        segments.append(np.concatenate([pts[:-1], pts[1:]], axis=1))
        seg_vals = (u[:-1] + u[1:]) / 2
        colors.append(seg_vals)
        widths.append(lw_min + (lw_max - lw_min) * norm(seg_vals))

    lc = LineCollection(
        np.concatenate(segments, axis=0),
        cmap=cmap, norm=norm,
        linewidths=np.concatenate(widths, axis=0),
        capstyle="round",
    )
    lc.set_array(np.concatenate(colors, axis=0))
    ax.add_collection(lc)

    labels = _default_labels(internal_nodes, external_bcs) | (node_labels or {})
    used = {key for ends in arcs.values() for key in ends}
    for key in used:
        x, y = node_pos[key]
        ax.scatter([x], [y], s=30, color="black", zorder=3)
        ax.annotate(labels.get(key, str(key)), (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)

    # limits from every drawn point (so bowed arcs aren't clipped), padded
    # equally on both axes (so a graph with all nodes on one line still shows)
    all_pts = np.concatenate(segments, axis=0).reshape(-1, 2)
    (x_lo, y_lo), (x_hi, y_hi) = all_pts.min(axis=0), all_pts.max(axis=0)
    pad = 0.15 * max(x_hi - x_lo, y_hi - y_lo, 1e-9)
    ax.set_xlim(x_lo - pad, x_hi + pad)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(lc, ax=ax)
    return fig, ax


def plot_density_snapshots(
    node_pos: dict[NodeKey, tuple[float, float]],
    internal_nodes: list[InternalNode],
    external_bcs: list[ExternalBC],
    u_minus: dict[int, np.ndarray],
    u_plus: dict[int, np.ndarray],
    indices: list[int],
    k: float,
    *,
    curvature: dict[int, float] | None = None,
    node_labels: dict[NodeKey, str] | None = None,
    cmap: str = "jet",
    lw_range: tuple[float, float] = (1.5, 8.0),
    ncols: int = 3,
    panel_width: float = 5.0,
) -> tuple[plt.Figure, np.ndarray]:
    """One panel per time row in indices, all on one colour scale with a
    single shared colorbar"""
    vmax = max(
        float((u_minus[arc][idx] + u_plus[arc][idx]).max())
        for arc in u_minus
        for idx in indices
    )
    ncols = min(ncols, len(indices))
    nrows = -(-len(indices) // ncols)
    # panels take the graph's own aspect ratio (plus room for the title), so
    # equal-aspect axes don't leave large gaps between rows
    xs, ys = zip(*node_pos.values())
    aspect = (max(ys) - min(ys) + 1e-9) / (max(xs) - min(xs) + 1e-9)
    panel_height = min(max(panel_width * aspect, 1.5), 2 * panel_width) + 0.8
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel_width * ncols, panel_height * nrows),
        squeeze=False, layout="constrained",
    )
    for ax, idx in zip(axes.flat, indices):
        u_values = {arc: u_minus[arc][idx] + u_plus[arc][idx] for arc in u_minus}
        plot_network_density(
            node_pos, internal_nodes, external_bcs, u_values,
            curvature=curvature, node_labels=node_labels, cmap=cmap,
            vmin=0.0, vmax=vmax, lw_range=lw_range,
            title=f"$u$ on the graph, t={idx * k:.2f}", ax=ax, colorbar=False,
        )
    for ax in axes.flat[len(indices):]:
        ax.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, vmax))
    fig.colorbar(sm, ax=axes.ravel().tolist(), label="$u$")
    return fig, axes
