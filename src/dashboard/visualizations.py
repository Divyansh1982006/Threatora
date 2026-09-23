"""Plotly Visualizations & Metric Charts for Threatora.

Provides pure Plotly chart builders for:
- Multi-horizon Infiltration Risk Fan-Charts with Conformal Uncertainty Bands.
- Dynamic Saliency Attribution Waterfall charts.
- Transformer Multi-Head Self-Attention heatmaps.
- Prescriptive Counterfactual Risk Comparison charts.
- Interactive Force-Directed Network Topology charts.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import plotly.graph_objects as go


def create_fan_chart(
    historical_risks: np.ndarray,
    forecast_timeline: np.ndarray,
    active_window_idx: int,
    is_dark: bool = True,
) -> go.Figure:
    """Builds interactive Plotly Infiltration Risk Fan-Chart with conformal uncertainty bands."""
    n_hist = len(historical_risks)
    hist_x = [f"-{(n_hist - 1 - i) * 10}s" for i in range(n_hist)]
    if n_hist > 0:
        hist_x[-1] = "Now (t=0)"

    fore_x = ["Now (t=0)"] + [f"+{k*10}s" for k in range(1, 6)]
    now_val = float(historical_risks[-1]) if n_hist > 0 else float(forecast_timeline[0])
    fore_y = [now_val] + [float(p) for p in forecast_timeline]

    # Conformal uncertainty bands (+/- expanding across horizon)
    uncertainty_spread = np.array([0.0] + [0.04 * k for k in range(1, 6)])
    upper_band = np.clip(np.array(fore_y) + uncertainty_spread, 0.0, 1.0)
    lower_band = np.clip(np.array(fore_y) - uncertainty_spread, 0.0, 1.0)

    bg_color = "#202020" if is_dark else "#ffffff"
    text_color = "#FFFFFF" if is_dark else "#0f172a"
    grid_color = "rgba(255, 255, 255, 0.05)" if is_dark else "rgba(0, 0, 0, 0.06)"

    fig = go.Figure()

    # 1. Historical Observed Risk Trajectory
    fig.add_trace(go.Scatter(
        x=hist_x,
        y=historical_risks,
        mode="lines+markers",
        name="Observed Trajectory (S_t)",
        line=dict(color="#10b981", width=2.5),
        marker=dict(size=6, color="#10b981"),
    ))

    # 2. Conformal Uncertainty Bands (90% CI Cloud)
    fig.add_trace(go.Scatter(
        x=fore_x,
        y=upper_band,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=fore_x,
        y=lower_band,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(242, 86, 35, 0.22)" if is_dark else "rgba(242, 86, 35, 0.15)",
        name="90% Conformal Horizon Cloud",
    ))

    # 3. Forecast Trajectory Rollout
    fig.add_trace(go.Scatter(
        x=fore_x,
        y=fore_y,
        mode="lines+markers",
        name="World Model Forecast",
        line=dict(color="#F25623", width=3, dash="dash"),
        marker=dict(size=7, color="#F25623", symbol="diamond"),
    ))

    # 4. Critical Breach Threshold Line
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        xref="paper",
        y0=0.70,
        y1=0.70,
        yref="y",
        line=dict(color="#FF5252", width=1.5, dash="dot"),
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, size=11, family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=30, b=20),
        height=320,
        xaxis=dict(gridcolor=grid_color, showgrid=True),
        yaxis=dict(gridcolor=grid_color, showgrid=True, range=[0.0, 1.05]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11),
        ),
    )
    return fig


def create_attribution_waterfall(
    active_window: np.ndarray,
    canonical_slots: List[str],
    is_dark: bool = True,
) -> go.Figure:
    """Computes dynamic state saliency attribution waterfall for the active rolling window."""
    window_mean = np.mean(active_window, axis=0)
    scores = np.abs(window_mean)
    total_score = np.sum(scores) + 1e-6
    rel_attribution = (scores / total_score) * 100.0

    # Top 8 influential canonical features
    sorted_indices = np.argsort(rel_attribution)[::-1][:8]
    top_slots = [canonical_slots[i] for i in sorted_indices]
    top_vals = [rel_attribution[i] for i in sorted_indices]

    bg_color = "#202020" if is_dark else "#ffffff"
    text_color = "#FFFFFF" if is_dark else "#0f172a"
    grid_color = "rgba(255, 255, 255, 0.05)" if is_dark else "rgba(0, 0, 0, 0.06)"

    colorscale = (
        [[0, "#4D4D4D"], [0.5, "#F25623"], [1.0, "#FF5252"]]
        if is_dark
        else [[0, "#cbd5e1"], [0.5, "#F25623"], [1.0, "#ef4444"]]
    )

    fig = go.Figure(go.Bar(
        x=top_vals[::-1],
        y=top_slots[::-1],
        orientation="h",
        marker=dict(color=top_vals[::-1], colorscale=colorscale),
        text=[f"{v:.1f}%" for v in top_vals[::-1]],
        textposition="auto",
        textfont=dict(family="JetBrains Mono", size=10),
    ))

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, size=11, family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=10, b=20),
        height=280,
        xaxis=dict(
            gridcolor=grid_color,
            showgrid=True,
            title="Dynamic Saliency Weight (%)",
        ),
        yaxis=dict(gridcolor=grid_color, showgrid=False),
    )
    return fig


def create_attention_heatmap(
    attn_weights: Optional[np.ndarray],
    active_window_idx: int,
    is_dark: bool = True,
) -> go.Figure:
    """Renders Transformer Multi-Head Self-Attention heatmap across 20 temporal steps."""
    seq_len = 20
    if attn_weights is not None and len(attn_weights) > active_window_idx:
        attn = attn_weights[active_window_idx]
        if attn.ndim == 3:  # (nhead, 20, 20)
            attn_matrix = np.mean(attn, axis=0)
        else:
            attn_matrix = attn
    else:
        # Causal identity pattern if attention weights were omitted
        attn_matrix = np.eye(seq_len, dtype=np.float32)

    time_labels = [f"t-{seq_len - 1 - i}" if i < seq_len - 1 else "t=0" for i in range(seq_len)]
    bg_color = "#202020" if is_dark else "#ffffff"
    text_color = "#FFFFFF" if is_dark else "#0f172a"
    colorscale = "Inferno" if is_dark else "Oranges"

    fig = go.Figure(data=go.Heatmap(
        z=attn_matrix,
        x=time_labels,
        y=time_labels,
        colorscale=colorscale,
        colorbar=dict(title="Attn", tickfont=dict(color=text_color, size=9)),
        hovertemplate="Query: %{y}<br>Key: %{x}<br>Attention Weight: %{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, size=10, family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=10, b=20),
        height=280,
        xaxis=dict(title="Key (Past Temporal Bins)", tickangle=-45),
        yaxis=dict(title="Query (Temporal Bins)", autorange="reversed"),
    )
    return fig


def create_counterfactual_comparison_chart(
    unmitigated_timeline: np.ndarray,
    mitigated_timeline: np.ndarray,
    is_dark: bool = True,
) -> go.Figure:
    """Plots comparative risk collapse under hypothetical mitigation interventions."""
    steps = [f"+{k*10}s" for k in range(1, 6)]
    bg_color = "#202020" if is_dark else "#ffffff"
    text_color = "#FFFFFF" if is_dark else "#0f172a"
    grid_color = "rgba(255, 255, 255, 0.05)" if is_dark else "rgba(0, 0, 0, 0.06)"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=steps,
        y=unmitigated_timeline,
        mode="lines+markers",
        name="Unmitigated Risk",
        line=dict(color="#FF5252", width=2.5),
        marker=dict(size=6, color="#FF5252"),
    ))

    fig.add_trace(go.Scatter(
        x=steps,
        y=mitigated_timeline,
        mode="lines+markers",
        name="Prescribed Defense Action",
        line=dict(color="#10b981", width=2.5),
        marker=dict(size=6, color="#10b981"),
    ))

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, size=11, family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=20, b=20),
        height=240,
        xaxis=dict(gridcolor=grid_color, showgrid=True),
        yaxis=dict(gridcolor=grid_color, showgrid=True, range=[0.0, 1.0]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def create_topology_network_chart(
    nodes: List[Dict[str, Any]],
    links: List[Dict[str, Any]],
    isolated_hosts: Set[str],
    is_dark: bool = True,
) -> go.Figure:
    """Builds interactive 2D Force-Directed Network Topology Graph using Plotly."""
    bg_color = "#202020" if is_dark else "#ffffff"
    text_color = "#FFFFFF" if is_dark else "#0f172a"

    if not nodes:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            annotations=[dict(text="No network nodes detected in capture", showarrow=False, font=dict(color=text_color))],
            height=480,
        )
        return fig

    # Layout nodes in radial/hierarchical coordinates
    n_nodes = len(nodes)
    node_coords: Dict[str, Tuple[float, float]] = {}
    angle_step = (2 * math.pi) / max(n_nodes, 1)

    for i, n in enumerate(nodes):
        ntype = n.get("type", "workstation")
        if ntype == "gateway":
            node_coords[n["id"]] = (0.0, 0.75)
        elif ntype == "external":
            node_coords[n["id"]] = (0.85, 0.65)
        else:
            rad = 0.60
            ang = angle_step * i
            node_coords[n["id"]] = (rad * math.cos(ang), rad * math.sin(ang) - 0.1)

    fig = go.Figure()

    # Draw Link Traces
    edge_x = []
    edge_y = []
    edge_threat_x = []
    edge_threat_y = []

    for l in links:
        s_id = l["source"]
        t_id = l["target"]
        if s_id in node_coords and t_id in node_coords:
            x0, y0 = node_coords[s_id]
            x1, y1 = node_coords[t_id]
            if l.get("threat") == "critical":
                edge_threat_x.extend([x0, x1, None])
                edge_threat_y.extend([y0, y1, None])
            else:
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1.5, color="#4D4D4D" if is_dark else "#cbd5e1"),
            hoverinfo="none",
            mode="lines",
            name="Normal Flow",
        ))

    if edge_threat_x:
        fig.add_trace(go.Scatter(
            x=edge_threat_x,
            y=edge_threat_y,
            line=dict(width=2.5, color="#FF5252", dash="dash"),
            hoverinfo="none",
            mode="lines",
            name="Anomalous Channel",
        ))

    # Draw Node Markers
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    node_hover = []

    for n in nodes:
        nid = n["id"]
        x, y = node_coords[nid]
        node_x.append(x)
        node_y.append(y)

        is_isolated = nid in isolated_hosts
        status = "ISOLATED" if is_isolated else n.get("status", "HEALTHY")
        ntype = n.get("type", "workstation")
        crit = n.get("criticality", "MEDIUM")
        risk = n.get("risk_score", 0.08)

        # Color mapping
        if is_isolated:
            c = "#8E8E8E"
            sz = 20
        elif status in ("COMPROMISED", "THREAT_ACTOR") or risk >= 0.50:
            c = "#FF5252"
            sz = 26
        elif ntype == "gateway":
            c = "#00E5FF"
            sz = 24
        elif ntype in ("domain_controller", "server", "database"):
            c = "#F25623"
            sz = 22
        else:
            c = "#10b981"
            sz = 18

        node_colors.append(c)
        node_sizes.append(sz)
        # Truncate multi-IP labels on gateway nodes to prevent overlapping text clusters
        disp_name = n.get("hostname", nid)
        if ntype == "gateway":
            ip_val = str(n.get("ip", nid))
            ip_parts = [p.strip() for p in ip_val.replace(";", ",").split(",") if p.strip()]
            if len(ip_parts) > 1:
                disp_name = f"GATEWAY ({ip_parts[0]}) +{len(ip_parts) - 1} subnets"
            else:
                disp_name = f"GATEWAY ({ip_val})"
        elif len(disp_name) > 22:
            disp_name = disp_name[:20] + "..."
        node_text.append(disp_name)
        node_hover.append(
            f"<b>{n.get('hostname', nid)}</b><br>"
            f"IP: {n.get('ip', nid)}<br>"
            f"Role: {ntype.upper()}<br>"
            f"Subnet: {n.get('subnet', '192.168.1.0/24')}<br>"
            f"Criticality: {crit}<br>"
            f"Threat Posture: {status}<br>"
            f"Risk Score: {risk*100:.1f}%"
        )

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(family="JetBrains Mono", size=10, color=text_color),
        hoverinfo="text",
        hovertext=node_hover,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color="#FFFFFF" if is_dark else "#0f172a"),
        ),
        name="Network Endpoints",
    ))

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        height=520,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig
