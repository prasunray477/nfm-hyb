import plotly.graph_objects as go
import numpy as np
import streamlit as st
from typing import List


def plot_neutrosophic_triples(
    T_vals: List[float],
    I_vals: List[float],
    F_vals: List[float],
    title: str = "Neutrosophic Triple Time Series"
) -> None:
    """
    Visualize T, I, F membership values over time.
    Helps verify neutrosophic normalization quality.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=T_vals, name="T — Truth (Bullish)",
        line=dict(color="#00CC96", width=1.5),
        fill='tozeroy', fillcolor='rgba(0,204,150,0.1)'
    ))
    fig.add_trace(go.Scatter(
        y=I_vals, name="I — Indeterminacy (Entropy)",
        line=dict(color="#FFA15A", width=1.5),
        fill='tozeroy', fillcolor='rgba(255,161,90,0.1)'
    ))
    fig.add_trace(go.Scatter(
        y=F_vals, name="F — Falsity (Bearish)",
        line=dict(color="#EF553B", width=1.5),
        fill='tozeroy', fillcolor='rgba(239,85,59,0.1)'
    ))

    fig.add_hline(
        y=0.333, line_dash="dot",
        line_color="gray", opacity=0.5,
        annotation_text="Neutral (1/3)"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Trading Day",
        yaxis_title="Membership Value [0, 1]",
        yaxis=dict(range=[0, 1]),
        template="plotly_dark",
        height=350,
        legend=dict(orientation="h", y=1.02)
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_triple_scatter(
    T_vals: np.ndarray,
    I_vals: np.ndarray,
    F_vals: np.ndarray
) -> None:
    """
    3D scatter of neutrosophic triples.
    Ideal distribution: spread across the unit cube,
    not clustered at a single point.
    """
    fig = go.Figure(go.Scatter3d(
        x=T_vals, y=I_vals, z=F_vals,
        mode='markers',
        marker=dict(
            size=2,
            color=T_vals,
            colorscale='RdYlGn',
            opacity=0.6,
            colorbar=dict(title="T (Bullish)")
        )
    ))

    fig.update_layout(
        title="Neutrosophic Triple Distribution (T, I, F)",
        scene=dict(
            xaxis_title="T — Truth",
            yaxis_title="I — Indeterminacy",
            zaxis_title="F — Falsity",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            zaxis=dict(range=[0, 1])
        ),
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
