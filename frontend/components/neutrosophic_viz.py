import plotly.graph_objects as go
import numpy as np
import streamlit as st
from typing import List


# ── Tab 2: Market Signal Encoding chart ────────────────────────────────────────

def plot_neutrosophic_mapping(triples, dates_trimmed, ticker):
    """
    Stacked area chart of neutrosophic (T, I, F) membership values
    over the full dataset timeline.
    """
    T_vals = triples[:, 0]
    I_vals = triples[:, 1]
    F_vals = triples[:, 2]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates_trimmed, y=T_vals,
        name='Truth (T) — Bullish Strength',
        fill='tozeroy',
        mode='lines',
        line=dict(color='#0D9488', width=1.2),
        fillcolor='rgba(13,148,136,0.18)',
        hovertemplate='%{x|%d %b %Y}<br>T: %{y:.3f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=dates_trimmed, y=F_vals,
        name='Falsity (F) — Bearish Pressure',
        fill='tozeroy',
        mode='lines',
        line=dict(color='#F43F5E', width=1.2),
        fillcolor='rgba(244,63,94,0.18)',
        hovertemplate='%{x|%d %b %Y}<br>F: %{y:.3f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=dates_trimmed, y=I_vals,
        name='Indeterminacy (I) — Market Uncertainty',
        mode='lines',
        line=dict(color='#F59E0B', width=1.5, dash='dot'),
        hovertemplate='%{x|%d %b %Y}<br>I: %{y:.3f}<extra></extra>'
    ))
    fig.add_hline(
        y=0.333, line_dash='dot', line_color='white',
        opacity=0.35, annotation_text='Neutral (1/3)',
        annotation_font_size=9, annotation_font_color='#94A3B8'
    )
    fig.update_layout(
        title=f'{ticker} — Neutrosophic Market Signal Encoding (T, I, F)',
        xaxis_title='Date',
        yaxis_title='Membership Value [0 – 1]',
        yaxis=dict(range=[0, 1]),
        template='plotly_dark',
        height=360,
        legend=dict(orientation='h', y=1.08),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interpretation row
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Avg Bullish Signal (T)", f"{T_vals.mean():.3f}",
        help="Mean truth membership across all days. Above 0.5 = structurally bullish dataset."
    )
    col2.metric(
        "Avg Uncertainty (I)", f"{I_vals.mean():.3f}",
        help="Mean indeterminacy. Closer to 1.0 = highly inconsistent market history."
    )
    col3.metric(
        "Avg Bearish Signal (F)", f"{F_vals.mean():.3f}",
        help="Mean falsity membership. Above 0.5 = structurally bearish dataset."
    )

    with st.expander("How Market Signal Encoding Works"):
        st.markdown("""
Market Signal Encoding (formally called Neutrosophic Logic) transforms each
trading day's price movement into three simultaneous signals rather than a
single number.

**Truth (T)** measures bullish momentum — how strongly prices are rising on a
given day relative to the typical daily swing seen during training. A value
close to 1.0 means the day was strongly up; a value near 0 means prices fell
sharply.

**Falsity (F)** is the mirror: it measures bearish pressure. A high F value
means prices declined significantly. Note that T and F are not simply
opposites — on a sideways or ambiguous day, both can be moderate.

**Indeterminacy (I)** captures something neither T nor F can see: how
*inconsistent* the recent market has been. It is calculated as the Shannon
information entropy of the last 10 days of price movements. When the market
has been trending cleanly in one direction, entropy is low and I ≈ 0. When
the market has been whipsawing — up one day, down the next — entropy is high
and I approaches 1.

By feeding all three signals into the deep learning model instead of raw prices,
the model learns not just where prices went, but how *confident* or *confused*
the market has been. This extra context is what gives the hybrid system an edge
over conventional normalization approaches.
        """)


# ── Legacy functions kept for backward compatibility ───────────────────────────

def plot_neutrosophic_triples(
    T_vals: List[float],
    I_vals: List[float],
    F_vals: List[float],
    title: str = "Neutrosophic Triple Time Series"
) -> None:
    """
    Visualize T, I, F membership values over time.
    Kept for backward compatibility.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=T_vals, name="T — Truth (Bullish)",
        line=dict(color="#00CC96", width=1.5),
        fill='tozeroy', fillcolor='rgba(0,204,150,0.1)'
    ))
    fig.add_trace(go.Scatter(
        y=I_vals, name="I — Indeterminacy (Market Uncertainty)",
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
    Kept for backward compatibility.
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
