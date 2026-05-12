import streamlit as st


def show_metrics_v2(metrics: dict) -> None:
    """
    Display model evaluation metrics with user-friendly renamed labels.
    Replaces show_metrics() entirely.
    """
    st.markdown("#### Performance Metrics — Evaluation Period")
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        label="Forecast Error (RMSE)",
        value=f"{metrics['RMSE']:.4f}",
        help=(
            "Root Mean Squared Error. Lower is better. "
            "Penalizes large prediction misses more heavily."
        )
    )
    col2.metric(
        label="Average Error (MAE)",
        value=f"{metrics['MAE']:.4f}",
        help=(
            "Mean Absolute Error. Average price difference between "
            "forecast and actual in currency units."
        )
    )
    col3.metric(
        label="Average Error Rate",
        value=f"{metrics['MAPE']:.2f}%",
        help=(
            "Mean Absolute Percentage Error. Scale-independent measure. "
            "Below 2% is considered strong."
        )
    )
    col4.metric(
        label="Forecast Reliability",
        value=f"{metrics['Theils_U']:.4f}",
        help=(
            "Forecast Reliability Score (Theil's U). "
            "0 = perfect. Below 0.05 = highly reliable."
        )
    )
    col5.metric(
        label="Trend Prediction Accuracy",
        value=f"{metrics['Directional_Accuracy']:.1f}%",
        help=(
            "Percentage of days the model correctly predicted the direction "
            "(up or down). Above 55% outperforms random."
        )
    )


# Backward-compatible alias
def show_metrics(metrics: dict) -> None:
    """Legacy alias — delegates to show_metrics_v2."""
    show_metrics_v2(metrics)
