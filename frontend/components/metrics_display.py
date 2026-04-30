import streamlit as st


def show_metrics(metrics: dict) -> None:
    """Display model evaluation metrics as a card row."""
    st.markdown("#### 📊 Evaluation Metrics (Test Set)")

    definitions = {
        "RMSE": ("📉 RMSE", "Root Mean Squared Error — penalizes large errors"),
        "MAE": ("📊 MAE", "Mean Absolute Error — average price error"),
        "MAPE": ("📐 MAPE %", "Mean Absolute % Error — scale-independent"),
        "Theils_U": ("🎯 Theil's U", "0=perfect, 1=naive forecast equivalent"),
        "Directional_Accuracy": ("🧭 Direction %", "% correct up/down predictions"),
    }

    cols = st.columns(5)
    for i, (key, (label, help_text)) in enumerate(definitions.items()):
        val = metrics.get(key, 0.0)
        with cols[i]:
            st.metric(
                label=label,
                value=f"{val:.4f}" if key != "Directional_Accuracy"
                      else f"{val:.1f}%",
                help=help_text
            )
