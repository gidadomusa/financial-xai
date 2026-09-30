"""Streamlit dashboard for portfolio forecasting and explainability."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .dataset import generate_synthetic_portfolio_data
from .pipeline import run_portfolio_forecasting_pipeline


REQUIRED_COLUMNS = {"date", "asset_id", "open", "high", "low", "close", "volume"}
MODEL_OPTIONS = {
    "Random forest": "random_forest",
    "Logistic regression": "logistic",
    "Gradient boosting": "gradient_boosting",
}


def prepare_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate OHLCV data before passing it to the pipeline."""
    data = data.copy()
    data.columns = [str(column).strip().lower() for column in data.columns]
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        required = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {required}")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", "asset_id", "open", "high", "low", "close", "volume"])
    data["asset_id"] = data["asset_id"].astype(str)
    data = data.sort_values(["asset_id", "date"])

    if data.empty:
        raise ValueError("No complete OHLCV rows remain after cleaning the input.")
    short_histories = data.groupby("asset_id").size()
    if (short_histories < 140).any():
        assets = ", ".join(short_histories[short_histories < 140].index.astype(str))
        raise ValueError(f"At least 140 rows per asset are needed for rolling features: {assets}")
    return data.reset_index(drop=True)


def render_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
        :root { --ink: #18221e; --muted: #69756e; --line: #dce3dc; --paper: #f5f7f3; --green: #176b4b; --lime: #d8f36a; }
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
        .stApp { background: var(--paper); }
        [data-testid="stSidebar"] { background: #e9eee8; border-right: 1px solid var(--line); }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.8rem; }
        .block-container { max-width: 1440px; padding-top: 2.4rem; padding-bottom: 3rem; }
        h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
        h1 { font-size: 2.4rem; font-weight: 600; line-height: 1.08; }
        h2 { font-size: 1.2rem; font-weight: 600; }
        .eyebrow { font-family: 'IBM Plex Mono', monospace; text-transform: uppercase; font-size: .72rem; color: var(--green); letter-spacing: 0; }
        .lede { max-width: 48rem; margin-top: -.6rem; color: var(--muted); }
        .section-rule { border-top: 1px solid var(--line); margin: 1.3rem 0 1rem; }
        div[data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); border-radius: 6px; padding: 1rem 1.1rem; }
        div[data-testid="stMetricLabel"] { color: var(--muted); }
        div[data-testid="stMetricValue"] { color: var(--ink); font-size: 1.65rem; }
        .run-note { color: var(--muted); font-size: .86rem; }
        .stButton > button[kind="primary"] { background: var(--green); border-color: var(--green); color: white; border-radius: 4px; min-height: 2.8rem; }
        .stButton > button[kind="primary"]:hover { background: #10553a; border-color: #10553a; }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Financial XAI | Market Lab", page_icon="FX", layout="wide")
    render_styles()

    st.sidebar.markdown('<div class="eyebrow">Financial XAI / Market Lab</div>', unsafe_allow_html=True)
    st.sidebar.title("Forecast setup")
    data_mode = st.sidebar.radio("Market data", ["Synthetic portfolio", "Upload CSV"])
    uploaded_file = None
    if data_mode == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("OHLCV market data", type=["csv"])
        st.sidebar.caption("Required: date, asset_id, open, high, low, close, volume")
    else:
        asset_count = st.sidebar.slider("Assets", min_value=1, max_value=8, value=3)
        history_size = st.sidebar.slider("History per asset", min_value=500, max_value=2500, value=1500, step=250)

    model_label = st.sidebar.selectbox("Classification model", list(MODEL_OPTIONS))
    horizon = st.sidebar.selectbox("Forecast horizon", [1, 5, 20], index=1, format_func=lambda days: f"{days} trading days")
    run_clicked = st.sidebar.button("Run analysis", type="primary", use_container_width=True)
    st.sidebar.markdown('<p class="run-note">Predictions are experimental and are not investment advice.</p>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">Model validation / Walk-forward split</div>', unsafe_allow_html=True)
    st.title("Market signals, made legible.")
    st.markdown('<p class="lede">Train a directional return model, inspect its holdout performance, and see which market features are driving its forecasts.</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)

    if run_clicked:
        try:
            if data_mode == "Synthetic portfolio":
                market_data = generate_synthetic_portfolio_data(n_samples=history_size, n_assets=asset_count)
            elif uploaded_file is not None:
                market_data = pd.read_csv(uploaded_file)
            else:
                raise ValueError("Upload a CSV file to run the analysis.")

            market_data = prepare_market_data(market_data)
            with st.spinner("Engineering features, fitting the model, and calculating explanations..."):
                results = run_portfolio_forecasting_pipeline(
                    data=market_data,
                    target_return_period=horizon,
                    model_type=MODEL_OPTIONS[model_label],
                    return_explanation=True,
                )
            st.session_state["analysis"] = {
                "results": results,
                "data": market_data,
                "model": model_label,
                "horizon": horizon,
            }
        except Exception as error:
            st.error(f"Analysis could not be completed: {error}")

    analysis = st.session_state.get("analysis")
    if analysis is None:
        st.info("Choose a data source and run an analysis to see model performance and feature explanations.")
        st.markdown("#### Included in the analysis")
        left, right = st.columns(2)
        left.markdown("**Holdout performance**  \nAccuracy, precision, recall, F1, and ROC AUC on a time-ordered test split.")
        right.markdown("**Feature attribution**  \nGlobal SHAP importance for the most influential engineered market signals.")
        return

    results = analysis["results"]
    market_data = analysis["data"]
    metrics = results["metrics"]
    latest_features = results["X_test"].tail(1)
    latest_prediction = int(results["model"].predict(latest_features)[0])
    latest_probability = float(results["model"].predict_proba(latest_features)[0, 1])
    signal = "Positive return" if latest_prediction else "Non-positive return"

    st.markdown(f"**{analysis['model']}** · {analysis['horizon']}-day horizon · {market_data['asset_id'].nunique()} assets · {len(market_data):,} observations")
    metric_columns = st.columns(5)
    for column, label, key in zip(
        metric_columns,
        ["Accuracy", "Precision", "Recall", "F1 score", "ROC AUC"],
        ["accuracy", "precision", "recall", "f1", "roc_auc"],
    ):
        column.metric(label, f"{metrics[key]:.1%}")

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    chart_column, signal_column = st.columns([1.65, 1], gap="large")
    with chart_column:
        st.subheader("Relative price history")
        indexed_prices = market_data.pivot(index="date", columns="asset_id", values="close").sort_index()
        indexed_prices = indexed_prices.div(indexed_prices.iloc[0]).mul(100)
        st.line_chart(indexed_prices, height=330, y_label="Indexed close · start = 100")
    with signal_column:
        st.subheader("Latest holdout signal")
        st.metric("Model direction", signal, delta=f"{latest_probability:.1%} positive-class probability")
        confusion = pd.DataFrame(
            [[metrics["true_negatives"], metrics["false_positives"]], [metrics["false_negatives"], metrics["true_positives"]]],
            index=["Actual non-positive", "Actual positive"],
            columns=["Predicted non-positive", "Predicted positive"],
        )
        st.caption("Confusion matrix")
        st.dataframe(confusion, use_container_width=True)

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.subheader("What moved the model")
    importance = results["feature_importance"]
    if importance is None or importance.empty:
        st.warning("Feature attribution was not returned for this run.")
    else:
        importance_chart = importance.set_index("feature")["mean_abs_shap"].sort_values()
        st.bar_chart(importance_chart, horizontal=True, height=360, x_label="Mean absolute SHAP value")
        with st.expander("Feature importance values"):
            st.dataframe(importance, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()