from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_client import create_forecast, history, products  # noqa: E402
from components import empty_state, page_setup, require_api, show_error  # noqa: E402

page_setup("Demand Forecasting")
if not require_api():
    st.stop()

try:
    catalog = products()
except Exception as exc:  # noqa: BLE001
    show_error(exc)
    st.stop()

if not catalog:
    empty_state("No products available.")
    st.stop()

frame = pd.DataFrame(catalog)
stores = sorted(frame["store_code"].unique().tolist())
store = st.sidebar.selectbox("Store", stores)
cats = sorted(frame.loc[frame["store_code"] == store, "category"].unique().tolist())
category = st.sidebar.selectbox("Category", cats)
options = frame[(frame["store_code"] == store) & (frame["category"] == category)]
product_code = st.sidebar.selectbox("Product", options["product_code"].tolist())
horizon = st.sidebar.slider("Horizon (days)", 1, 14, 7)
run_lgbm = st.sidebar.button("Forecast with LightGBM", type="primary")
run_base = st.sidebar.button("Forecast with seasonal naive")

try:
    hist = history(product_code)
except Exception as exc:  # noqa: BLE001
    show_error(exc)
    st.stop()

st.caption(f"History source: {hist['data_label']}")
hist_df = pd.DataFrame(hist["points"])
hist_df["date"] = pd.to_datetime(hist_df["date"])

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist_df["date"], y=hist_df["demand"], name="Actual demand", mode="lines"))

def overlay(run: dict, name: str) -> None:
    pts = pd.DataFrame(run["results"])
    pts["forecast_date"] = pd.to_datetime(pts["forecast_date"])
    fig.add_trace(go.Scatter(x=pts["forecast_date"], y=pts["predicted_demand"], name=name, mode="lines+markers"))
    if pts["lower_bound"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=list(pts["forecast_date"]) + list(pts["forecast_date"][::-1]),
                y=list(pts["upper_bound"]) + list(pts["lower_bound"][::-1]),
                fill="toself",
                name=f"{name} interval",
                line={"width": 0},
                opacity=0.2,
            )
        )

if run_lgbm or run_base:
    try:
        with st.spinner("Generating forecast from the trained model..."):
            if run_lgbm:
                lgbm_run = create_forecast(product_code, horizon, "lightgbm")
                st.session_state["lgbm_run"] = lgbm_run
            if run_base:
                base_run = create_forecast(product_code, horizon, "seasonal_naive")
                st.session_state["base_run"] = base_run
    except Exception as exc:  # noqa: BLE001
        show_error(exc)

lgbm_run = st.session_state.get("lgbm_run")
base_run = st.session_state.get("base_run")
if lgbm_run and lgbm_run.get("product_code") == product_code:
    overlay(lgbm_run, "LightGBM")
    st.subheader("LightGBM metadata")
    st.json(
        {
            "model_name": lgbm_run["model_name"],
            "model_version": lgbm_run["model_version"],
            "status": lgbm_run["status"],
            "warnings": lgbm_run.get("warnings", []),
        }
    )
    st.dataframe(pd.DataFrame(lgbm_run["results"]), use_container_width=True, hide_index=True)
if base_run and base_run.get("product_code") == product_code:
    overlay(base_run, "Seasonal naive")
    st.subheader("Baseline metadata")
    st.json({"model_name": base_run["model_name"], "model_version": base_run["model_version"]})

fig.update_layout(title=f"Demand history and forecasts for {product_code}", xaxis_title="Date", yaxis_title="Units")
st.plotly_chart(fig, use_container_width=True)

if not lgbm_run and not base_run:
    empty_state("Trigger a forecast to overlay predictions. Numbers come from the ML service, not placeholder values.")
