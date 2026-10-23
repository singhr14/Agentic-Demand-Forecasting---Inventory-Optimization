from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_client import latest_evaluation, run_evaluation  # noqa: E402
from components import empty_state, page_setup, require_api, show_error  # noqa: E402

page_setup("Model Evaluation")
if not require_api():
    st.stop()

horizon = st.sidebar.slider("Holdout horizon", 7, 14, 7)
if st.button("Run chronological evaluation", type="primary"):
    try:
        with st.spinner("Training-time artifacts are reused; scoring runs in a worker thread..."):
            payload = run_evaluation(horizon)
            st.session_state["eval"] = {"available": True, "metrics": payload}
    except Exception as exc:  # noqa: BLE001
        show_error(exc)

payload = st.session_state.get("eval")
if not payload:
    try:
        payload = latest_evaluation()
    except Exception as exc:  # noqa: BLE001
        show_error(exc)
        st.stop()

if not payload.get("available") and "metrics" not in payload:
    empty_state("No evaluation results yet. Run the holdout comparison to compute MAE, RMSE, WAPE, and bias.")
    st.stop()

metrics = payload.get("metrics") or payload
accuracy = metrics.get("accuracy") or {}
models = accuracy.get("models") or {}
st.subheader("Measured forecast accuracy")
st.caption("Baseline and LightGBM are scored on the same chronological holdout dates and products.")
if models:
    table = pd.DataFrame(models).T.reset_index().rename(columns={"index": "model"})
    st.dataframe(table, use_container_width=True, hide_index=True)
    fig = px.bar(table, x="model", y="wape", title="WAPE on holdout (lower is better)")
    st.plotly_chart(fig, use_container_width=True)
else:
    empty_state("Accuracy block missing.")

st.subheader("Rolling backtest")
windows = (metrics.get("backtest") or {}).get("windows") or []
if windows:
    st.dataframe(pd.DataFrame(windows), use_container_width=True, hide_index=True)
else:
    st.info((metrics.get("backtest") or {}).get("note") or "No rolling windows available.")

st.subheader("Simulated inventory outcomes")
inv = metrics.get("inventory_simulation") or {}
st.caption(inv.get("note") or "Inventory results are simulated.")
proposed = inv.get("proposed_policy")
naive = inv.get("naive_policy")
if proposed and naive:
    cmp = pd.DataFrame([{"policy": "proposed", **proposed}, {"policy": "naive", **naive}])
    st.dataframe(cmp, use_container_width=True, hide_index=True)
    fig = px.bar(
        cmp.melt(id_vars=["policy"], value_vars=["stockout_rate", "fill_rate", "total_cost"]),
        x="variable",
        y="value",
        color="policy",
        barmode="group",
        title="Simulated service and cost comparison",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    empty_state("Inventory simulation has not been stored.")

st.subheader("Limitations")
for item in metrics.get("limitations") or []:
    st.write(f"- {item}")
st.write("- MAE: mean absolute error in units.")
st.write("- RMSE: root mean squared error in units.")
st.write("- WAPE: sum of absolute errors divided by sum of absolute actuals; 0 vs 0 is treated as 0.")
st.write("- Bias: mean(prediction - actual); positive values over-forecast.")
