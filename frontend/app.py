from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import latest_evaluation, list_forecasts, products
from components import empty_state, page_setup, require_api, show_error

page_setup("Executive Dashboard")
if not require_api():
    st.stop()

try:
    catalog = products()
    forecasts = list_forecasts()
    evaluation = latest_evaluation()
except Exception as exc:  # noqa: BLE001
    show_error(exc)
    st.stop()

at_risk = 0
try:
    from api_client import get_inventory

    for item in catalog[:40]:
        inv = get_inventory(item["product_code"])
        if inv["current_stock"] <= max(inv["lead_time_days"] * 2, 5):
            at_risk += 1
except Exception:
    at_risk = 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Products available", len(catalog))
col2.metric("Forecast runs", len(forecasts))
col3.metric("Low-stock flags (heuristic)", at_risk)
if evaluation.get("available"):
    metrics = evaluation.get("metrics", {}).get("accuracy", {}).get("models", {})
    lgbm = metrics.get("lightgbm", {})
    col4.metric("Latest LightGBM WAPE", lgbm.get("wape", "n/a"))
else:
    col4.metric("Latest model performance", "Not evaluated")

st.subheader("Catalog snapshot")
if catalog:
    st.dataframe(pd.DataFrame(catalog), use_container_width=True, hide_index=True)
else:
    empty_state("No products loaded. Initialize sample data from the API startup or scripts/generate_sample_data.py.")

st.subheader("Recent forecast runs")
if forecasts:
    frame = pd.DataFrame(
        [
            {
                "id": row["id"],
                "product_code": row["product_code"],
                "model_name": row["model_name"],
                "horizon": row["forecast_horizon"],
                "status": row["status"],
                "created_at": row["created_at"],
            }
            for row in forecasts
        ]
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)
    totals = []
    for row in forecasts[:15]:
        total = sum(p["predicted_demand"] for p in row.get("results", []))
        totals.append({"created_at": row["created_at"], "predicted_7d": total, "product_code": row["product_code"]})
    if totals:
        fig = px.bar(pd.DataFrame(totals), x="created_at", y="predicted_7d", color="product_code", title="Recent forecast totals")
        st.plotly_chart(fig, use_container_width=True)
else:
    empty_state("No forecasts yet. Create one on the Demand Forecasting page.")

st.subheader("Evaluation snapshot")
if evaluation.get("available"):
    inv = evaluation["metrics"].get("inventory_simulation", {})
    st.caption(inv.get("note", "Inventory figures are simulated."))
    proposed = inv.get("proposed_policy") or {}
    naive = inv.get("naive_policy") or {}
    if proposed:
        cost_df = pd.DataFrame(
            [
                {"policy": "proposed", **proposed},
                {"policy": "naive", **naive},
            ]
        )
        fig = px.bar(cost_df, x="policy", y="total_cost", title="Simulated total inventory cost")
        st.plotly_chart(fig, use_container_width=True)
else:
    empty_state("Run an evaluation on the Model Evaluation page to populate measured metrics and simulated costs.")
