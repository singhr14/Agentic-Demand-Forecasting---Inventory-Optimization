from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_client import get_inventory, products, recommend, update_inventory  # noqa: E402
from components import empty_state, page_setup, require_api, show_error  # noqa: E402

page_setup("Inventory Optimization")
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

product_code = st.sidebar.selectbox("Product", [p["product_code"] for p in catalog])
try:
    state = get_inventory(product_code)
except Exception as exc:  # noqa: BLE001
    show_error(exc)
    st.stop()

st.subheader("Simulation assumptions")
st.caption("These values are planner inputs. They are not fields in the M5 sales files.")
c1, c2, c3 = st.columns(3)
current_stock = c1.number_input("Current stock", min_value=0.0, value=float(state["current_stock"]))
on_order = c2.number_input("On-order quantity", min_value=0.0, value=float(state["on_order_quantity"]))
lead_time = c3.number_input("Lead time (days)", min_value=1, max_value=30, value=int(state["lead_time_days"]))
c4, c5, c6 = st.columns(3)
holding = c4.number_input("Holding cost / unit", min_value=0.0, value=float(state["holding_cost_per_unit"]))
ordering = c5.number_input("Ordering cost", min_value=0.0, value=float(state["ordering_cost"]))
service = c6.number_input("Service level target", min_value=0.51, max_value=0.99, value=float(state["service_level_target"]))

if st.button("Save inventory state"):
    try:
        state = update_inventory(
            product_code,
            {
                "current_stock": current_stock,
                "on_order_quantity": on_order,
                "lead_time_days": int(lead_time),
                "holding_cost_per_unit": holding,
                "ordering_cost": ordering,
                "service_level_target": service,
            },
        )
        st.success("Inventory state saved.")
    except Exception as exc:  # noqa: BLE001
        show_error(exc)

if st.button("Recalculate recommendation", type="primary"):
    try:
        with st.spinner("Forecasting demand and calculating reorder quantities..."):
            rec = recommend(
                {
                    "product_code": product_code,
                    "horizon": 7,
                    "model_name": "lightgbm",
                    "current_stock": current_stock,
                    "on_order_quantity": on_order,
                    "lead_time_days": int(lead_time),
                    "holding_cost_per_unit": holding,
                    "ordering_cost": ordering,
                    "service_level_target": service,
                }
            )
            st.session_state["recommendation"] = rec
    except Exception as exc:  # noqa: BLE001
        show_error(exc)

rec = st.session_state.get("recommendation")
if rec and rec.get("product_code") == product_code:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Expected demand over lead time", rec["expected_demand_lead_time"])
    m2.metric("Safety stock", rec["safety_stock"])
    m3.metric("Reorder point", rec["reorder_point"])
    m4.metric("Recommended order quantity", rec["recommended_order_quantity"])
    n1, n2, n3 = st.columns(3)
    n1.metric("Stockout risk (approx.)", rec["expected_stockout_risk"])
    n2.metric("Estimated holding cost", rec["estimated_holding_cost"])
    n3.metric("Estimated ordering cost", rec["estimated_ordering_cost"])
    if rec["stockout_flag"]:
        st.warning("Inventory position is below the reorder point.")
    if rec["excess_inventory_flag"]:
        st.info("Inventory position is well above the reorder point.")
    st.subheader("Documented assumptions")
    st.json(rec["assumptions"])
else:
    empty_state("Calculate a recommendation to see planner outputs from the inventory service.")
