from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_client import products, run_agents  # noqa: E402
from components import empty_state, page_setup, require_api, show_error  # noqa: E402

page_setup("Agent Activity")
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

product_code = st.selectbox("Product", [p["product_code"] for p in catalog])
horizon = st.slider("Horizon", 1, 14, 7)
if st.button("Run multi-agent workflow", type="primary"):
    try:
        with st.spinner("Running Forecast → Critic → Planner → Explanation..."):
            result = run_agents(product_code, horizon, "lightgbm")
            st.session_state["agent_run"] = result
    except Exception as exc:  # noqa: BLE001
        show_error(exc)

result = st.session_state.get("agent_run")
if not result:
    empty_state("Run the LangGraph workflow to inspect agent statuses and evidence-backed summaries.")
    st.stop()

st.subheader("Workflow status")
c1, c2, c3 = st.columns(3)
c1.metric("Run ID", result["run_id"][:8])
c2.metric("Status", result["status"])
c3.metric("Confidence", result["confidence"])
if result["review_required"]:
    st.warning("The critic flagged this run for review. The planner did not treat the forecast as fully reliable.")

st.subheader("Explanation")
st.write(result.get("explanation") or "No explanation stored.")

if result.get("warnings"):
    st.subheader("Validation warnings")
    for warning in result["warnings"]:
        st.write(f"- {warning}")

if result.get("recommendation"):
    st.subheader("Planner output")
    st.json(result["recommendation"])

st.subheader("Agent execution records")
logs = result.get("logs") or []
if logs:
    table = pd.DataFrame(
        [
            {
                "agent": log["agent_name"],
                "status": log["status"],
                "started_at": log["started_at"],
                "completed_at": log["completed_at"],
                "warnings": ", ".join(log.get("warnings") or []),
                "error": log.get("error_message") or "",
            }
            for log in logs
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    for log in logs:
        with st.expander(f"{log['agent_name']} ({log['status']})"):
            st.caption("Structured summaries only. Hidden chain-of-thought is not stored.")
            st.write("Input summary")
            st.code(log["input_summary"])
            st.write("Output summary")
            st.code(log["output_summary"])
else:
    empty_state("No agent logs returned.")
