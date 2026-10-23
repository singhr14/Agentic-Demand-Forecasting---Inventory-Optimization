from __future__ import annotations

import streamlit as st

from api_client import APIError, BackendUnavailable, health


def page_setup(title: str) -> None:
    st.set_page_config(page_title=title, layout="wide", page_icon="📦")
    st.title(title)


def show_error(exc: Exception) -> None:
    if isinstance(exc, BackendUnavailable):
        st.error(str(exc))
        st.info("Start the API with: `uvicorn backend.app.main:app --reload --port 8000`")
        return
    if isinstance(exc, APIError):
        st.error(f"API error ({exc.status_code}): {exc}")
        return
    st.error(str(exc))


def require_api() -> dict | None:
    try:
        status = health()
        if not status.get("llm_configured"):
            st.caption("Running without an LLM key. Explanations use the deterministic fallback. OpenRouter is optional.")
        if status.get("sample_data_mode"):
            st.caption("Sample-data mode is on. Charts and KPIs are computed from labeled demonstration data.")
        return status
    except Exception as exc:  # noqa: BLE001
        show_error(exc)
        return None


def empty_state(message: str) -> None:
    st.info(message)
