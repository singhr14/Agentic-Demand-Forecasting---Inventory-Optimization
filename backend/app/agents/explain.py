from __future__ import annotations

import logging

from backend.app.agents.llm import build_chat_model

logger = logging.getLogger(__name__)

SYSTEM = (
    "You explain retail demand forecasts and inventory recommendations. "
    "Use only the provided evidence. Do not invent metrics, sales, or stock figures. "
    "Keep the answer under 180 words."
)


def deterministic_explanation(evidence: dict) -> str:
    forecast = evidence.get("forecast", [])
    critic = evidence.get("critic", {})
    plan = evidence.get("plan") or {}
    hist_mean = evidence.get("history_mean", 0)
    recent = evidence.get("recent_demand", [])[-7:]
    events = [r.get("event_name_1") for r in recent if r.get("event_name_1")]
    prices = [r.get("sell_price") for r in recent if r.get("sell_price") is not None]
    pred_sum = sum(forecast) if forecast else 0
    parts = [
        f"Seven-day predicted demand totals {pred_sum:.1f} units "
        f"(recent daily mean {hist_mean:.1f}).",
        f"Inventory position implies reorder point {plan.get('reorder_point', 'n/a')} "
        f"and recommended order quantity {plan.get('recommended_order_quantity', 'n/a')}.",
    ]
    if events:
        parts.append(f"Recent calendar events in the evidence: {', '.join(str(e) for e in events if e)}.")
    if len(prices) >= 2 and prices[0] and prices[-1] and prices[-1] != prices[0]:
        parts.append(f"Selling price moved from {prices[0]} to {prices[-1]} in the recent window.")
    warnings = critic.get("warnings") or []
    if warnings:
        parts.append("Critic warnings: " + "; ".join(warnings))
    if evidence.get("synthetic"):
        parts.append("Figures are computed from labeled demonstration data, not live retailer POS feeds.")
    parts.append(
        "Safety stock uses a z-score times daily residual sigma times sqrt(lead time). "
        "Inventory costs and on-hand stock are simulation inputs, not M5 dataset fields."
    )
    if critic.get("review_required"):
        parts.append("This run is flagged for review and should not be treated as a high-confidence plan.")
    return " ".join(parts)


def explain(evidence: dict) -> str:
    fallback = deterministic_explanation(evidence)
    llm = build_chat_model()
    if llm is None:
        return fallback
    try:
        prompt = (
            f"{SYSTEM}\n\nEvidence JSON (authoritative; do not add numbers that are not present):\n{evidence}\n\n"
            "Write a concise explanation for a supply-chain manager."
        )
        result = llm.invoke(prompt)
        text = getattr(result, "content", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        return fallback
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM explanation failed; using fallback: %s", exc)
        return fallback
