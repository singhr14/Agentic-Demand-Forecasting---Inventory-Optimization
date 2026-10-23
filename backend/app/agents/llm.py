from __future__ import annotations

import logging

from backend.app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


def build_chat_model(settings: Settings | None = None):
    settings = settings or get_settings()
    if not settings.llm_configured:
        return None
    provider = settings.LLM_PROVIDER.lower().strip()
    if provider not in {"openrouter", "openai"}:
        logger.warning("Unsupported LLM_PROVIDER=%s; using deterministic explanations", provider)
        return None
    from langchain_openai import ChatOpenAI

    base_url = settings.OPENROUTER_BASE_URL if provider == "openrouter" else None
    headers = {}
    if provider == "openrouter":
        headers = {
            "HTTP-Referer": settings.OPENROUTER_SITE_URL,
            "X-Title": settings.OPENROUTER_APP_NAME,
        }
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=base_url,
        temperature=0.1,
        default_headers=headers or None,
        timeout=30,
        max_retries=1,
    )
