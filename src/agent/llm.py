from __future__ import annotations

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Create the chat model used by the agent.

    Env:
      - OPENAI_API_KEY: required
      - OPENAI_MODEL: optional (default: gpt-4o-mini)
      - OPENAI_TEMPERATURE: optional (default: 0)
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.environ.get("OPENAI_TEMPERATURE", "0"))
    return ChatOpenAI(model=model, temperature=temperature)

