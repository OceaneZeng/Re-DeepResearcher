from __future__ import annotations

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple Python math expression. Example: '2*(3+4)'."""
    allowed = {"__builtins__": {}}
    try:
        result = eval(expression, allowed, {})
        return str(result)
    except Exception as e:
        return f"error: {e}"

