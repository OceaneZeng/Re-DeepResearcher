from __future__ import annotations

import os
from typing import Optional

from langchain_core.tools import tool


def _repo_root() -> str:
    # Use CWD as the repo root (works when running from repo root).
    return os.path.abspath(os.getcwd())


def _safe_path(path: str, *, root: Optional[str] = None) -> str:
    root = os.path.abspath(root or _repo_root())
    p = os.path.abspath(os.path.join(root, path))
    if os.path.commonpath([root, p]) != root:
        raise ValueError(f"Path escapes repo root: {path!r}")
    return p


@tool
def list_dir(path: str = ".") -> str:
    """List files in a directory under the repo root."""
    try:
        p = _safe_path(path)
        if not os.path.isdir(p):
            return f"error: not a directory: {path}"
        entries = sorted(os.listdir(p))
        return "\n".join(entries[:200])
    except Exception as e:
        return f"error: {e}"


@tool
def read_text_file(path: str, max_chars: int = 4000) -> str:
    """Read a UTF-8 text file under the repo root (truncated)."""
    try:
        p = _safe_path(path)
        if not os.path.isfile(p):
            return f"error: not a file: {path}"
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            s = f.read(max(0, int(max_chars)))
        return s
    except Exception as e:
        return f"error: {e}"

