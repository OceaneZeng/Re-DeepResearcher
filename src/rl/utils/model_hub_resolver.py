from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ResolvedModelRef:
    original: str
    resolved_path: str
    source: str  # "hf" | "modelscope" | "local"


def _is_rank0_process() -> bool:
    """Best-effort rank0 detection across accelerate/torchrun."""
    for k in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        v = os.environ.get(k)
        if v is None:
            continue
        try:
            return int(v) == 0
        except Exception:
            return True
    return True


def _wait_for_local_model(dir_path: str, *, timeout_s: int = 60 * 60) -> None:
    """Wait until a local model directory is ready (rank>0)."""
    start = time.time()
    # Heuristic "ready" markers for Transformers-compatible folders.
    ready_markers = ("config.json", "model.safetensors.index.json", "pytorch_model.bin.index.json")
    while True:
        if os.path.isdir(dir_path) and any(os.path.exists(os.path.join(dir_path, m)) for m in ready_markers):
            return
        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for model download to finish at: {dir_path}. "
                "If you're using multi-process training, ensure rank0 can download the model."
            )
        time.sleep(2.0)


def _model_id_to_local_dir(model_id: str, *, base_dir: str) -> str:
    safe = model_id.replace("/", "__").replace("\\", "__").replace(":", "__")
    return os.path.join(base_dir, safe)


def resolve_model_name_or_path(
    model_name_or_path: str,
    *,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    source: Optional[str] = None,
) -> ResolvedModelRef:
    """Resolve a model reference to a local path when needed.

    Supported forms:
    - Local path (directory): returned as-is (source="local")
    - Hugging Face id/path: returned as-is by default (source="hf")
    - ModelScope id with prefix: "modelscope:<model_id>" (downloaded to ./models/<id>)
    - Force ModelScope via env var: OPEN_R1_MODEL_SOURCE=modelscope
    """
    if not isinstance(model_name_or_path, str) or not model_name_or_path.strip():
        raise ValueError("model_name_or_path must be a non-empty string.")

    original = model_name_or_path
    model_name_or_path = model_name_or_path.strip()

    # If user already points to a local folder, do nothing.
    if os.path.isdir(model_name_or_path):
        return ResolvedModelRef(original=original, resolved_path=model_name_or_path, source="local")

    env_source = os.environ.get("OPEN_R1_MODEL_SOURCE")
    effective_source = (source or env_source or "hf").strip().lower()

    ms_prefix = "modelscope:"
    ms_requested = model_name_or_path.lower().startswith(ms_prefix) or effective_source == "modelscope"
    if not ms_requested:
        return ResolvedModelRef(original=original, resolved_path=model_name_or_path, source="hf")

    model_id = model_name_or_path[len(ms_prefix) :].strip() if model_name_or_path.lower().startswith(ms_prefix) else model_name_or_path
    if not model_id:
        raise ValueError("ModelScope model id is empty. Use e.g. `modelscope:Qwen/Qwen3.5-4B`.")

    repo_root = os.getcwd()
    models_dir = models_dir or os.path.join(repo_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    local_dir = _model_id_to_local_dir(model_id, base_dir=models_dir)

    if _is_rank0_process():
        try:
            # ModelScope API: https://github.com/modelscope/modelscope
            from modelscope.hub.snapshot_download import snapshot_download  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "ModelScope support requested but `modelscope` is not installed. "
                "Install it with: `pip install modelscope`."
            ) from e

        kwargs = {}
        if revision and str(revision).strip():
            kwargs["revision"] = revision
        if cache_dir and str(cache_dir).strip():
            # Some versions use `cache_dir`, some only use a global cache.
            kwargs["cache_dir"] = cache_dir

        # We use `local_dir` to make downstream `from_pretrained(local_dir)` deterministic.
        try:
            snapshot_download(model_id, local_dir=local_dir, **kwargs)
        except TypeError:
            # Backward/forward compat for modelscope API changes.
            snapshot_download(model_id, local_dir=local_dir)
    else:
        _wait_for_local_model(local_dir)

    return ResolvedModelRef(original=original, resolved_path=local_dir, source="modelscope")
