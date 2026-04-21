from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HFResolvedModel:
    original: str
    local_dir: str


def _is_rank0_process() -> bool:
    # Covers torchrun/accelerate/slurm best-effort.
    for k in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        v = os.environ.get(k)
        if v is None:
            continue
        try:
            return int(v) == 0
        except Exception:
            return True
    return True


def _looks_like_ready_model_dir(dir_path: str) -> bool:
    if not os.path.isdir(dir_path):
        return False
    # Transformers-compatible "ready" markers.
    markers = (
        "config.json",
        "tokenizer.json",
        "tokenizer.model",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    return any(os.path.exists(os.path.join(dir_path, m)) for m in markers)


def _wait_for_model_dir(dir_path: str, *, timeout_s: int = 60 * 60) -> None:
    start = time.time()
    while True:
        if _looks_like_ready_model_dir(dir_path):
            return
        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for model files to appear in: {dir_path}. "
                "If running multi-process, ensure rank0 can download the model."
            )
        time.sleep(2.0)


def resolve_hf_model_to_local_dir(
    model_name_or_path: str,
    *,
    local_dir: Optional[str],
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> HFResolvedModel:
    """Optionally download a HF Hub model into a user-chosen local directory.

    If `local_dir` is provided and `model_name_or_path` is a HF repo id (not an existing directory),
    we download the snapshot into `local_dir` and return the local path.

    This is useful when you want full control over *where* HF files are stored (beyond cache_dir),
    and it works for both Transformers and Unsloth because they can load from a local folder.
    """
    if not isinstance(model_name_or_path, str) or not model_name_or_path.strip():
        raise ValueError("model_name_or_path must be a non-empty string.")
    model_name_or_path = model_name_or_path.strip()

    if not local_dir:
        return HFResolvedModel(original=model_name_or_path, local_dir=model_name_or_path)

    # If user already passed a local folder, nothing to do.
    if os.path.isdir(model_name_or_path):
        return HFResolvedModel(original=model_name_or_path, local_dir=model_name_or_path)

    local_dir = os.path.abspath(os.path.expanduser(str(local_dir)))
    os.makedirs(local_dir, exist_ok=True)

    # Rank0 downloads; others wait.
    if _is_rank0_process():
        from huggingface_hub import snapshot_download

        # Always run snapshot_download with resume enabled when local_dir is set.
        # A partial/incomplete snapshot (e.g. missing sharded safetensors files)
        # can look "ready" just because `config.json` exists; this would later
        # crash when loading weights. snapshot_download is cache-aware and will
        # quickly no-op when everything is already present.
        kwargs = dict(
            local_dir=local_dir,
            resume_download=True,
            local_dir_use_symlinks=False,
        )
        if revision and str(revision).strip():
            kwargs["revision"] = revision
        if cache_dir and str(cache_dir).strip():
            kwargs["cache_dir"] = cache_dir

        snapshot_download(repo_id=model_name_or_path, **kwargs)
    else:
        _wait_for_model_dir(local_dir)

    return HFResolvedModel(original=model_name_or_path, local_dir=local_dir)
