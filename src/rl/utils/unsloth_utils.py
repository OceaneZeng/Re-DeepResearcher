"""Unsloth integration helpers.

This repo primarily uses TRL + Transformers. Unsloth can speed up SFT on CUDA
by providing optimized model loading + fast LoRA.

Contract:
- Only activate on CUDA.
- Only activate when user explicitly opts in.
- Keep the rest of the training script unchanged.

We keep the implementation defensive because:
- Unsloth APIs evolve.
- Some environments may have partial installs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import os
import torch
from transformers import AutoConfig


@dataclass
class UnslothConfig:
    use_unsloth: bool = False
    load_in_4bit: bool = False
    max_seq_length: Optional[int] = None
    local_files_only: bool = False

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[list[str]] = None


def is_cuda() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


def _default_qwen_lora_targets() -> list[str]:
    # Reasonable defaults for Qwen-family. Users can override via config.
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def load_unsloth_model_and_tokenizer(
    *,
    model_name_or_path: str,
    torch_dtype: Any,
    trust_remote_code: bool,
    cfg: UnslothConfig,
    cache_dir: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load model/tokenizer via Unsloth FastLanguageModel.

    Returns:
        (model, tokenizer)

    Raises:
        RuntimeError if unsloth is unavailable or CUDA isn't available.
    """

    if not is_cuda():
        raise RuntimeError("Unsloth requested but CUDA is not available.")

    try:
        from unsloth import FastLanguageModel  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Unsloth is not installed or failed to import. Install it in your CUDA env, e.g. `pip install unsloth`."
        ) from e

    max_seq_length = cfg.max_seq_length
    if max_seq_length is None:
        # fall back to a safe-ish default; caller should usually pass training_args.max_seq_length
        max_seq_length = 4096

    # Respect common offline flags. If set, never attempt network access.
    env_offline = os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE")
    local_files_only = bool(cfg.local_files_only) or (str(env_offline).strip() in {"1", "true", "True", "YES", "yes"})

    if local_files_only and os.path.isdir(model_name_or_path):
        cfg_path = os.path.join(model_name_or_path, "config.json")
        if not os.path.isfile(cfg_path):
            raise RuntimeError(
                f"Unsloth local load requested, but config.json is missing under: {model_name_or_path}"
            )

    # Pre-validate model config so unsupported/corrupt snapshots fail with
    # actionable information before entering Unsloth internals.
    try:
        cfg_obj = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=bool(trust_remote_code),
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        arch = getattr(cfg_obj, "architectures", None)
        print(
            f"[unsloth] resolved model={model_name_or_path!r} "
            f"model_type={getattr(cfg_obj, 'model_type', None)!r} architectures={arch!r}"
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to load model config before Unsloth init. "
            f"model={model_name_or_path!r} local_files_only={local_files_only} cache_dir={cache_dir!r}. "
            f"Original error: {e}"
        ) from e

    base_kwargs = dict(
        max_seq_length=max_seq_length,
        dtype=torch_dtype,
        load_in_4bit=bool(cfg.load_in_4bit),
        trust_remote_code=bool(trust_remote_code),
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )

    # Compatibility: different Unsloth versions may expect either `model_name`
    # or `model_name_or_path`. Try both deterministically.
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path,
            **base_kwargs,
        )
    except TypeError:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name_or_path=model_name_or_path,
            **base_kwargs,
        )

    target_modules = cfg.lora_target_modules or _default_qwen_lora_targets()

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(cfg.lora_r),
        target_modules=target_modules,
        lora_alpha=int(cfg.lora_alpha),
        lora_dropout=float(cfg.lora_dropout),
        bias="none",
        use_gradient_checkpointing=True,  # matches common SFT usage
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer