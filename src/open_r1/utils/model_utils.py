import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from ..configs import GRPOConfig, SFTConfig


def get_tokenizer(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def get_model(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> AutoModelForCausalLM:
    """Get the model"""
    raw_torch_dtype = getattr(model_args, "torch_dtype", None)
    if raw_torch_dtype in [None, "auto"]:
        torch_dtype = torch.float16
    else:
        try:
            torch_dtype = getattr(torch, raw_torch_dtype)
        except Exception:
            torch_dtype = torch.float16

    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    attn_impl = getattr(model_args, "attn_implementation", None)
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    try:
        return AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    except Exception:
        model_kwargs.pop("attn_implementation", None)
        model_kwargs["torch_dtype"] = torch.float16
        return AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)