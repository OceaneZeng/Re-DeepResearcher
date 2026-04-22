# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=recipes/accelerate_configs/ddp.yaml src/open_r1/sft.py \
    --model_name_or_path open-r1/Qwen2.5-Math-7B-RoPE-300k \
    --dataset_name open-r1/Mixture-of-Thoughts \
    --dataset_config all \
    --eos_token '<|im_end|>' \
    --learning_rate 4.0e-5 \
    --num_train_epochs 5 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 2 \
    --gradient_checkpointing \
    --bf16 \
    --use_liger_kernel \
    --output_dir data/OpenR1-Distill-7B
"""

# Ensure UTF-8-related environment variables are set early on Windows so
# packages that read text files during import (e.g. trl reading Jinja templates)
# don't try to decode as the system ANSI code page (e.g., GBK) and fail.
import os
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("LANG", "en_US.UTF-8")

# If available, import Unsloth as early as possible so it can patch
# transformers/peft/trl for maximum performance.
try:  # pragma: no cover
    import unsloth  # type: ignore  # noqa: F401
except Exception:
    pass

import glob
import logging
import sys

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

import torch

from trl import ModelConfig, SFTTrainer, TrlParser, get_peft_config, setup_chat_format

from open_r1.configs import ScriptArguments, SFTConfig
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training

# Optional Unsloth integration (CUDA-only)
try:
    from open_r1.utils.unsloth_utils import UnslothConfig, load_unsloth_model_and_tokenizer
except Exception:  # pragma: no cover
    UnslothConfig = None  # type: ignore
    load_unsloth_model_and_tokenizer = None  # type: ignore

# Transformers <4.26.1 had a bug where `from transformers import *` would leak
# `transformers.file_utils` symbols into the global namespace; this caused
# `getattr(transformers, 'is_datasets_available', None)` to always return True
# even if the datasets library was not installed.
# See: https://github.com/huggingface/transformers/pull/22462
try:
    from transformers import __version__ as transfo_version

    if transfo_version.startswith("4.26.") or transfo_version.startswith("4.27."):
        import warnings

        warnings.warn(
            "Transformers version 4.26.x and 4.27.x have a bug that may cause runtime errors. "
            "Please upgrade to at least 4.28.0.",
            UserWarning,
        )
except Exception:
    pass

logger = logging.getLogger(__name__)


def sanity_checks(training_args):
    """Fail fast for common multi-process GPU misconfigurations."""
    try:
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        logger.info(f"Detected GPU count: {gpu_count}")
    except Exception as e:
        logger.warning(f"Could not query CUDA device count: {e}")
        gpu_count = 0

    # training_args.num_processes is set by accelerate; when running under accelerate the
    # visible world size is training_args.process_index or training_args.local_processes
    # but we check for training_args._n_processes if available; otherwise rely on environment.
    try:
        requested_procs = getattr(training_args, 'num_processes', None) or int(os.environ.get('WORLD_SIZE', '1'))
    except Exception:
        requested_procs = 1

    if requested_procs > max(1, gpu_count):
        raise RuntimeError(
            f"Accelerate requested {requested_procs} processes but only {gpu_count} GPU(s) detected.\n"
            "If you're training on a single GPU, use an accelerate config with num_processes=1 (e.g., recipes/accelerate_configs/ddp.yaml)."
        )


def _try_load_local_parquet_dataset(script_args: ScriptArguments) -> datasets.DatasetDict | None:
    """Optionally load a local parquet dataset from `dataset_name`.

    Supports:
    - dataset_name as a local `.parquet` file path
    - dataset_name as a local directory path containing parquet files (recursive)
    - dataset_name as a glob pattern (e.g. `data/**/*.parquet`)
    - dataset_name = "data": load all `data/**/*.parquet` under repo CWD
    - dataset_name = "local_parquet": load all `data/**/*.parquet` under repo CWD
    """
    dataset_name = getattr(script_args, "dataset_name", None)
    if not isinstance(dataset_name, str) or not dataset_name:
        return None

    def _load_parquet_files(files: list[str], *, hint: str) -> datasets.DatasetDict | None:
        files = [f for f in files if os.path.isfile(f) and f.lower().endswith(".parquet")]
        if not files:
            return None

        # HuggingFace datasets' parquet loader will try to cast every file into a single schema.
        # If users have multiple parquet datasets under `data/` with different schemas, this fails.
        # We proactively group files by column set. By default we pick the largest group to "do what I mean"
        # for the common case (e.g. `data/ShareGPT/*.parquet` plus unrelated `data/FineWeb/*.parquet`).
        # If `script_args.allow_mixed_parquet_schemas=true`, we instead load each schema-group and normalize
        # to a shared training format before concatenation.
        try:
            import pyarrow.parquet as pq  # local import to keep base imports light

            by_cols: dict[tuple[str, ...], list[str]] = {}
            for f in files:
                schema = pq.read_schema(f)
                cols = tuple(schema.names)
                by_cols.setdefault(cols, []).append(f)

            if len(by_cols) > 1:
                groups = sorted(by_cols.items(), key=lambda kv: len(kv[1]), reverse=True)
                if bool(getattr(script_args, "allow_mixed_parquet_schemas", False)):
                    logger.warning(
                        "Multiple parquet schemas detected under %s. "
                        "allow_mixed_parquet_schemas=true: will load %d schema-group(s) and normalize them before concatenation.",
                        hint,
                        len(groups),
                    )

                    def _to_text(example: dict) -> dict:
                        """Normalize heterogeneous parquet schemas into a single `text` column.

                        We prefer `text` over `messages` here because concatenating datasets with
                        mixed column layouts (some chat, some plain text) can fail feature alignment.
                        """
                        cols = set(example.keys())

                        # Chat-style already in `messages`
                        if "messages" in cols and isinstance(example.get("messages"), list):
                            parts: list[str] = []
                            for m in example["messages"]:
                                if not isinstance(m, dict):
                                    continue
                                role = str(m.get("role", "user"))
                                content = str(m.get("content", "") or "")
                                parts.append(f"{role}: {content}")
                            return {"text": "\n".join(parts)}

                        # ShareGPT-style conversations
                        if "conversations" in cols and isinstance(example.get("conversations"), list):
                            try:
                                converted = _sharegpt_conversations_to_messages(example)["messages"]
                                parts = [f"{m['role']}: {m['content']}" for m in converted if isinstance(m, dict)]
                                return {"text": "\n".join(parts)}
                            except Exception:
                                pass

                        # Alpaca / instruction tuning
                        if "instruction" in cols and "output" in cols:
                            instr = str(example.get("instruction", "") or "")
                            inp = str(example.get("input", "") or "")
                            user = instr if not inp else (instr + "\n" + inp)
                            out = str(example.get("output", "") or "")
                            return {"text": f"user: {user}\nassistant: {out}"}

                        # Prompt-response style
                        if "prompt" in cols and ("response" in cols or "completion" in cols):
                            resp_key = "response" if "response" in cols else "completion"
                            p = str(example.get("prompt", "") or "")
                            r = str(example.get(resp_key, "") or "")
                            return {"text": f"user: {p}\nassistant: {r}"}

                        # QA style
                        if "question" in cols and "answer" in cols:
                            q = str(example.get("question", "") or "")
                            a = str(example.get("answer", "") or "")
                            return {"text": f"user: {q}\nassistant: {a}"}

                        # Plain text
                        if "text" in cols:
                            return {"text": str(example.get("text", "") or "")}

                        # Fallback: pick the first string-ish column.
                        for k in example.keys():
                            v = example.get(k)
                            if isinstance(v, str) and v:
                                return {"text": v}
                        return {"text": str(example)}

                    loaded_trains: list[datasets.Dataset] = []
                    for cols, group_files in groups:
                        ds_dict = datasets.load_dataset("parquet", data_files={"train": sorted(group_files)})
                        ds = ds_dict["train"]
                        ds = ds.map(_to_text, remove_columns=list(ds.column_names))
                        # Ensure all groups have identical feature types for concatenation.
                        # Some parquet sources yield Arrow `large_string` which Datasets treats as incompatible with `string`.
                        ds = ds.cast(datasets.Features({"text": datasets.Value("string")}))
                        loaded_trains.append(ds)

                    logger.info(
                        "Loaded %d parquet schema-group(s) from %s; concatenating (%d total files).",
                        len(loaded_trains),
                        hint,
                        sum(len(v) for _, v in groups),
                    )
                    train = datasets.concatenate_datasets(loaded_trains)
                    return datasets.DatasetDict({"train": train})
                else:
                    chosen_cols, chosen_files = groups[0]
                    logger.warning(
                        "Multiple parquet schemas detected under %s. "
                        "Auto-selecting the largest schema-group (%d files, %d columns) and skipping %d other file(s). "
                        "If this is not desired, set `allow_mixed_parquet_schemas=true` or point dataset_name to a specific subdir/glob "
                        "(e.g. data/ShareGPT or data/ShareGPT/*.parquet).",
                        hint,
                        len(chosen_files),
                        len(chosen_cols),
                        sum(len(v) for _, v in groups[1:]),
                    )
                    files = sorted(chosen_files)
        except Exception as e:
            if bool(getattr(script_args, "allow_mixed_parquet_schemas", False)):
                # In mixed-schema mode, falling back to a single load will almost certainly crash
                # with a CastError. Fail fast with a clearer message.
                raise RuntimeError(
                    f"Failed to load mixed-schema parquet dataset from {hint}. "
                    "You have allow_mixed_parquet_schemas=true, but normalization/concatenation failed. "
                    f"Original error: {e}"
                ) from e
            logger.warning(f"Could not pre-scan parquet schemas ({hint}): {e}. Falling back to direct load.")

        logger.info(f"Loading local parquet dataset from {hint} ({len(files)} files)")
        return datasets.load_dataset("parquet", data_files={"train": sorted(files)})

    # Special alias to avoid escaping long paths in config files.
    if dataset_name in {"local_parquet", "data"}:
        root = os.path.join(os.getcwd(), "data")
        if not os.path.isdir(root):
            return None
        parquet_files = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
        return _load_parquet_files(parquet_files, hint=root)

    p = dataset_name
    # Glob support, e.g. `data/**/*.parquet`
    if any(ch in p for ch in ["*", "?", "["]):
        matches = sorted(glob.glob(p, recursive=True))
        return _load_parquet_files(matches, hint=p)

    if os.path.isfile(p) and p.lower().endswith(".parquet"):
        logger.info(f"Loading local parquet dataset from file: {p}")
        return datasets.load_dataset("parquet", data_files={"train": [p]})
    if os.path.isdir(p):
        parquet_files = sorted(glob.glob(os.path.join(p, "**", "*.parquet"), recursive=True))
        return _load_parquet_files(parquet_files, hint=p)
    return None


def _sharegpt_conversations_to_messages(example: dict) -> dict:
    """Convert ShareGPT-style `conversations` -> `messages`.

    Input:
      conversations: list[{"from": "...", "value": "...", ...}, ...]
    Output:
      messages: list[{"role": "user"|"assistant"|"system", "content": "..."}]
    """
    conv = example.get("conversations")
    if not isinstance(conv, list):
        raise ValueError("Expected `conversations` to be a list.")

    messages = []
    for t in conv:
        if not isinstance(t, dict):
            continue
        frm = str(t.get("from", "")).strip().lower()
        content = t.get("value")
        if content is None:
            content = ""
        content = str(content)
        if frm in {"human", "user"}:
            role = "user"
        elif frm in {"gpt", "assistant", "model"}:
            role = "assistant"
        elif frm in {"system"}:
            role = "system"
        else:
            # Unknown tag: default to user to keep a valid conversation.
            role = "user"
        messages.append({"role": role, "content": content})
    return {"messages": messages}


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    # Run sanity checks before heavy imports/initialization
    sanity_checks(training_args)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ######################################
    # Load dataset, tokenizer, and model #
    ######################################
    # Prefer local parquet if user points dataset_name to a path (or uses the alias).
    local_ds = _try_load_local_parquet_dataset(script_args)
    dataset = local_ds if local_ds is not None else get_dataset(script_args)

    # If this is ShareGPT-style parquet, convert `conversations` -> `messages` for TRL SFT.
    # TRL supports chat datasets with a `messages` column (list of {role, content} dicts).
    for split in list(dataset.keys()):
        cols = set(dataset[split].column_names)
        if "messages" not in cols and "conversations" in cols:
            logger.info(f"Detected ShareGPT `conversations` in split={split}; converting to `messages`.")
            dataset[split] = dataset[split].map(_sharegpt_conversations_to_messages)

    # If we have a chat dataset with `messages`, ensure TRL reads the right column.
    # TRL's SFTTrainer defaults to `dataset_text_field='text'`; for chat datasets it should be `messages`.
    try:
        train_split = script_args.dataset_train_split
        train_cols = set(dataset[train_split].column_names)
        if "messages" in train_cols and getattr(training_args, "dataset_text_field", "text") == "text":
            logger.info("Detected chat dataset (`messages`); setting training_args.dataset_text_field='messages'.")
            training_args.dataset_text_field = "messages"
    except Exception:
        pass

    # If user opts in, prefer Unsloth on CUDA.
    use_unsloth = bool(getattr(training_args, "use_unsloth", False))
    if use_unsloth and load_unsloth_model_and_tokenizer is not None:
        logger.info("use_unsloth=true: loading model/tokenizer via Unsloth (CUDA).")

        raw_torch_dtype = getattr(model_args, "torch_dtype", None)
        if raw_torch_dtype in [None, "auto"]:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = getattr(torch, raw_torch_dtype, torch.float16)

        ucfg = UnslothConfig(
            use_unsloth=True,
            load_in_4bit=bool(getattr(training_args, "unsloth_load_in_4bit", False)),
            max_seq_length=int(getattr(training_args, "max_seq_length", 4096)),
            local_files_only=bool(getattr(training_args, "unsloth_local_files_only", False)),
            lora_r=int(getattr(training_args, "unsloth_lora_r", 16)),
            lora_alpha=int(getattr(training_args, "unsloth_lora_alpha", 32)),
            lora_dropout=float(getattr(training_args, "unsloth_lora_dropout", 0.05)),
            lora_target_modules=getattr(training_args, "unsloth_lora_target_modules", None),
        )

        model, tokenizer = load_unsloth_model_and_tokenizer(
            model_name_or_path=model_args.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=bool(getattr(model_args, "trust_remote_code", True)),
            cfg=ucfg,
            cache_dir=getattr(training_args, "model_cache_dir", None),
        )

        # Apply chat template override if present
        if getattr(training_args, "chat_template", None) is not None:
            tokenizer.chat_template = training_args.chat_template
    else:
        tokenizer = get_tokenizer(model_args, training_args)
        model = get_model(model_args, training_args)

    if tokenizer.chat_template is None:
        logger.info("No chat template provided, defaulting to ChatML.")
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Normalize dataset selection from the config file: require one of the two supported forms.
    # This keeps the config file as the source of truth while tolerating parser/version quirks.
    if getattr(script_args, "dataset_name", None) is None and getattr(script_args, "dataset_mixture", None) is None:
        raise ValueError(
            "Either `dataset_name` or `dataset_mixture` must be provided in the config file. "
            "For local parquet-backed SFT, set dataset_name to `data` (or a parquet path / glob)."
        )

    # Provide conservative local defaults when running locally and arguments are omitted.
    # Default model to Qwen3.5-9B if not specified (user target).
    if getattr(model_args, "model_name_or_path", None) is None:
        model_args.model_name_or_path = "Qwen/Qwen3.5-9B"
        print(f"Defaulting model to {model_args.model_name_or_path}")

    # If CUDA is available and Unsloth is installed, default to using it unless explicitly disabled.
    if torch.cuda.is_available():
        try:
            if getattr(training_args, "use_unsloth", False) is False:
                training_args.use_unsloth = True
        except Exception:
            pass

    # If a local model folder exists in ./models/<name>, prefer it for offline runs.
    try:
        model_name = model_args.model_name_or_path
        model_basename = os.path.basename(model_name)
        local_model_paths = [
            os.path.join(os.getcwd(), "models", model_basename),
            os.path.join(os.getcwd(), model_basename),
            os.path.join(os.getcwd(), "models", model_name.replace("/", "_")),
        ]
        for p in local_model_paths:
            if os.path.isdir(p):
                print(f"Found local model directory, using local model at: {p}")
                model_args.model_name_or_path = p
                break
    except Exception:
        # non-fatal; fall back to provided model_name_or_path
        pass

    main(script_args, training_args, model_args)