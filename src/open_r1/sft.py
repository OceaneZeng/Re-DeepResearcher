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
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
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

import logging
import os as _os
import sys

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

# New imports for runtime checks
import torch

from open_r1.configs import ScriptArguments, SFTConfig
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training

from trl import ModelConfig, SFTTrainer, TrlParser, get_peft_config
try:
    from trl import setup_chat_format
except Exception:
    import warnings

    def setup_chat_format(model, tokenizer, format: str = "chatml"):
        warnings.warn(
            "trl.setup_chat_format not found in installed trl version; proceeding without modifying model/tokenizer chat format."
        )
        # Ensure tokenizer has a chat_template attribute to avoid attribute errors later
        if not hasattr(tokenizer, "chat_template"):
            setattr(tokenizer, "chat_template", None)
        return model, tokenizer


logger = logging.getLogger(__name__)


def sanity_checks(training_args):
    """Perform runtime checks to avoid common multi-process / ROCm pitfalls.

    - Ensure number of accelerate processes does not exceed available GPUs.
    - Detect whether Torch is a ROCm build and warn about CUDA-only libs like bitsandbytes.
    """
    # Detect available devices
    try:
        gpu_count = 0
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
        else:
            # On ROCm builds torch.cuda.is_available() should still be True; fallback check for HIP
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                # Some ROCm builds still expose CUDA API compat; attempt device_count
                try:
                    gpu_count = torch.cuda.device_count()
                except Exception:
                    gpu_count = 0
        logger.info(f"Detected GPU count: {gpu_count}")
    except Exception as e:
        logger.warning(f"Could not query CUDA/ROCm device count: {e}")
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
            "On single-GPU ROCm DTK please use an accelerate config with num_processes=1 (e.g., recipes/accelerate_configs/ddp.yaml).")

    # Warn if bitsandbytes requested on ROCm or when Torch build is ROCm
    try:
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        if is_rocm:
            logger.info(f"Detected ROCm-enabled PyTorch: {torch.__version__} (hip={torch.version.hip})")
            # bitsandbytes is CUDA-only in many builds
            try:
                import bitsandbytes as bnb  # type: ignore
                logger.warning("bitsandbytes appears installed — it requires CUDA; on ROCm this may not work.")
            except Exception:
                logger.info("bitsandbytes not installed (recommended for ROCm). If you need 8-bit, use a CPU fallback or CUDA node.")
    except Exception:
        pass


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
    dataset = get_dataset(script_args)
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
            "For local parquet-backed SFT, set dataset_name: open-r1/Mixture-of-Thoughts."
        )

    # Provide conservative local defaults when running locally and arguments are omitted.
    # Default model to a Qwen 7B instruct family if not specified.
    if getattr(model_args, "model_name_or_path", None) is None:
        model_args.model_name_or_path = "open-r1/Qwen2.5-7B-Instruct"
        print(f"Defaulting model to {model_args.model_name_or_path}")

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
