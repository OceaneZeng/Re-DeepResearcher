import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import AutoPeftModelForCausalLM
except Exception:
    AutoPeftModelForCausalLM = None

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


FORMAT_PATTERN = re.compile(
    r"^<think>\s*\n.*?\n\s*</think>\s*\n<answer>\s*\n(.*?)\n\s*</answer>\s*$",
    re.DOTALL | re.MULTILINE,
)

REFUSAL_MARKERS = (
    "i can't",
    "i cannot",
    "i'm unable",
    "as an ai",
    "无法",
    "不能",
    "抱歉",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model format-following and reasoning-style behavior.")
    parser.add_argument("--model-path", type=str, required=True, help="Model or adapter directory path.")
    parser.add_argument(
        "--base-model-path",
        type=str,
        default=None,
        help="Base model path/id when --model-path is a LoRA adapter-only directory.",
    )
    parser.add_argument("--dataset-path", type=str, required=True, help="Validation parquet file path.")
    parser.add_argument("--output-report", type=str, default="data/eval/gspo_format_report.json")
    parser.add_argument("--output-failures", type=str, default="data/eval/gspo_format_failures.jsonl")
    parser.add_argument("--prompt-column", type=str, default="question")
    parser.add_argument("--options-column", type=str, default="options")
    parser.add_argument("--answer-column", type=str, default="answer")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str, base_model_path: str | None = None):
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    if AutoPeftModelForCausalLM is not None:
        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        except Exception:
            model = None

    if model is None:
        if base_model_path is not None and PeftModel is not None:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )

    model.eval()
    return model, tokenizer


def is_adapter_dir(model_path: str) -> bool:
    return (Path(model_path) / "adapter_config.json").exists()


def build_user_prompt(row: pd.Series, prompt_column: str, options_column: str) -> str:
    question = str(row.get(prompt_column, "")).strip()
    options = row.get(options_column, None)
    if options is None or (isinstance(options, float) and pd.isna(options)):
        return question
    return f"{question}\n\nOptions:\n{options}"


def render_prompt(tokenizer, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_prompt


def extract_answer_text(text: str) -> tuple[bool, str]:
    match = FORMAT_PATTERN.match(text.strip())
    if match is None:
        return False, text.strip()
    return True, match.group(1).strip()


def analyze_output(text: str) -> dict[str, Any]:
    strict_ok, answer_text = extract_answer_text(text)
    lowered = answer_text.lower()
    has_refusal = any(marker in lowered for marker in REFUSAL_MARKERS)

    tag_score = 0
    if text.count("<think>") == 1:
        tag_score += 1
    if text.count("</think>") == 1:
        tag_score += 1
    if text.count("<answer>") == 1:
        tag_score += 1
    if text.count("</answer>") == 1:
        tag_score += 1

    return {
        "strict_format_ok": strict_ok,
        "tag_complete_ok": tag_score == 4,
        "tag_score_0_4": tag_score,
        "answer_length_chars": len(answer_text),
        "empty_answer": len(answer_text.strip()) == 0,
        "refusal_like": has_refusal,
    }


def batch_generate(model, tokenizer, prompts: list[str], max_new_tokens: int, temperature: float, top_p: float) -> list[str]:
    encoded = tokenizer(prompts, padding=True, return_tensors="pt")
    if torch.cuda.is_available():
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generations = []
    input_lens = encoded["attention_mask"].sum(dim=1).tolist()
    for out, in_len in zip(outputs, input_lens):
        gen_tokens = out[int(in_len) :]
        generations.append(tokenizer.decode(gen_tokens, skip_special_tokens=True))
    return generations


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if is_adapter_dir(args.model_path) and args.base_model_path is None:
        raise ValueError(
            "--model-path points to a LoRA adapter directory. "
            "Please also provide --base-model-path (e.g. Qwen/Qwen3.5-4B or your local base model path)."
        )

    dataset = pd.read_parquet(args.dataset_path)
    if args.prompt_column not in dataset.columns:
        raise ValueError(f"Column '{args.prompt_column}' not found in dataset.")

    dataset = dataset.head(args.max_samples).copy()
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model_path)

    prompts = [
        render_prompt(tokenizer, build_user_prompt(row, args.prompt_column, args.options_column))
        for _, row in dataset.iterrows()
    ]

    rows = []
    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i : i + args.batch_size]
        batch_rows = dataset.iloc[i : i + args.batch_size]
        outputs = batch_generate(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        for (_, row), output_text in zip(batch_rows.iterrows(), outputs):
            metrics = analyze_output(output_text)
            rows.append(
                {
                    "prompt": str(row.get(args.prompt_column, "")),
                    "options": row.get(args.options_column, None),
                    "gold_answer": row.get(args.answer_column, None),
                    "output": output_text,
                    **metrics,
                }
            )

    df = pd.DataFrame(rows)
    total = len(df)
    report = {
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "num_samples": total,
        "strict_format_rate": float(df["strict_format_ok"].mean()) if total else 0.0,
        "tag_complete_rate": float(df["tag_complete_ok"].mean()) if total else 0.0,
        "empty_answer_rate": float(df["empty_answer"].mean()) if total else 0.0,
        "refusal_rate": float(df["refusal_like"].mean()) if total else 0.0,
        "avg_answer_length_chars": float(df["answer_length_chars"].mean()) if total else 0.0,
        "median_answer_length_chars": float(df["answer_length_chars"].median()) if total else 0.0,
    }

    failure_mask = (~df["strict_format_ok"]) | (~df["tag_complete_ok"]) | (df["empty_answer"]) | (df["refusal_like"])
    failures = df[failure_mask].copy()

    report_path = Path(args.output_report)
    failures_path = Path(args.output_failures)
    ensure_parent(report_path)
    ensure_parent(failures_path)

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    failures.to_json(failures_path, orient="records", force_ascii=False, lines=True)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved report to: {report_path}")
    print(f"Saved failures to: {failures_path}")


if __name__ == "__main__":
    main()
