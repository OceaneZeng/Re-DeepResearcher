import argparse
import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset


THINK_ANSWER_PATTERN = re.compile(
    r"^<think>\s*\n.*?\n\s*</think>\s*\n<answer>\s*\n.*?\n\s*</answer>\s*$",
    re.DOTALL | re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity-check unified parquet for SFT + GSPO readiness."
    )
    parser.add_argument(
        "--parquet-path",
        type=str,
        required=True,
        help="Path to parquet file or glob pattern.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="data/subsets/sft/parquet_sanity_report.json",
        help="Path to write JSON sanity report.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If >0, only inspect the first N rows (fast check).",
    )
    parser.add_argument(
        "--warn-empty-threshold",
        type=float,
        default=0.01,
        help="Warn if any key field empty-rate exceeds this ratio.",
    )
    parser.add_argument(
        "--warn-format-threshold",
        type=float,
        default=0.95,
        help="Warn if <think>/<answer> valid ratio is below this threshold.",
    )
    return parser.parse_args()


def _is_blank(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return len(v.strip()) == 0
    if isinstance(v, list):
        return len(v) == 0
    return False


def _safe_len(v: Any) -> int:
    if isinstance(v, str):
        return len(v)
    return 0


def _messages_valid(messages: Any) -> bool:
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    has_user = False
    has_assistant = False
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        content = item.get("content", "")
        if role == "user" and isinstance(content, str) and content.strip():
            has_user = True
        if role == "assistant" and isinstance(content, str) and content.strip():
            has_assistant = True
    return has_user and has_assistant


def main() -> None:
    args = parse_args()

    ds = load_dataset("parquet", data_files={"train": args.parquet_path})["train"]
    total_rows = len(ds)
    if args.max_samples > 0:
        inspect_rows = min(total_rows, args.max_samples)
        ds = ds.select(range(inspect_rows))
    else:
        inspect_rows = total_rows

    required_columns = [
        "id",
        "source",
        "domain",
        "language",
        "prompt",
        "solution",
        "assistant",
        "messages",
        "verifiable",
    ]

    missing_columns = [c for c in required_columns if c not in ds.column_names]

    empty_counts = {k: 0 for k in ["id", "prompt", "solution", "assistant", "messages"]}
    format_valid_count = 0
    messages_valid_count = 0
    verifiable_true_count = 0
    source_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    prompt_len_sum = 0
    solution_len_sum = 0
    assistant_len_sum = 0

    for row in ds:
        source = str(row.get("source", "") or "")
        domain = str(row.get("domain", "") or "")
        source_counts[source] = source_counts.get(source, 0) + 1
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        if bool(row.get("verifiable", False)):
            verifiable_true_count += 1

        prompt = row.get("prompt", "")
        solution = row.get("solution", "")
        assistant = row.get("assistant", "")
        messages = row.get("messages", [])
        sample_id = row.get("id", "")

        prompt_len_sum += _safe_len(prompt)
        solution_len_sum += _safe_len(solution)
        assistant_len_sum += _safe_len(assistant)

        if _is_blank(sample_id):
            empty_counts["id"] += 1
        if _is_blank(prompt):
            empty_counts["prompt"] += 1
        if _is_blank(solution):
            empty_counts["solution"] += 1
        if _is_blank(assistant):
            empty_counts["assistant"] += 1
        if _is_blank(messages):
            empty_counts["messages"] += 1

        if isinstance(assistant, str) and THINK_ANSWER_PATTERN.match(assistant):
            format_valid_count += 1
        if _messages_valid(messages):
            messages_valid_count += 1

    def ratio(x: int) -> float:
        if inspect_rows == 0:
            return 0.0
        return x / inspect_rows

    empty_ratios = {k: ratio(v) for k, v in empty_counts.items()}

    report = {
        "input": {
            "parquet_path": args.parquet_path,
            "total_rows": total_rows,
            "inspected_rows": inspect_rows,
            "columns": ds.column_names,
        },
        "schema": {
            "required_columns": required_columns,
            "missing_columns": missing_columns,
            "schema_ok": len(missing_columns) == 0,
        },
        "quality": {
            "empty_counts": empty_counts,
            "empty_ratios": empty_ratios,
            "think_answer_format_valid_count": format_valid_count,
            "think_answer_format_valid_ratio": ratio(format_valid_count),
            "messages_valid_count": messages_valid_count,
            "messages_valid_ratio": ratio(messages_valid_count),
            "verifiable_true_count": verifiable_true_count,
            "verifiable_true_ratio": ratio(verifiable_true_count),
            "avg_prompt_length": (prompt_len_sum / inspect_rows) if inspect_rows else 0.0,
            "avg_solution_length": (solution_len_sum / inspect_rows) if inspect_rows else 0.0,
            "avg_assistant_length": (assistant_len_sum / inspect_rows) if inspect_rows else 0.0,
        },
        "distribution": {
            "by_source": source_counts,
            "by_domain": domain_counts,
        },
        "warnings": [],
    }

    warnings = report["warnings"]
    if missing_columns:
        warnings.append(f"Missing required columns: {missing_columns}")
    for field_name, r in empty_ratios.items():
        if r > args.warn_empty_threshold:
            warnings.append(
                f"Empty ratio too high for '{field_name}': {r:.4f} > {args.warn_empty_threshold:.4f}"
            )
    if report["quality"]["think_answer_format_valid_ratio"] < args.warn_format_threshold:
        warnings.append(
            "Low <think>/<answer> format valid ratio: "
            f"{report['quality']['think_answer_format_valid_ratio']:.4f} < {args.warn_format_threshold:.4f}"
        )
    if report["quality"]["messages_valid_ratio"] < 0.99:
        warnings.append(
            f"messages_valid_ratio below 0.99: {report['quality']['messages_valid_ratio']:.4f}"
        )
    if report["quality"]["verifiable_true_ratio"] < 0.5:
        warnings.append(
            f"verifiable_true_ratio below 0.5: {report['quality']['verifiable_true_ratio']:.4f}"
        )

    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] sanity report: {output_path}")
    print(f"[summary] inspected_rows={inspect_rows}, warnings={len(warnings)}")
    if warnings:
        for w in warnings:
            print(f"[warn] {w}")


if __name__ == "__main__":
    main()
