import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset


DEFAULT_DATASETS = {
    # Strong verifiable math signal for SFT cold-start + later GSPO rewards.
    "numina": {
        "hf_id": "AI-MO/NuminaMath-CoT",
        "config": None,
        "split": "train",
        "target_count": 60000,
        "source": "AI-MO/NuminaMath-CoT",
        "domain": "math",
        "verifiable": True,
    },
    # Code reasoning + algorithmic explanation.
    "codeforces": {
        "hf_id": "open-r1/codeforces-cots",
        "config": "solutions_decontaminated",
        "split": "train",
        "target_count": 20000,
        "source": "open-r1/codeforces-cots",
        "domain": "code",
        "verifiable": True,
    },
    # Multi-hop QA for logic and evidence integration.
    "hotpotqa": {
        "hf_id": "hotpotqa/hotpot_qa",
        "config": "fullwiki",
        "split": "train",
        "target_count": 12000,
        "source": "hotpotqa/hotpot_qa",
        "domain": "multi_hop_qa",
        "verifiable": False,
    },
    # Structured science QA with options; improves short-form answer discipline.
    "openbookqa": {
        "hf_id": "allenai/openbookqa",
        "config": "additional",
        "split": "train",
        "target_count": 8000,
        "source": "allenai/openbookqa",
        "domain": "science_qa",
        "verifiable": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a unified parquet for SFT and later GSPO training."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data",
        help="Hugging Face datasets cache dir (project-local cache recommended).",
    )
    parser.add_argument(
        "--output-parquet",
        type=str,
        default="data/subsets/sft/gspo_sft_unified_v2.parquet",
        help="Output parquet path.",
    )
    parser.add_argument(
        "--minimal-schema",
        action="store_true",
        help="Keep only minimal columns (id/source/messages). By default, keep CARL-MoA-friendly metadata.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic shuffling/subsampling.",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=100000,
        help="Total target number of samples after mixing.",
    )
    parser.add_argument(
        "--numina-ratio",
        type=float,
        default=0.60,
        help="Mixture ratio for NuminaMath-CoT.",
    )
    parser.add_argument(
        "--codeforces-ratio",
        type=float,
        default=0.20,
        help="Mixture ratio for codeforces-cots.",
    )
    parser.add_argument(
        "--hotpotqa-ratio",
        type=float,
        default=0.12,
        help="Mixture ratio for HotpotQA.",
    )
    parser.add_argument(
        "--openbookqa-ratio",
        type=float,
        default=0.08,
        help="Mixture ratio for OpenBookQA.",
    )
    return parser.parse_args()


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _build_id(source: str, prompt: str, solution: str) -> str:
    h = hashlib.sha1(f"{source}\n{prompt}\n{solution}".encode("utf-8")).hexdigest()
    return f"{source.split('/')[-1]}-{h[:16]}"


def _carl_moa_route_hint(domain: str, verifiable: bool) -> str:
    domain = _safe_str(domain).lower()
    if domain in {"math", "code"} and verifiable:
        return "router->solver->tool->critic->refiner"
    if "qa" in domain:
        return "router->solver->critic->refiner"
    return "router->solver->refiner"


def _make_messages(prompt: str, assistant_text: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_text},
    ]


def _assistant_with_contract(reasoning: str, final_answer: str) -> str:
    reasoning = _safe_str(reasoning)
    final_answer = _safe_str(final_answer)
    if not reasoning:
        reasoning = "I solve this step by step and focus on correctness."
    if not final_answer:
        final_answer = "N/A"
    return f"<think>\n{reasoning}\n</think>\n<answer>\n{final_answer}\n</answer>"


def _extract_final_answer_from_text(text: str) -> str:
    """Best-effort final answer extraction from reasoning-rich solution text."""
    text = _safe_str(text)
    if not text:
        return ""

    # Prefer LaTeX boxed final answer if present (common in math datasets).
    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed_matches:
        return _safe_str(boxed_matches[-1])

    # Fall back to the last non-empty line, lightly cleaned.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""

    candidate = lines[-1]
    candidate = re.sub(r"^\s*(therefore|thus|so|hence)[,:]?\s*", "", candidate, flags=re.IGNORECASE)
    return _safe_str(candidate)


def normalize_numina_row(prompt: str, raw_solution: str, cot: str) -> tuple[str, str]:
    """Split Numina sample into (reasoning, final_answer).

    Heuristics:
    1) If explicit CoT exists, use it as reasoning.
    2) Final answer prefers last \\boxed{...}.
    3) If no boxed answer, use last concise equation/line.
    """
    prompt = _safe_str(prompt)
    raw_solution = _safe_str(raw_solution)
    cot = _safe_str(cot)

    reasoning = cot if cot else raw_solution
    final_answer = _extract_final_answer_from_text(raw_solution)

    if not final_answer:
        final_answer = _extract_final_answer_from_text(reasoning)

    if not final_answer:
        final_answer = raw_solution

    # If reasoning ends with the exact final answer sentence, trim light duplication.
    if reasoning and final_answer:
        tail_patterns = [
            rf"(Thus|Therefore|Hence|So).{{0,30}}{re.escape(final_answer)}\.?\s*$",
            rf"{re.escape(final_answer)}\s*$",
        ]
        for pat in tail_patterns:
            reasoning = re.sub(pat, "", reasoning, flags=re.IGNORECASE | re.DOTALL).strip()

    if not reasoning:
        reasoning = raw_solution

    return reasoning, final_answer


def _extract_numina(row: dict[str, Any], source: str, domain: str, verifiable: bool) -> dict[str, Any] | None:
    prompt = _safe_str(row.get("problem") or row.get("question") or row.get("prompt"))
    raw_solution = _safe_str(row.get("solution") or row.get("answer") or row.get("final_answer"))
    cot = _safe_str(row.get("cot") or row.get("cot_content") or row.get("reasoning") or row.get("explanation"))
    if not prompt or not raw_solution:
        return None

    reasoning, final_answer = normalize_numina_row(
        prompt=prompt,
        raw_solution=raw_solution,
        cot=cot,
    )

    assistant = _assistant_with_contract(reasoning, final_answer)
    return {
        "id": _build_id(source, prompt, final_answer),
        "source": source,
        "domain": domain,
        "language": "en",
        "prompt": prompt,
        "solution": final_answer,
        "assistant": assistant,
        "messages": _make_messages(prompt, assistant),
        "verifiable": verifiable,
        "difficulty": _safe_str(row.get("difficulty")),
    }


def _extract_message_content(messages: Any, role: str) -> str:
    if not isinstance(messages, list):
        return ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if _safe_str(msg.get("role")).lower() == role.lower():
            return _safe_str(msg.get("content"))
    return ""


def _extract_cpp_block(text: str) -> str:
    text = _safe_str(text)
    if not text:
        return ""
    m = re.search(r"```(?:cpp|c\+\+|cxx)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    return _safe_str(m.group(1))


def _extract_codeforces(row: dict[str, Any], source: str, domain: str, verifiable: bool) -> dict[str, Any] | None:
    # Prefer chat-style rows when available (common in distilled CoT datasets).
    messages = row.get("messages")
    prompt = _extract_message_content(messages, "user")
    assistant_msg = _extract_message_content(messages, "assistant")

    # Fallback to structured columns.
    if not prompt:
        prompt = _safe_str(row.get("prompt") or row.get("question") or row.get("problem"))
    if not prompt:
        description = _safe_str(row.get("description"))
        statement = _safe_str(row.get("statement"))
        title = _safe_str(row.get("title") or row.get("name"))
        inp = _safe_str(row.get("input"))
        out = _safe_str(row.get("output"))
        prompt = "\n\n".join(
            x
            for x in [
                title,
                statement if statement else description,
                f"Input: {inp}" if inp else "",
                f"Output: {out}" if out else "",
            ]
            if x
        )

    # Pull useful supervision signals from multiple possible fields.
    reasoning = _safe_str(
        row.get("cot")
        or row.get("cot_content")
        or row.get("reasoning")
        or row.get("explanation")
        or row.get("editorial")
        or row.get("tutorial")
    )
    code = _safe_str(row.get("code") or row.get("completion") or row.get("program"))
    final_answer = _safe_str(row.get("answer") or row.get("final_answer") or row.get("solution"))

    # If the row is chat-form and assistant has content, mine reasoning/code from it.
    if assistant_msg:
        if not reasoning:
            reasoning = assistant_msg
        if not code:
            code = _extract_cpp_block(assistant_msg)
        if not final_answer:
            final_answer = _extract_final_answer_from_text(assistant_msg)

    if not prompt:
        return None

    if not final_answer:
        final_answer = code if code else "Implementation provided in reasoning."

    if code:
        reasoning = (reasoning + "\n\nReference implementation:\n```cpp\n" + code + "\n```").strip()

    assistant = _assistant_with_contract(reasoning, final_answer)
    return {
        "id": _build_id(source, prompt, final_answer),
        "source": source,
        "domain": domain,
        "language": "en",
        "prompt": prompt,
        "solution": final_answer,
        "assistant": assistant,
        "messages": _make_messages(prompt, assistant),
        "verifiable": verifiable,
        "difficulty": _safe_str(row.get("difficulty")),
    }


def _extract_hotpot(row: dict[str, Any], source: str, domain: str, verifiable: bool) -> dict[str, Any] | None:
    question = _safe_str(row.get("question"))
    answer = _safe_str(row.get("answer"))
    if not question or not answer:
        return None

    # HotpotQA fields often include:
    # - supporting_facts: {"title":[...], "sent_id":[...]}
    # - context: {"title":[...], "sentences":[[...], ...]}
    # Build concise evidence-aware reasoning from those fields.
    reasoning_parts: list[str] = []
    context = row.get("context")
    supporting = row.get("supporting_facts")

    context_title_to_sents: dict[str, list[str]] = {}
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        if isinstance(titles, list) and isinstance(sentences, list):
            for title, sents in zip(titles, sentences):
                key = _safe_str(title)
                if not key:
                    continue
                if isinstance(sents, list):
                    context_title_to_sents[key] = [_safe_str(x) for x in sents if _safe_str(x)]

    used_support = False
    if isinstance(supporting, dict):
        s_titles = supporting.get("title", [])
        s_ids = supporting.get("sent_id", [])
        if isinstance(s_titles, list) and isinstance(s_ids, list):
            for title, sid in zip(s_titles, s_ids):
                key = _safe_str(title)
                if key in context_title_to_sents:
                    sents = context_title_to_sents[key]
                    try:
                        idx = int(sid)
                    except Exception:
                        idx = -1
                    if 0 <= idx < len(sents):
                        evidence = sents[idx]
                    else:
                        evidence = sents[0] if sents else ""
                    if evidence:
                        reasoning_parts.append(f"{key}: {evidence}")
                        used_support = True

    # Fallback: use first 1-2 context snippets if supporting facts unavailable.
    if not used_support and context_title_to_sents:
        added = 0
        for title, sents in context_title_to_sents.items():
            if not sents:
                continue
            snippet = " ".join(sents[:2]).strip()
            if snippet:
                reasoning_parts.append(f"{title}: {snippet}")
                added += 1
            if added >= 2:
                break

    reasoning = " ".join(reasoning_parts).strip()
    if not reasoning:
        reasoning = "I identify the relevant facts and connect them to answer the question."

    assistant = _assistant_with_contract(reasoning, answer)
    return {
        "id": _build_id(source, question, answer),
        "source": source,
        "domain": domain,
        "language": "en",
        "prompt": question,
        "solution": answer,
        "assistant": assistant,
        "messages": _make_messages(question, assistant),
        "verifiable": verifiable,
        "difficulty": "",
    }


def _extract_openbookqa(row: dict[str, Any], source: str, domain: str, verifiable: bool) -> dict[str, Any] | None:
    question_stem = _safe_str((row.get("question_stem") if isinstance(row.get("question_stem"), str) else None))
    if not question_stem and isinstance(row.get("question"), dict):
        question_stem = _safe_str(row["question"].get("stem"))
    if not question_stem and isinstance(row.get("question"), str):
        question_stem = _safe_str(row.get("question"))
    if not question_stem:
        return None

    choices = None
    choice_pairs: list[tuple[str, str]] = []
    if isinstance(row.get("choices"), dict):
        labels = row["choices"].get("label", [])
        texts = row["choices"].get("text", [])
        if isinstance(labels, list) and isinstance(texts, list):
            for l, t in zip(labels, texts):
                ll = _safe_str(l)
                tt = _safe_str(t)
                if ll and tt:
                    choice_pairs.append((ll, tt))
            pairs = [f"{l}. {t}" for l, t in choice_pairs]
            choices = "\n".join(pairs)

    answer_key = _safe_str(row.get("answerKey"))
    final_answer = answer_key if answer_key else _safe_str(row.get("answer"))
    if not final_answer:
        return None

    final_answer_text = final_answer
    # Normalize "D" to option text when possible.
    if choice_pairs:
        lookup = {k: v for k, v in choice_pairs}
        if final_answer in lookup:
            final_answer_text = lookup[final_answer]

    prompt = question_stem if not choices else f"{question_stem}\n\nOptions:\n{choices}"
    # OpenBook often contains a supporting science fact under fact1.
    fact1 = _safe_str(row.get("fact1") or row.get("fact"))
    if fact1:
        reasoning = f"I use the given science fact and option elimination.\nFact: {fact1}"
    else:
        reasoning = "I use science facts and option elimination to choose the best answer."
    assistant = _assistant_with_contract(reasoning, final_answer_text)
    return {
        "id": _build_id(source, prompt, final_answer_text),
        "source": source,
        "domain": domain,
        "language": "en",
        "prompt": prompt,
        "solution": final_answer_text,
        "assistant": assistant,
        "messages": _make_messages(prompt, assistant),
        "verifiable": verifiable,
        "difficulty": "",
    }


def _map_record(ds_name: str, row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    extractors = {
        "numina": _extract_numina,
        "codeforces": _extract_codeforces,
        "hotpotqa": _extract_hotpot,
        "openbookqa": _extract_openbookqa,
    }
    extractor = extractors[ds_name]
    record = extractor(row, cfg["source"], cfg["domain"], cfg["verifiable"])
    if record is None:
        # Keep a fixed schema in map stage; drop in filter stage.
        return {
            "id": "",
            "source": "",
            "domain": "",
            "language": "",
            "prompt": "",
            "solution": "",
            "assistant": "",
            "messages": [],
            "verifiable": False,
            "difficulty": "",
            "carl_moa_route_hint": "",
            "carl_moa_needs_tools": False,
        }
    record["carl_moa_route_hint"] = _carl_moa_route_hint(record.get("domain", ""), bool(record.get("verifiable", False)))
    record["carl_moa_needs_tools"] = bool(record.get("verifiable", False) or record.get("domain", "") == "code")
    return record


def _has_valid_id(row: dict[str, Any]) -> bool:
    return bool(_safe_str(row.get("id")))


def _load_and_extract(
    ds_name: str,
    cfg: dict[str, Any],
    cache_dir: str,
    seed: int,
    target_count: int,
) -> Dataset:
    ds = load_dataset(
        cfg["hf_id"],
        cfg["config"],
        split=cfg["split"],
        cache_dir=cache_dir,
    )

    # Avoid loading full massive datasets into Python lists.
    # We pre-sample a bounded working set, then map/filter to final unified schema.
    working_n = min(len(ds), max(target_count * 3, target_count + 5000))
    ds = ds.shuffle(seed=seed).select(range(working_n))

    original_cols = ds.column_names
    ds = ds.map(
        lambda row: _map_record(ds_name, row, cfg),
        remove_columns=original_cols,
        desc=f"normalize-{ds_name}",
    )
    ds = ds.filter(_has_valid_id, desc=f"filter-invalid-{ds_name}")
    if len(ds) == 0:
        raise RuntimeError(f"No usable rows extracted from {cfg['hf_id']} ({cfg['config']}).")
    return ds


def _compute_targets(args: argparse.Namespace) -> dict[str, int]:
    ratios = {
        "numina": args.numina_ratio,
        "codeforces": args.codeforces_ratio,
        "hotpotqa": args.hotpotqa_ratio,
        "openbookqa": args.openbookqa_ratio,
    }
    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        raise ValueError("Sum of ratios must be > 0.")
    ratios = {k: v / total_ratio for k, v in ratios.items()}

    targets = {k: int(args.total_samples * v) for k, v in ratios.items()}
    # Ensure exact total by assigning the remainder to the largest bucket (numina).
    remainder = args.total_samples - sum(targets.values())
    targets["numina"] += remainder
    return targets


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    targets = _compute_targets(args)

    loaded: dict[str, Dataset] = {}
    for name, cfg in DEFAULT_DATASETS.items():
        print(f"[load] {name}: {cfg['hf_id']} ({cfg['config']})")
        ds = _load_and_extract(
            name,
            cfg,
            args.cache_dir,
            args.seed,
            targets[name],
        )
        loaded[name] = ds
        print(f"[ok]   {name}: extracted {len(ds)} rows")

    mixed_parts: list[Dataset] = []
    for name, target in targets.items():
        ds = loaded[name]
        take_n = min(target, len(ds))
        if take_n < target:
            print(f"[warn] {name}: requested {target}, only {len(ds)} available; taking {take_n}")
        mixed_parts.append(ds.select(range(take_n)))

    mixed = concatenate_datasets(mixed_parts).shuffle(seed=args.seed)

    if args.minimal_schema:
        keep_cols = [
            "id",
            "source",
            "messages",
        ]
    else:
        # CARL-MoA-friendly schema: includes direct prompt/solution supervision
        # and routing hints that can be reused in later GSPO routing studies.
        keep_cols = [
            "id",
            "source",
            "domain",
            "language",
            "verifiable",
            "difficulty",
            "prompt",
            "solution",
            "messages",
            "carl_moa_route_hint",
            "carl_moa_needs_tools",
        ]
    mixed = mixed.select_columns(keep_cols)

    out_path = Path(args.output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mixed.to_parquet(str(out_path))

    summary = {
        "output_parquet": str(out_path),
        "total_rows": len(mixed),
        "targets": targets,
        "actual_by_source": {
            name: min(targets[name], len(loaded[name])) for name in DEFAULT_DATASETS
        },
        "columns": keep_cols,
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] parquet written: {out_path}")
    print(f"[done] summary written: {summary_path}")


if __name__ == "__main__":
    main()
