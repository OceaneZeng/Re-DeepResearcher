#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse
import glob
import os
from typing import Iterable, List

from datasets import Dataset, load_dataset


def _expand_files(patterns: Iterable[str]) -> List[str]:
    files: List[str] = []
    for p in patterns:
        if not p:
            continue
        # Allow comma-separated lists in a single arg.
        for part in str(p).split(","):
            part = part.strip().strip('"').strip("'")
            if not part:
                continue
            matched = glob.glob(part)
            if matched:
                files.extend(matched)
            else:
                # Keep as-is (may be an exact path).
                files.append(part)
    # Normalize + de-dup while preserving order
    seen = set()
    out: List[str] = []
    for f in files:
        nf = os.path.normpath(f)
        if nf not in seen:
            seen.add(nf)
            out.append(nf)
    return out


def _load_parquet_dataset(files: List[str]) -> Dataset:
    if not files:
        raise ValueError("No parquet files provided.")
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Parquet file not found: {f}")
    ds = load_dataset("parquet", data_files=files, split="train")
    return ds


def _sample_to_parquet(
    *,
    input_files: List[str],
    n: int,
    seed: int,
    out_path: str,
) -> None:
    ds = _load_parquet_dataset(input_files)
    total = len(ds)
    if n > total:
        raise ValueError(f"Requested {n} rows but dataset only has {total} rows. ({out_path})")

    sampled = ds.shuffle(seed=seed).select(range(n))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    sampled.to_parquet(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Randomly sample subsets from parquet datasets and save as new parquet files.\n"
            "Uses Hugging Face Datasets (Arrow) for efficient shuffling + writing."
        )
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for shuffling.")

    ap.add_argument(
        "--sharegpt",
        nargs="+",
        required=True,
        help="ShareGPT parquet path(s) or glob(s). Example: data/sharegpt/*.parquet",
    )
    ap.add_argument(
        "--dolly",
        nargs="+",
        required=True,
        help="Dolly parquet path(s) or glob(s).",
    )
    ap.add_argument(
        "--codealpaca",
        nargs="+",
        required=True,
        help="Code Alpaca parquet path(s) or glob(s).",
    )
    ap.add_argument(
        "--gsm8k",
        nargs="+",
        required=True,
        help="GSM8K parquet path(s) or glob(s).",
    )

    ap.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Output directory. Files will be written as sharegpt_4000.parquet, dolly_1600.parquet, code_alpaca_800.parquet, gsm8k_800.parquet",
    )

    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    sharegpt_out = os.path.join(out_dir, "sharegpt_4000.parquet")
    dolly_out = os.path.join(out_dir, "dolly_1600.parquet")
    codealpaca_out = os.path.join(out_dir, "code_alpaca_800.parquet")
    gsm8k_out = os.path.join(out_dir, "gsm8k_800.parquet")

    _sample_to_parquet(
        input_files=_expand_files(args.sharegpt),
        n=4000,
        seed=args.seed,
        out_path=sharegpt_out,
    )
    _sample_to_parquet(
        input_files=_expand_files(args.dolly),
        n=1600,
        seed=args.seed,
        out_path=dolly_out,
    )
    _sample_to_parquet(
        input_files=_expand_files(args.codealpaca),
        n=800,
        seed=args.seed,
        out_path=codealpaca_out,
    )
    _sample_to_parquet(
        input_files=_expand_files(args.gsm8k),
        n=800,
        seed=args.seed,
        out_path=gsm8k_out,
    )

    print("Wrote:")
    print(f"- {sharegpt_out}")
    print(f"- {dolly_out}")
    print(f"- {codealpaca_out}")
    print(f"- {gsm8k_out}")


if __name__ == "__main__":
    main()

"""
python scripts/sample_parquet_subsets.py --sharegpt "data/ShareGPT_html_cleaned.parquet" --dolly "data/dolly.parquet" --codealpaca "data/code_alpaca.parquet" --gsm8k "data/gsm8k_train-00000-of-00001.parquet" --out_dir "data/subsets" --seed 42
"""