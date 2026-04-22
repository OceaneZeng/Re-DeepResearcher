"""Fix synthetic CoT parquet by removing error rows.

This script rewrites the target parquet in-place but uses a temporary file for safety.
It removes rows whose `cot_content` is considered an error:
- NA / None
- empty string
- starts with `__ERROR__`

Example:
  python scripts/fix_synthetic_cot.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT = Path("data/subsets/synthetic_cot_1000.parquet")


def _is_error(v: object) -> bool:
    if v is None or v is pd.NA:
        return True
    s = str(v).strip()
    return s == "" or s.startswith("__ERROR__")


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, engine="pyarrow", index=True)
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Target parquet path, default: {DEFAULT_OUTPUT}",
    )
    args = ap.parse_args()

    output_path = Path(args.output).resolve()
    if not output_path.exists():
        raise SystemExit(f"Output parquet not found: {output_path}")

    out = pd.read_parquet(output_path)
    if "cot_content" not in out.columns:
        raise SystemExit("Output parquet has no cot_content column")

    before = len(out)
    mask_err = out["cot_content"].map(_is_error)
    out2 = out.loc[~mask_err].copy()
    after = len(out2)

    _atomic_write_parquet(out2, output_path)

    print(f"Dropped error rows: {before - after}")
    print(f"Remaining rows: {after}")
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
