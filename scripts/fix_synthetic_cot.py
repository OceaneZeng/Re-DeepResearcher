"""Fix synthetic CoT parquet (drop rows).

Implements:
1) Delete the first N rows from an existing output parquet (default 0).
2) Delete every row whose `cot_content` is considered an error:
   - NA / None
   - empty string
   - starts with `__ERROR__`

This script rewrites OUTPUT_PARQUET in-place but uses a temporary file for safety.

Example:
  python scripts/fix_synthetic_cot.py --output synthetic_cot_1000.parquet --drop-first 25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _is_error(v: object) -> bool:
    if v is None or v is pd.NA:
        return True
    s = str(v).strip()
    return s == "" or s.startswith("__ERROR__")


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, engine="pyarrow", index=True)
    tmp.replace(path)


def _find_repo_root(start: Path) -> Path:
    """Best-effort locate repo root (folder containing pyproject.toml)."""
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start.resolve()


def _resolve_output_path(output_arg: str) -> Path:
    """Resolve output path.

    Accepts:
    - absolute path
    - relative path
    - bare filename: will try repo root and then search within repo for a matching parquet
    """
    p = Path(output_arg)
    if p.is_absolute() and p.exists():
        return p

    # Try as relative to CWD
    if p.exists():
        return p.resolve()

    repo_root = _find_repo_root(Path.cwd())

    # Try relative to repo root
    p2 = repo_root / output_arg
    if p2.exists():
        return p2.resolve()

    # Try to locate by filename within repo
    filename = Path(output_arg).name
    matches = list(repo_root.rglob(filename))
    matches = [m for m in matches if m.is_file()]
    if len(matches) == 1:
        return matches[0].resolve()

    # Provide helpful error with candidates
    candidates = list(repo_root.rglob("*.parquet"))
    cand_str = "\n".join(f"  - {c}" for c in sorted(candidates)[:50])
    raise SystemExit(
        "Output parquet not found: "
        f"{output_arg}\n\n"
        "Tried:\n"
        f"  - {Path.cwd() / output_arg}\n"
        f"  - {repo_root / output_arg}\n\n"
        + ("Found multiple matches for filename:\n" + "\n".join(f"  - {m}" for m in matches) + "\n\n" if len(matches) > 1 else "")
        + ("Available parquet files under repo:\n" + cand_str + ("\n  ..." if len(candidates) > 50 else "") + "\n")
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=str)
    # Default to 0: do NOT drop the first rows unless explicitly requested.
    ap.add_argument("--drop-first", type=int, default=0)
    args = ap.parse_args()

    output_path = _resolve_output_path(args.output)

    out = pd.read_parquet(output_path)
    if "cot_content" not in out.columns:
        raise SystemExit("Output parquet has no cot_content column")

    # 1) optionally drop first N rows
    drop_n = max(0, min(args.drop_first, len(out)))
    out2 = out.iloc[drop_n:].copy()

    # 2) drop error rows
    mask_err = out2["cot_content"].map(_is_error)
    before = len(out2)
    out2 = out2.loc[~mask_err].copy()
    after = len(out2)

    _atomic_write_parquet(out2, output_path)

    print(f"Dropped first rows: {drop_n}")
    print(f"Dropped error rows: {before - after}")
    print(f"Remaining rows: {after}")
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
