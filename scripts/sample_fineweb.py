<<<<<<< HEAD
"""Sample rows from a fineweb parquet file.

Usage:
  python scripts/sample_fineweb.py --input data/FineWeb/fineweb_000_00000.parquet --output data/FineWeb/fineweb.parquet --num-rows 800

This script randomly samples rows without replacement and writes them to a new parquet file.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=path.suffix, dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_parquet(tmp_path, engine="pyarrow", index=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def sample_fineweb(input_path: Path, output_path: Path, num_rows: int, seed: int) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    df = pd.read_parquet(input_path, engine="pyarrow")
    if df.empty:
        raise ValueError(f"Input parquet is empty: {input_path}")
    if num_rows <= 0:
        raise ValueError("num_rows must be a positive integer")
    if len(df) < num_rows:
        raise ValueError(f"Requested {num_rows} rows, but only {len(df)} rows are available")

    sampled = df.sample(n=num_rows, replace=False, random_state=seed).reset_index(drop=True)
    _atomic_write_parquet(sampled, output_path)
    return sampled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Randomly sample rows from a FineWeb parquet file.")
    parser.add_argument("--input", type=Path, default=Path("data/FineWeb/fineweb_000_00000.parquet"), help="Input parquet file")
    parser.add_argument("--output", type=Path, default=Path("data/FineWeb/fineweb.parquet"), help="Output parquet file")
    parser.add_argument("--num-rows", type=int, default=800, help="Number of rows to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sampled = sample_fineweb(args.input, args.output, args.num_rows, args.seed)
    print(f"Wrote {len(sampled)} sampled rows to {args.output}")


if __name__ == "__main__":
    main()

=======
"""Sample rows from a fineweb parquet file.

Usage:
  python scripts/sample_fineweb.py --input data/FineWeb/fineweb_000_00000.parquet --output data/FineWeb/fineweb.parquet --num-rows 800

This script randomly samples rows without replacement and writes them to a new parquet file.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=path.suffix, dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_parquet(tmp_path, engine="pyarrow", index=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def sample_fineweb(input_path: Path, output_path: Path, num_rows: int, seed: int) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    df = pd.read_parquet(input_path, engine="pyarrow")
    if df.empty:
        raise ValueError(f"Input parquet is empty: {input_path}")
    if num_rows <= 0:
        raise ValueError("num_rows must be a positive integer")
    if len(df) < num_rows:
        raise ValueError(f"Requested {num_rows} rows, but only {len(df)} rows are available")

    sampled = df.sample(n=num_rows, replace=False, random_state=seed).reset_index(drop=True)
    _atomic_write_parquet(sampled, output_path)
    return sampled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Randomly sample rows from a FineWeb parquet file.")
    parser.add_argument("--input", type=Path, default=Path("data/FineWeb/fineweb_000_00000.parquet"), help="Input parquet file")
    parser.add_argument("--output", type=Path, default=Path("data/FineWeb/fineweb.parquet"), help="Output parquet file")
    parser.add_argument("--num-rows", type=int, default=800, help="Number of rows to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sampled = sample_fineweb(args.input, args.output, args.num_rows, args.seed)
    print(f"Wrote {len(sampled)} sampled rows to {args.output}")


if __name__ == "__main__":
    main()

>>>>>>> 13bd3ebcf84d7955159bd742ecfa3aabe56b2500
