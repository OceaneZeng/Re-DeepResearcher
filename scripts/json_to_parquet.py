"""Convert JSON or JSONL files to Parquet.

Input shapes:
- JSON: a single object (one row) or an array of objects (many rows).
- JSONL / NDJSON: one JSON object per line (blank lines skipped).

By default, nested ``list[dict]`` values are normalized so every dict in the same list
shares the same key set (missing keys become ``None``). This avoids PyArrow errors when
a column mixes incompatible struct shapes (e.g. ShareGPT ``conversations`` with both
``{from, value}`` and ``{from, value, markdown, text}`` turns in one thread).

If the same struct field still mixes ``dict`` with scalars/lists (e.g. ``markdown`` as
either a string or a JSON object across the dataset), dict values at that field are
serialized to JSON strings so Parquet can use a single string type.

Examples:
  python scripts/json_to_parquet.py data.jsonl -o out.parquet
  python scripts/json_to_parquet.py data.json -o out.parquet --compression zstd
  python scripts/json_to_parquet.py records.ndjson -o out_dir --partition-rows 500000
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Marks struct fields inside a homogeneous list[dict] when building (path, field) keys.
_LD_ITEM = "__listdict_item__"


def _normalize_for_arrow_structs(v: Any) -> Any:
    """Align dict keys inside homogeneous ``list[dict]`` subtrees for stable Parquet structs."""
    if isinstance(v, list):
        if v and all(isinstance(x, dict) for x in v):
            keys = sorted({kk for x in v for kk in x.keys()})
            return [{k: _normalize_for_arrow_structs(x.get(k)) for k in keys} for x in v]
        return [_normalize_for_arrow_structs(x) for x in v]
    if isinstance(v, dict):
        return {k: _normalize_for_arrow_structs(vv) for k, vv in v.items()}
    return v


def _collect_listdict_field_types(v: Any, path: tuple[str, ...], acc: dict[tuple[str, ...], set[type]]) -> None:
    """Record Python types seen at each list-of-struct field path (for mixed dict/scalar fixes)."""
    if isinstance(v, dict):
        for k, vv in v.items():
            _collect_listdict_field_types(vv, path + (k,), acc)
        return
    if isinstance(v, list):
        if v and all(isinstance(x, dict) for x in v):
            keys = sorted({kk for x in v for kk in x.keys()})
            for kk in keys:
                fp = path + (_LD_ITEM, kk)
                for x in v:
                    cell = x.get(kk)
                    if cell is None:
                        continue
                    acc.setdefault(fp, set()).add(type(cell))
                    _collect_listdict_field_types(cell, fp, acc)
        else:
            for x in v:
                _collect_listdict_field_types(x, path, acc)


def _path_needs_dict_to_json(types: set[type]) -> bool:
    """True when some cells are dicts and others are non-dict (excluding None)."""
    if dict not in types:
        return False
    return any(t is not dict and t is not type(None) for t in types)


def _coerce_mixed_dict_fields(v: Any, path: tuple[str, ...], stringify_paths: set[tuple[str, ...]]) -> Any:
    if isinstance(v, dict):
        return {k: _coerce_mixed_dict_fields(vv, path + (k,), stringify_paths) for k, vv in v.items()}
    if isinstance(v, list):
        if v and all(isinstance(x, dict) for x in v):
            keys = sorted({kk for x in v for kk in x.keys()})
            out: list[dict[str, Any]] = []
            for x in v:
                nd: dict[str, Any] = {}
                for kk in keys:
                    fp = path + (_LD_ITEM, kk)
                    cv = x.get(kk)
                    if fp in stringify_paths and isinstance(cv, dict):
                        cv = json.dumps(cv, ensure_ascii=False)
                    else:
                        cv = _coerce_mixed_dict_fields(cv, fp, stringify_paths)
                    nd[kk] = cv
                out.append(nd)
            return out
        return [_coerce_mixed_dict_fields(x, path, stringify_paths) for x in v]
    return v


def _prepare_records(records: list[dict[str, Any]], *, normalize_structs: bool) -> list[dict[str, Any]]:
    if not normalize_structs:
        return records
    aligned = [_normalize_for_arrow_structs(r) for r in records]
    acc: dict[tuple[str, ...], set[type]] = {}
    for r in aligned:
        _collect_listdict_field_types(r, (), acc)
    stringify_paths = {p for p, ts in acc.items() if _path_needs_dict_to_json(ts)}
    if not stringify_paths:
        return aligned
    return [_coerce_mixed_dict_fields(r, (), stringify_paths) for r in aligned]


def _open_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz" or path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    with _open_text(path) as f:
        raw = f.read()
    data = json.loads(raw)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        if not data:
            return []
        if not all(isinstance(x, dict) for x in data):
            raise SystemExit("JSON array must contain only objects (dicts) for tabular Parquet output.")
        return data
    raise SystemExit(f"Unsupported JSON root type: {type(data).__name__}")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise SystemExit(f"{path}:{line_no}: each JSONL line must be a JSON object (dict).")
            yield obj


def _detect_format(path: Path) -> str:
    suf = path.suffix.lower()
    name = path.name.lower()
    if name.endswith(".jsonl.gz") or name.endswith(".ndjson.gz"):
        return "jsonl"
    if suf in {".jsonl", ".ndjson", ".jsonlines"}:
        return "jsonl"
    if suf == ".json" or name.endswith(".json.gz"):
        return "json"
    # Heuristic: peek first non-empty line
    with _open_text(path) as f:
        for line in f:
            s = line.lstrip()
            if not s:
                continue
            if s[0] in "[{":
                return "json"
            return "jsonl"
    return "jsonl"


def _records_to_table(records: list[dict[str, Any]], *, normalize_structs: bool) -> pa.Table:
    if not records:
        return pa.table({})
    recs = _prepare_records(records, normalize_structs=normalize_structs)
    return pa.Table.from_pylist(recs)


def _write_single_table(table: pa.Table, out: Path, compression: str | None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression=compression)


def _write_partitioned_batches(
    record_iter: Iterator[dict[str, Any]],
    out_dir: Path,
    batch_size: int,
    compression: str | None,
    stem: str,
    *,
    normalize_structs: bool,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    batch: list[dict[str, Any]] = []
    part_idx = 0
    total = 0
    for row in record_iter:
        batch.append(row)
        total += 1
        if len(batch) >= batch_size:
            table = _records_to_table(batch, normalize_structs=normalize_structs)
            _write_single_table(table, out_dir / f"{stem}_part-{part_idx:05d}.parquet", compression)
            part_idx += 1
            batch.clear()
    if batch:
        table = _records_to_table(batch, normalize_structs=normalize_structs)
        _write_single_table(table, out_dir / f"{stem}_part-{part_idx:05d}.parquet", compression)
    return total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input .json / .jsonl path (default: read stdin as JSONL)",
    )
    p.add_argument("-o", "--output", required=True, help="Output .parquet file or directory (when partitioning)")
    p.add_argument(
        "--input-format",
        choices=("auto", "json", "jsonl"),
        default="auto",
        help="How to parse input (default: auto from extension / first bytes)",
    )
    p.add_argument(
        "--compression",
        default="snappy",
        help="Parquet compression codec (snappy, zstd, gzip, none, …). Default: snappy",
    )
    p.add_argument(
        "--partition-rows",
        type=int,
        default=0,
        metavar="N",
        help="If >0 and input is JSONL, write multiple {stem}_part-*.parquet files with at most N rows each",
    )
    p.add_argument(
        "--index",
        action="store_true",
        help="If set, keep pandas RangeIndex as a column in the Parquet output",
    )
    p.set_defaults(normalize_structs=True)
    p.add_argument(
        "--no-normalize-structs",
        dest="normalize_structs",
        action="store_false",
        help="Disable struct normalization (ShareGPT-style mixed keys / dict-vs-string fields will often fail)",
    )
    args = p.parse_args(argv)
    compression = None if args.compression.lower() in {"none", "null"} else args.compression
    if args.input is None:
        if args.input_format not in {"auto", "jsonl"}:
            p.error("stdin mode only supports JSONL; use --input-format jsonl or auto.")
        records = []
        for line_no, line in enumerate(sys.stdin, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise SystemExit(f"stdin:{line_no}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise SystemExit(f"stdin:{line_no}: each line must be a JSON object (dict).")
            records.append(obj)
        records = _prepare_records(records, normalize_structs=args.normalize_structs)
        df = pd.DataFrame.from_records(records)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, engine="pyarrow", index=args.index, compression=compression)
        print(f"Wrote {len(df)} rows -> {out.resolve()}")
        return 0
    in_path = Path(args.input)
    if not in_path.is_file():
        raise SystemExit(f"Input not found or not a file: {in_path}")
    fmt = args.input_format
    if fmt == "auto":
        fmt = _detect_format(in_path)

    out_path = Path(args.output)
    if fmt == "json":
        records = _prepare_records(_load_json_records(in_path), normalize_structs=args.normalize_structs)
        df = pd.DataFrame.from_records(records)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, engine="pyarrow", index=args.index, compression=compression)
        print(f"Wrote {len(df)} rows -> {out_path.resolve()}")
        return 0
    # JSONL
    if args.partition_rows and args.partition_rows > 0:
        # If user passed `out.parquet`, use sibling folder `out/` for shard files.
        if out_path.suffix.lower() == ".parquet":
            stem = out_path.stem
            out_dir = out_path.with_suffix("")
        else:
            out_dir = out_path
            stem = in_path.stem
        n = _write_partitioned_batches(
            _iter_jsonl(in_path),
            out_dir,
            args.partition_rows,
            compression,
            stem,
            normalize_structs=args.normalize_structs,
        )
        print(f"Wrote {n} rows into partitioned parquet under {out_dir.resolve()}")
        return 0

    records = _prepare_records(list(_iter_jsonl(in_path)), normalize_structs=args.normalize_structs)
    df = pd.DataFrame.from_records(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=args.index, compression=compression)
    print(f"Wrote {len(df)} rows -> {out_path.resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
