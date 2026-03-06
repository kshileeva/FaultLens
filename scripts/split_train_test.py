from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def train_test_split(
    rows: List[Dict[str, Any]],
    test_frac: float,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not (0.0 < test_frac < 1.0):
        raise ValueError("test_frac must be between 0 and 1.")
    n = len(rows)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_test = max(1, int(round(n * test_frac)))
    test_idx = set(indices[:n_test])
    train_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    for i, obj in enumerate(rows):
        if i in test_idx:
            test_rows.append(obj)
        else:
            train_rows.append(obj)
    return train_rows, test_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Split cleaned JSONL into train/test JSONL files.")
    ap.add_argument(
        "--in",
        dest="inp",
        type=Path,
        default=Path("data/train_sft.cleaned.jsonl"),
        help="Input cleaned JSONL file.",
    )
    ap.add_argument(
        "--train_out",
        type=Path,
        default=Path("data/train.jsonl"),
        help="Output train JSONL path.",
    )
    ap.add_argument(
        "--test_out",
        type=Path,
        default=Path("data/test.jsonl"),
        help="Output test JSONL path.",
    )
    ap.add_argument(
        "--test_frac",
        type=float,
        default=0.1,
        help="Fraction of examples to put in the test split (default: 0.1).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling before split.",
    )
    args = ap.parse_args()

    inp: Path = args.inp
    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")

    rows = load_jsonl(inp)
    print(f"Loaded {len(rows)} examples from {inp}")
    train_rows, test_rows = train_test_split(rows, test_frac=args.test_frac, seed=args.seed)
    print(f"Train: {len(train_rows)}  Test: {len(test_rows)} (test_frac={args.test_frac})")

    save_jsonl(args.train_out, train_rows)
    save_jsonl(args.test_out, test_rows)
    print(f"Wrote train split to: {args.train_out}")
    print(f"Wrote test split to:  {args.test_out}")


if __name__ == "__main__":
    main()

