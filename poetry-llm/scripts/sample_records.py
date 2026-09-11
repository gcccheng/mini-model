#!/usr/bin/env python3
"""Create a deterministic, human-readable sample from each dataset split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=10)
    parser.add_argument("--validation-count", type=int, default=5)
    parser.add_argument("--test-count", type=int, default=5)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    requested = {
        "train": (args.train, args.train_count),
        "validation": (args.validation, args.validation_count),
        "test": (args.test, args.test_count),
    }

    output_lines = [
        "随机人工抽查样本 / Random manual-review sample",
        "=" * 55,
        f"随机种子 / Random seed: {args.seed}",
        "",
    ]
    for split_name, (path, count) in requested.items():
        records = load_jsonl(path)
        samples = rng.sample(records, min(count, len(records)))
        output_lines.extend([f"[{split_name}]", "-" * 20])
        for index, record in enumerate(samples, start=1):
            output_lines.extend(
                [
                    f"Sample {index}",
                    f"ID: {record['id']}",
                    f"标题 / Title: {record['title']}",
                    f"朝代 / Dynasty: {record['dynasty']}",
                    f"作者 / Author: {record['author'] or '<EMPTY>'}",
                    f"正文长度 / Body length: {len(record['text'])}",
                    f"正文 / Body: {record['text']}",
                    "",
                ]
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
