#!/usr/bin/env python3
"""Independently verify split integrity and leakage prevention."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def body_fingerprint(record: dict[str, Any]) -> str:
    body = re.sub(r"\s+", "", str(record["text"]))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    source = load_jsonl(args.source)
    splits = {
        "train": load_jsonl(args.train),
        "validation": load_jsonl(args.validation),
        "test": load_jsonl(args.test),
    }

    source_by_id = {record["id"]: record for record in source}
    id_sets = {
        name: {record["id"] for record in records}
        for name, records in splits.items()
    }
    body_sets = {
        name: {body_fingerprint(record) for record in records}
        for name, records in splits.items()
    }
    source_block_sets = {
        name: {record["source_block"] for record in records}
        for name, records in splits.items()
    }

    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    id_overlaps = {f"{a}/{b}": len(id_sets[a] & id_sets[b]) for a, b in pairs}
    body_overlaps = {f"{a}/{b}": len(body_sets[a] & body_sets[b]) for a, b in pairs}
    source_overlaps = {
        f"{a}/{b}": len(source_block_sets[a] & source_block_sets[b]) for a, b in pairs
    }

    combined = [record for records in splits.values() for record in records]
    combined_by_id = {record["id"]: record for record in combined}
    missing_ids = set(source_by_id) - set(combined_by_id)
    extra_ids = set(combined_by_id) - set(source_by_id)
    changed_ids = {
        record_id
        for record_id in set(source_by_id) & set(combined_by_id)
        if source_by_id[record_id] != combined_by_id[record_id]
    }
    duplicate_ids_within_splits = {
        name: len(records) - len(id_sets[name]) for name, records in splits.items()
    }

    passed = not any(
        [
            *id_overlaps.values(),
            *body_overlaps.values(),
            *source_overlaps.values(),
            *duplicate_ids_within_splits.values(),
            len(missing_ids),
            len(extra_ids),
            len(changed_ids),
        ]
    )

    report_lines = [
        "数据集完整性报告 / Dataset integrity report",
        "=" * 49,
        f"源记录 / Source records: {len(source):,}",
        f"训练记录 / Train records: {len(splits['train']):,}",
        f"验证记录 / Validation records: {len(splits['validation']):,}",
        f"测试记录 / Test records: {len(splits['test']):,}",
        f"合计 / Combined records: {len(combined):,}",
        "",
        "跨集合重叠 / Cross-split overlap",
        "-" * 38,
    ]
    for pair in id_overlaps:
        report_lines.append(
            f"{pair}: IDs={id_overlaps[pair]:,}, "
            f"body fingerprints={body_overlaps[pair]:,}, "
            f"source blocks={source_overlaps[pair]:,}"
        )

    report_lines.extend(
        [
            "",
            "记录守恒 / Record accounting",
            "-" * 34,
            f"缺失 ID / Missing IDs: {len(missing_ids):,}",
            f"额外 ID / Extra IDs: {len(extra_ids):,}",
            f"内容被改写 / Changed records: {len(changed_ids):,}",
            f"训练集内部重复 ID / Duplicate train IDs: {duplicate_ids_within_splits['train']:,}",
            "验证集内部重复 ID / Duplicate validation IDs: "
            f"{duplicate_ids_within_splits['validation']:,}",
            f"测试集内部重复 ID / Duplicate test IDs: {duplicate_ids_within_splits['test']:,}",
            "",
            f"最终结果 / Final result: {'PASS' if passed else 'FAIL'}",
        ]
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
