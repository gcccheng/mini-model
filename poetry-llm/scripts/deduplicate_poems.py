#!/usr/bin/env python3
"""Conservatively remove exact title-and-body duplicates from parsed poetry JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_whitespace(value: str) -> str:
    return re.sub(r"\s+", "", value)


def fingerprint(title: str, body: str) -> str:
    normalized = f"{compact_whitespace(title)}\n{compact_whitespace(body)}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def body_fingerprint(body: str) -> str:
    return hashlib.sha256(compact_whitespace(body).encode("utf-8")).hexdigest()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--duplicates", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    with args.source.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            records.append(record)

    first_by_key: dict[str, dict[str, Any]] = {}
    group_counts: Counter[str] = Counter()
    unique: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    conflict_counts: Counter[str] = Counter()
    body_titles: dict[str, set[str]] = defaultdict(set)

    for record in records:
        title = str(record["title"])
        body = str(record["text"])
        key = fingerprint(title, body)
        group_counts[key] += 1

        body_titles[body_fingerprint(body)].add(compact_whitespace(title))

        if key not in first_by_key:
            first_by_key[key] = record
            unique.append(record)
            continue

        kept = first_by_key[key]
        conflicts = [
            field
            for field in ("dynasty", "author")
            if record.get(field) != kept.get(field)
        ]
        conflict_counts.update(conflicts)
        duplicates.append(
            {
                "fingerprint": key,
                "kept_id": kept["id"],
                "kept_source_block": kept["source_block"],
                "removed_id": record["id"],
                "removed_source_block": record["source_block"],
                "metadata_conflicts": conflicts,
                "removed_record": record,
            }
        )

    write_jsonl(args.destination, unique)
    write_jsonl(args.duplicates, duplicates)

    duplicate_groups = [count for count in group_counts.values() if count > 1]
    same_body_different_title_groups = sum(
        len(titles) > 1 for titles in body_titles.values()
    )
    largest_group = max(duplicate_groups, default=1)

    report_lines = [
        "诗词精确去重报告 / Exact poetry deduplication report",
        "=" * 58,
        f"源文件 / Source: {args.source}",
        f"唯一记录 / Unique output: {args.destination}",
        f"重复审计 / Duplicate audit: {args.duplicates}",
        f"源文件 SHA-256 / Source SHA-256: {sha256_file(args.source)}",
        "",
        "去重结果 / Deduplication results",
        "-" * 38,
        f"输入记录 / Input records: {len(records):,}",
        f"保留记录 / Retained records: {len(unique):,}",
        f"移除重复 / Removed duplicates: {len(duplicates):,}",
        f"重复组数 / Duplicate groups: {len(duplicate_groups):,}",
        f"最大重复组 / Largest duplicate group: {largest_group:,}",
        "正文相同但标题不同的组 / Same-body, different-title groups: "
        f"{same_body_different_title_groups:,}",
        "",
        "元数据冲突 / Metadata conflicts among removed duplicates",
        "-" * 60,
        f"朝代冲突 / Dynasty conflicts: {conflict_counts['dynasty']:,}",
        f"作者冲突 / Author conflicts: {conflict_counts['author']:,}",
        "",
        f"唯一文件 SHA-256 / Unique SHA-256: {sha256_file(args.destination)}",
        f"重复文件 SHA-256 / Duplicate SHA-256: {sha256_file(args.duplicates)}",
        "",
        "规则 / Rule:",
        "删除所有空白后，标题与正文都相同才视为精确重复。保留首次出现的记录。",
        "A record is an exact duplicate only when both title and body match after",
        "whitespace removal. The first occurrence is retained.",
        "正文相同但标题不同的记录不会自动删除。",
        "Records with the same body but different titles are not removed automatically.",
    ]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
