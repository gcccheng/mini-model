#!/usr/bin/env python3
"""Parse the normalized four-line poetry corpus into JSON Lines."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse blank-line-separated, four-line poetry records into JSONL."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    text = args.source.read_text(encoding="utf-8")
    blocks = [block for block in text.split("\n\n") if block.strip()]

    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    physical_line_counts: Counter[int] = Counter()
    dynasty_counts: Counter[str] = Counter()
    empty_author_count = 0
    empty_dynasty_count = 0
    dynasty_without_closing_bracket = 0

    for block_number, block in enumerate(blocks, start=1):
        # Keep whitespace-only physical lines. One source record uses such a
        # line to represent an unknown author; removing it would corrupt the
        # four-field layout.
        lines = block.splitlines()
        physical_line_counts[len(lines)] += 1

        if len(lines) != 4:
            rejected.append(
                {
                    "source_block": block_number,
                    "reason": "physical_line_count_not_4",
                    "physical_line_count": len(lines),
                    "raw_lines": lines,
                }
            )
            continue

        title = lines[0].strip()
        raw_dynasty = lines[1].strip()
        author = lines[2].strip()
        body = lines[3].strip()

        errors: list[str] = []
        if not title:
            errors.append("empty_title")
        if not body:
            errors.append("empty_body")

        if errors:
            rejected.append(
                {
                    "source_block": block_number,
                    "reason": ",".join(errors),
                    "physical_line_count": len(lines),
                    "raw_lines": lines,
                }
            )
            continue

        if raw_dynasty.endswith("]"):
            dynasty = raw_dynasty[:-1].strip()
        else:
            dynasty = raw_dynasty
            dynasty_without_closing_bracket += 1

        if not dynasty:
            empty_dynasty_count += 1
        if not author:
            empty_author_count += 1

        dynasty_counts[dynasty] += 1
        accepted.append(
            {
                "id": f"poem-{block_number:07d}",
                "title": title,
                "dynasty": dynasty,
                "author": author,
                "text": body,
                "source_block": block_number,
            }
        )

    write_jsonl(args.destination, accepted)
    write_jsonl(args.rejected, rejected)

    total_body_characters = sum(len(str(record["text"])) for record in accepted)
    report_lines = [
        "诗词语料解析报告 / Poetry corpus parsing report",
        "=" * 52,
        f"源文件 / Source: {args.source}",
        f"输出文件 / Output: {args.destination}",
        f"拒绝记录 / Rejected: {args.rejected}",
        f"源文件 SHA-256 / Source SHA-256: {sha256(args.source)}",
        "",
        "解析结果 / Parsing results",
        "-" * 30,
        f"记录块总数 / Total blocks: {len(blocks):,}",
        f"接受记录 / Accepted records: {len(accepted):,}",
        f"拒绝记录 / Rejected records: {len(rejected):,}",
        f"正文字符数 / Body characters: {total_body_characters:,}",
        f"朝代为空 / Empty dynasty: {empty_dynasty_count:,}",
        f"作者为空 / Empty author: {empty_author_count:,}",
        "缺少右方括号的朝代字段 / Dynasty fields without trailing ']': "
        f"{dynasty_without_closing_bracket:,}",
        "",
        "物理行数分布 / Physical-line distribution",
        "-" * 45,
    ]
    for line_count, count in sorted(physical_line_counts.items()):
        report_lines.append(f"{line_count} lines: {count:,} blocks")

    report_lines.extend(
        [
            "",
            "朝代分布 / Dynasty distribution",
            "-" * 35,
        ]
    )
    for dynasty, count in dynasty_counts.most_common():
        report_lines.append(f"{dynasty or '<EMPTY>'}: {count:,}")

    report_lines.extend(
        [
            "",
            f"输出 SHA-256 / Output SHA-256: {sha256(args.destination)}",
            f"拒绝文件 SHA-256 / Rejected SHA-256: {sha256(args.rejected)}",
            "",
            "说明 / Note:",
            "本步骤仅解析结构，不删除正文中的注释、赏析或非诗歌内容。",
            "This step parses structure only; it does not remove annotations,",
            "commentary, or non-poetry content from accepted bodies.",
        ]
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
