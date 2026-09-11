#!/usr/bin/env python3
"""Validate parsed poetry JSONL and report suspicious content."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "id": str,
    "title": str,
    "dynasty": str,
    "author": str,
    "text": str,
    "source_block": int,
}

KNOWN_DYNASTIES = {
    "",
    "先秦",
    "两汉",
    "魏晋",
    "南北朝",
    "隋朝",
    "唐朝",
    "五代",
    "宋朝",
    "元朝",
    "明朝",
    "清朝",
    "近代",
    "现代",
}

CONTAMINATION_MARKERS = (
    "相关成语",
    "相关楹联",
    "词句解析",
    "英文翻译",
    "白话译文",
    "出自",
    "注释",
    "赏析",
    "译文",
    "更多关于",
)

ID_PATTERN = re.compile(r"^poem-\d{7}$")
HTML_PATTERN = re.compile(r"<[^>\n]{1,200}>")
URL_PATTERN = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unexpected_controls(value: str) -> list[str]:
    return sorted(
        {
            character
            for character in value
            if unicodedata.category(character) == "Cc"
            and character not in "\n\r\t"
        }
    )


def is_latin_letter(character: str) -> bool:
    return unicodedata.category(character).startswith("L") and "LATIN" in unicodedata.name(
        character, ""
    )


def content_reasons(record: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    title = record["title"]
    dynasty = record["dynasty"]
    author = record["author"]
    body = record["text"]

    if not title.strip():
        reasons.append("empty_title")
    if not body.strip():
        reasons.append("empty_body")
    if not dynasty.strip():
        reasons.append("empty_dynasty")
    elif dynasty not in KNOWN_DYNASTIES:
        reasons.append("unknown_dynasty")
    if not author.strip():
        reasons.append("empty_author")

    body_length = len(body)
    if body_length < 8:
        reasons.append("body_shorter_than_8")
    if body_length > 500:
        reasons.append("body_longer_than_500")
    if body_length > 1_000:
        reasons.append("body_longer_than_1000")
    if body_length > 5_000:
        reasons.append("body_longer_than_5000")

    if "\ufffd" in title + dynasty + author + body:
        reasons.append("unicode_replacement_character")
    if unexpected_controls(title + dynasty + author + body):
        reasons.append("unexpected_control_character")
    if HTML_PATTERN.search(body):
        reasons.append("possible_html")
    if URL_PATTERN.search(body):
        reasons.append("contains_url")

    nonspace_length = sum(not character.isspace() for character in body)
    latin_count = sum(is_latin_letter(character) for character in body)
    if nonspace_length >= 20 and latin_count / nonspace_length > 0.25:
        reasons.append("latin_heavy_body")
    if sum(is_latin_letter(character) for character in body[:80]) >= 15:
        reasons.append("latin_heavy_prefix")

    for marker in CONTAMINATION_MARKERS:
        if marker in body:
            reasons.append(f"contains_marker:{marker}")

    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--suspicious", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    valid_records = 0
    json_errors = 0
    schema_errors = 0
    duplicate_ids = 0
    duplicate_source_blocks = 0
    seen_ids: set[str] = set()
    seen_source_blocks: set[int] = set()
    suspicious_records: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    body_lengths: list[int] = []

    with args.source.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                json_errors += 1
                reason = "invalid_json"
                reason_counts[reason] += 1
                suspicious_records.append(
                    {
                        "line_number": line_number,
                        "reasons": [reason],
                        "error": str(error),
                        "raw_excerpt": line[:500],
                    }
                )
                continue

            structural_reasons: list[str] = []
            if not isinstance(record, dict):
                structural_reasons.append("record_not_object")
            else:
                for field, expected_type in REQUIRED_FIELDS.items():
                    if field not in record:
                        structural_reasons.append(f"missing_field:{field}")
                    elif not isinstance(record[field], expected_type):
                        structural_reasons.append(f"wrong_type:{field}")

            if structural_reasons:
                schema_errors += 1
                reason_counts.update(structural_reasons)
                suspicious_records.append(
                    {
                        "line_number": line_number,
                        "reasons": structural_reasons,
                        "record": record,
                    }
                )
                continue

            valid_records += 1
            record_id = record["id"]
            source_block = record["source_block"]
            reasons: list[str] = []

            if not ID_PATTERN.fullmatch(record_id):
                reasons.append("invalid_id_format")
            if record_id in seen_ids:
                duplicate_ids += 1
                reasons.append("duplicate_id")
            seen_ids.add(record_id)

            if source_block in seen_source_blocks:
                duplicate_source_blocks += 1
                reasons.append("duplicate_source_block")
            seen_source_blocks.add(source_block)

            reasons.extend(content_reasons(record))
            body_lengths.append(len(record["text"]))

            if reasons:
                reason_counts.update(reasons)
                suspicious_records.append(
                    {
                        "line_number": line_number,
                        "reasons": reasons,
                        "record": record,
                    }
                )

    args.suspicious.parent.mkdir(parents=True, exist_ok=True)
    with args.suspicious.open("w", encoding="utf-8", newline="\n") as output:
        for item in suspicious_records:
            output.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            output.write("\n")

    report_lines = [
        "诗词 JSONL 验证报告 / Poetry JSONL validation report",
        "=" * 55,
        f"源文件 / Source: {args.source}",
        f"可疑记录 / Suspicious output: {args.suspicious}",
        f"源文件 SHA-256 / Source SHA-256: {sha256(args.source)}",
        "",
        "结构验证 / Structural validation",
        "-" * 36,
        f"合法结构记录 / Structurally valid records: {valid_records:,}",
        f"JSON 错误 / JSON errors: {json_errors:,}",
        f"模式错误 / Schema errors: {schema_errors:,}",
        f"重复 ID / Duplicate IDs: {duplicate_ids:,}",
        f"重复来源编号 / Duplicate source blocks: {duplicate_source_blocks:,}",
        "",
        "内容筛查 / Content screening",
        "-" * 31,
        f"可疑记录总数 / Unique suspicious records: {len(suspicious_records):,}",
        f"正文最短长度 / Minimum body length: {min(body_lengths) if body_lengths else 0:,}",
        f"正文最长长度 / Maximum body length: {max(body_lengths) if body_lengths else 0:,}",
        "",
        "原因统计 / Reason counts",
        "-" * 26,
    ]
    for reason, count in reason_counts.most_common():
        report_lines.append(f"{reason}: {count:,}")

    report_lines.extend(
        [
            "",
            f"可疑文件 SHA-256 / Suspicious SHA-256: {sha256(args.suspicious)}",
            "",
            "结论 / Conclusion:",
            "结构错误属于阻断性问题；内容标记只是候选问题，不会自动删除记录。",
            "Structural errors are blocking issues. Content flags are review candidates",
            "and do not automatically remove records.",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))

    if json_errors or schema_errors or duplicate_ids or duplicate_source_blocks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
