#!/usr/bin/env python3
"""Create a high-confidence ancient-poetry corpus with full audit trails."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


ANCIENT_DYNASTIES = {
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
}

BODY_REVIEW_MARKERS = (
    "相关成语",
    "相关楹联",
    "词句解析",
    "词句注释",
    "英文翻译",
    "白话译文",
    "创作背景",
    "作品鉴赏",
    "更多关于",
    "注释",
    "赏析",
    "译文",
)

TITLE_REVIEW_MARKERS = (
    "日记",
    "游记",
    "全文讲解",
    "白话译文",
    "英文翻译",
    "作品赏析",
    "课文解析",
)

HTML_PATTERN = re.compile(r"<[^>\n]{1,200}>")
URL_PATTERN = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
ASCII_ALNUM_PATTERN = re.compile(r"[A-Za-z0-9]")
MISSING_CHARACTER_PATTERN = re.compile(r"[□＿]")
CORRUPTION_SYMBOL_PATTERN = re.compile(r"[¤＋＜─■○●〓囗]")
ASCII_PUNCTUATION_PATTERN = re.compile(r'''[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]''')
PROSE_TITLE_SUFFIX_PATTERN = re.compile(r"(?:表|策|论|传|记|序|疏)$")
PARENTHETICAL_CITATION_PATTERN = re.compile(r"[（(]《[^》]{1,100}》")
SOURCE_EXPLANATION_PATTERN = re.compile(r"出自[^。！？\n]{0,40}《[^》]{1,80}》")
CLAUSE_SPLIT_PATTERN = re.compile(r"[，。！？；：]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_latin_letter(character: str) -> bool:
    return unicodedata.category(character).startswith("L") and "LATIN" in unicodedata.name(
        character, ""
    )


def classify(record: dict[str, Any]) -> tuple[str, list[str]]:
    title = str(record["title"])
    dynasty = str(record["dynasty"])
    author = str(record["author"])
    body = str(record["text"])
    hard_reasons: list[str] = []
    review_reasons: list[str] = []

    if not dynasty:
        hard_reasons.append("unknown_dynasty")
    elif dynasty in {"近代", "现代"}:
        hard_reasons.append("non_ancient_dynasty")
    elif dynasty not in ANCIENT_DYNASTIES:
        hard_reasons.append("unsupported_dynasty")

    body_length = len(body)
    if body_length < 8:
        hard_reasons.append("body_shorter_than_8")
    if body_length > 5_000:
        hard_reasons.append("body_longer_than_5000")
    elif body_length > 1_000:
        review_reasons.append("body_longer_than_1000")

    if URL_PATTERN.search(body):
        hard_reasons.append("contains_url")
    if ASCII_ALNUM_PATTERN.search(body):
        hard_reasons.append("contains_ascii_alphanumeric")
    if ASCII_ALNUM_PATTERN.search(title + author + dynasty):
        hard_reasons.append("contains_ascii_alphanumeric_metadata")
    if MISSING_CHARACTER_PATTERN.search(title + author + dynasty + body):
        hard_reasons.append("contains_missing_character_placeholder")
    if CORRUPTION_SYMBOL_PATTERN.search(title + author + dynasty + body):
        hard_reasons.append("contains_corruption_symbol")

    nonspace_length = sum(not character.isspace() for character in body)
    latin_count = sum(is_latin_letter(character) for character in body)
    if nonspace_length >= 20 and latin_count / nonspace_length > 0.25:
        hard_reasons.append("latin_heavy_body")
    elif sum(is_latin_letter(character) for character in body[:80]) >= 15:
        hard_reasons.append("latin_heavy_prefix")

    for marker in BODY_REVIEW_MARKERS:
        if marker in body:
            review_reasons.append(f"contains_marker:{marker}")
    if SOURCE_EXPLANATION_PATTERN.search(body):
        review_reasons.append("contains_source_explanation")
    for marker in TITLE_REVIEW_MARKERS:
        if marker in title:
            review_reasons.append(f"title_marker:{marker}")
    if HTML_PATTERN.search(body):
        review_reasons.append("possible_html_or_character_placeholder")
    if "_" in title:
        review_reasons.append("underscore_joined_title_metadata")
    if PARENTHETICAL_CITATION_PATTERN.search(body):
        review_reasons.append("contains_parenthetical_citation")
    if ASCII_PUNCTUATION_PATTERN.search(body):
        review_reasons.append("contains_ascii_punctuation")

    clauses = [
        len(clause.strip())
        for clause in CLAUSE_SPLIT_PATTERN.split(body)
        if clause.strip()
    ]
    prose_like_clauses = False
    if len(body) >= 80 and len(clauses) >= 6:
        mean_clause_length = sum(clauses) / len(clauses)
        short_clause_ratio = sum(length <= 8 for length in clauses) / len(clauses)
        if mean_clause_length > 10 and short_clause_ratio < 0.35:
            prose_like_clauses = True
            review_reasons.append("prose_like_clause_lengths")
    prose_signals = prose_like_clauses or any(
        marker in body for marker in ("曰：", "曰:", "“", "\u3000\u3000")
    )
    if PROSE_TITLE_SUFFIX_PATTERN.search(title) and prose_signals:
        review_reasons.append("prose_genre_title_with_prose_signals")

    if hard_reasons:
        return "quarantine", hard_reasons + review_reasons
    if review_reasons:
        return "review", review_reasons
    return "clean", []


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
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    clean: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    input_count = 0

    with args.source.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            input_count += 1
            status, reasons = classify(record)
            status_counts[status] += 1
            reason_counts.update(reasons)

            if status == "clean":
                clean.append(record)
                continue

            audit_item = {
                "line_number": line_number,
                "status": status,
                "reasons": reasons,
                "record": record,
            }
            if status == "quarantine":
                quarantine.append(audit_item)
            else:
                review.append(audit_item)

    write_jsonl(args.destination, clean)
    write_jsonl(args.quarantine, quarantine)
    write_jsonl(args.review, review)

    report_lines = [
        "古诗内容清理报告 / Ancient-poetry content-cleaning report",
        "=" * 62,
        f"源文件 / Source: {args.source}",
        f"干净数据 / Clean output: {args.destination}",
        f"隔离数据 / Quarantine: {args.quarantine}",
        f"人工复核 / Manual review: {args.review}",
        f"源文件 SHA-256 / Source SHA-256: {sha256_file(args.source)}",
        "",
        "分类结果 / Classification results",
        "-" * 40,
        f"输入记录 / Input records: {input_count:,}",
        f"干净记录 / Clean records: {len(clean):,}",
        f"人工复核 / Review records: {len(review):,}",
        f"隔离记录 / Quarantined records: {len(quarantine):,}",
        f"记录守恒 / Accounted records: {len(clean) + len(review) + len(quarantine):,}",
        "",
        "原因统计 / Reason counts",
        "-" * 28,
    ]
    for reason, count in reason_counts.most_common():
        report_lines.append(f"{reason}: {count:,}")

    report_lines.extend(
        [
            "",
            f"干净文件 SHA-256 / Clean SHA-256: {sha256_file(args.destination)}",
            f"隔离文件 SHA-256 / Quarantine SHA-256: {sha256_file(args.quarantine)}",
            f"复核文件 SHA-256 / Review SHA-256: {sha256_file(args.review)}",
            "",
            "策略 / Policy:",
            "仅明确标注为先秦至清朝、正文不少于 8 字且没有强污染信号的记录",
            "可直接进入干净集。1000–5000 字长篇及含解释性标记的记录进入人工复核。",
            "近代、现代、朝代未知、超过 5000 字、含网址或大量拉丁字符的记录被隔离。",
            "Only records explicitly dated from Pre-Qin through Qing, with at least eight",
            "body characters and no strong contamination signal, can enter the clean set.",
            "Works of 1,000–5,000 characters and records with explanatory markers require",
            "manual review. Modern, unknown-dynasty, >5,000-character, URL-containing,",
            "or Latin-heavy records are quarantined.",
            "原记录不会被改写；所有排除记录均保留在审计文件中。",
            "Source records are never rewritten; every excluded record remains auditable.",
        ]
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))

    if input_count != len(clean) + len(review) + len(quarantine):
        raise RuntimeError("Record accounting failed")


if __name__ == "__main__":
    main()
