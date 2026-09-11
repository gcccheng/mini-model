#!/usr/bin/env python3
"""Conservatively normalize the poetry corpus without parsing records."""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from collections import Counter
from pathlib import Path


PUNCTUATION_MAP = {
    "﹐": "，",
    "︐": "，",
    "﹒": "。",
    "．": "。",
    "﹔": "；",
    "︔": "；",
    "﹕": "：",
    "︓": "：",
    "﹖": "？",
    "﹗": "！",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def count_blocks(text: str) -> int:
    return len([block for block in text.split("\n\n") if block.strip()])


def normalize(text: str) -> tuple[str, dict[str, object]]:
    stats: dict[str, object] = {}

    stats["crlf_newlines"] = text.count("\r\n")
    without_crlf = text.replace("\r\n", "\n")
    stats["bare_cr_newlines"] = without_crlf.count("\r")
    text = without_crlf.replace("\r", "\n")

    normalized = unicodedata.normalize("NFC", text)
    stats["unicode_nfc_changed"] = normalized != text
    stats["unicode_nfc_character_delta"] = len(normalized) - len(text)
    text = normalized

    stats["bom_removed"] = text.count("\ufeff")
    stats["nul_removed"] = text.count("\x00")
    text = text.replace("\ufeff", "").replace("\x00", "")

    punctuation_counts = Counter({source: text.count(source) for source in PUNCTUATION_MAP})
    text = text.translate(str.maketrans(PUNCTUATION_MAP))
    stats["punctuation_replacements"] = dict(punctuation_counts)

    original_lines = text.split("\n")
    # Preserve whitespace-only lines. The source uses one such line as an empty
    # author-field placeholder, while truly empty lines delimit records.
    whitespace_only_lines = sum(bool(line) and not line.strip() for line in original_lines)
    stripped_lines = [line if line and not line.strip() else line.strip() for line in original_lines]
    stats["whitespace_only_field_lines_preserved"] = whitespace_only_lines
    stats["lines_trimmed"] = sum(a != b for a, b in zip(original_lines, stripped_lines))
    stats["characters_removed_by_trimming"] = sum(
        len(a) - len(b) for a, b in zip(original_lines, stripped_lines)
    )
    text = "\n".join(stripped_lines)

    blank_runs = re.findall(r"\n{3,}", text)
    stats["extra_blank_lines_removed"] = sum(len(run) - 2 for run in blank_runs)
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = text.strip() + "\n"
    return text, stats


def build_report(
    source: Path,
    destination: Path,
    before: str,
    after: str,
    before_bytes: bytes,
    after_bytes: bytes,
    stats: dict[str, object],
) -> str:
    punctuation = stats["punctuation_replacements"]
    assert isinstance(punctuation, dict)

    report = [
        "Poetry corpus normalization report",
        "=" * 34,
        f"Source: {source}",
        f"Destination: {destination}",
        f"Source SHA-256: {sha256(before_bytes)}",
        f"Destination SHA-256: {sha256(after_bytes)}",
        "",
        "Corpus invariants",
        "-" * 17,
        f"Characters before: {len(before):,}",
        f"Characters after: {len(after):,}",
        f"Lines before: {len(before.splitlines()):,}",
        f"Lines after: {len(after.splitlines()):,}",
        f"Blocks before: {count_blocks(before):,}",
        f"Blocks after: {count_blocks(after):,}",
        "",
        "Transformations",
        "-" * 15,
        f"CRLF newlines normalized: {stats['crlf_newlines']:,}",
        f"Bare CR newlines normalized: {stats['bare_cr_newlines']:,}",
        f"Unicode NFC changed text: {stats['unicode_nfc_changed']}",
        f"Unicode NFC character delta: {stats['unicode_nfc_character_delta']:+,}",
        f"BOM characters removed: {stats['bom_removed']:,}",
        f"NUL characters removed: {stats['nul_removed']:,}",
        f"Lines trimmed: {stats['lines_trimmed']:,}",
        f"Characters removed by trimming: {stats['characters_removed_by_trimming']:,}",
        "Whitespace-only field lines preserved: "
        f"{stats['whitespace_only_field_lines_preserved']:,}",
        f"Extra blank lines removed: {stats['extra_blank_lines_removed']:,}",
        "",
        "Punctuation replacements",
        "-" * 24,
    ]

    total_replacements = 0
    for source_character, target_character in PUNCTUATION_MAP.items():
        count = int(punctuation[source_character])
        total_replacements += count
        report.append(f"{source_character!r} -> {target_character!r}: {count:,}")
    report.append(f"Total punctuation replacements: {total_replacements:,}")

    return "\n".join(report) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    before_bytes = args.source.read_bytes()
    before = before_bytes.decode("utf-8")
    after, stats = normalize(before)
    after_bytes = after.encode("utf-8")

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_bytes(after_bytes)

    report = build_report(
        args.source,
        args.destination,
        before,
        after,
        before_bytes,
        after_bytes,
        stats,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
