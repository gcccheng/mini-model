#!/usr/bin/env python3
"""Generate a read-only diagnostic report for the poetry corpus."""

from __future__ import annotations

import argparse
import hashlib
import unicodedata
from collections import Counter
from pathlib import Path


CONTAMINATION_MARKERS = (
    "相关成语",
    "注释",
    "赏析",
    "译文",
    "英文翻译",
    "词句解析",
    "出自",
    "更多关于",
    "相关楹联",
)


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def visible_character(character: str) -> str:
    labels = {"\n": r"\n", "\r": r"\r", "\t": r"\t", " ": "<SPACE>"}
    return labels.get(character, character)


def build_report(source: Path) -> str:
    raw = source.read_bytes()
    text = raw.decode("utf-8")
    lines = text.splitlines()
    nonempty_lines = [line for line in lines if line.strip()]
    blocks = [block for block in text.split("\n\n") if block.strip()]

    field_counts: Counter[int] = Counter()
    dynasty_counts: Counter[str] = Counter()
    character_counts = Counter(text)
    marker_counts = {marker: text.count(marker) for marker in CONTAMINATION_MARKERS}
    body_lengths: list[int] = []
    malformed_examples: list[tuple[int, int, str]] = []
    longest: list[tuple[int, int, str]] = []

    for block_number, block in enumerate(blocks, start=1):
        fields = [line.strip() for line in block.splitlines() if line.strip()]
        field_counts[len(fields)] += 1

        if len(fields) != 4:
            if len(malformed_examples) < 20:
                malformed_examples.append((block_number, len(fields), fields[0][:80]))
            continue

        title, dynasty, _author, body = fields
        dynasty_counts[dynasty] += 1
        body_lengths.append(len(body))
        longest.append((len(body), block_number, title))

    longest.sort(reverse=True)
    control_counts = Counter(
        character
        for character in text
        if unicodedata.category(character) == "Cc" and character not in "\n\r\t"
    )

    report: list[str] = []
    report.append("Poetry corpus inspection report")
    report.append("=" * 32)
    report.append(f"Source: {source}")
    report.append(f"SHA-256: {hashlib.sha256(raw).hexdigest()}")
    report.append("Encoding: UTF-8 (strict decoding succeeded)")
    report.append(f"Bytes: {len(raw):,}")
    report.append(f"Characters: {len(text):,}")
    report.append(f"Lines: {len(lines):,}")
    report.append(f"Non-empty lines: {len(nonempty_lines):,}")
    report.append(f"Blank lines: {len(lines) - len(nonempty_lines):,}")
    report.append(f"Unique characters: {len(character_counts):,}")
    report.append(f"Unicode replacement characters: {text.count(chr(0xFFFD)):,}")
    report.append(f"Unexpected control characters: {sum(control_counts.values()):,}")

    report.append("\nRecord structure")
    report.append("-" * 16)
    report.append(f"Blank-line-separated blocks: {len(blocks):,}")
    for fields, count in sorted(field_counts.items()):
        report.append(f"Blocks with {fields} non-empty fields: {count:,}")
    report.append(f"Structurally valid four-field blocks: {field_counts[4]:,}")
    report.append(f"Malformed blocks: {len(blocks) - field_counts[4]:,}")

    report.append("\nBody-length statistics (valid four-field blocks)")
    report.append("-" * 48)
    if body_lengths:
        report.append(f"Minimum: {min(body_lengths):,}")
        report.append(f"Mean: {sum(body_lengths) / len(body_lengths):,.2f}")
        report.append(f"Median: {percentile(body_lengths, 0.50):,}")
        report.append(f"90th percentile: {percentile(body_lengths, 0.90):,}")
        report.append(f"95th percentile: {percentile(body_lengths, 0.95):,}")
        report.append(f"99th percentile: {percentile(body_lengths, 0.99):,}")
        report.append(f"Maximum: {max(body_lengths):,}")
        report.append(f"Shorter than 8 characters: {sum(n < 8 for n in body_lengths):,}")
        report.append(f"Longer than 500 characters: {sum(n > 500 for n in body_lengths):,}")
        report.append(f"Longer than 1,000 characters: {sum(n > 1000 for n in body_lengths):,}")
        report.append(f"Longer than 5,000 characters: {sum(n > 5000 for n in body_lengths):,}")

    report.append("\nDynasty-field values")
    report.append("-" * 20)
    for dynasty, count in dynasty_counts.most_common():
        report.append(f"{dynasty or '<EMPTY>'}: {count:,}")

    report.append("\nPotential contamination markers")
    report.append("-" * 31)
    for marker, count in marker_counts.items():
        report.append(f"{marker}: {count:,}")

    report.append("\nTwenty longest valid bodies")
    report.append("-" * 27)
    for length, block_number, title in longest[:20]:
        report.append(f"block={block_number:,} length={length:,} title={title}")

    report.append("\nMalformed-block examples")
    report.append("-" * 24)
    for block_number, fields, title in malformed_examples:
        report.append(f"block={block_number:,} fields={fields} first_line={title}")

    report.append("\nFifty most common characters")
    report.append("-" * 29)
    for character, count in character_counts.most_common(50):
        name = unicodedata.name(character, "UNKNOWN")
        report.append(f"{visible_character(character)!r}: {count:,} ({name})")

    return "\n".join(report) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_report(args.source)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
