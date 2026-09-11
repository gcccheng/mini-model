#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path


PUNCTUATION = re.compile(r"[，。！？]")
FORMS = {
    (4, 5): "五言绝句",
    (4, 7): "七言绝句",
    (8, 5): "五言律诗",
    (8, 7): "七言律诗",
}


def is_han_character(char: str) -> bool:
    name = unicodedata.name(char, "")
    return "CJK UNIFIED IDEOGRAPH" in name or "CJK COMPATIBILITY IDEOGRAPH" in name


def classify(text: str) -> str | None:
    """Return a form only for a strict alternating comma/full-stop structure."""
    parts = PUNCTUATION.split(text)
    if not parts or parts[-1] != "":
        return None
    clauses = parts[:-1]
    marks = PUNCTUATION.findall(text)
    if len(clauses) not in (4, 8) or len(marks) != len(clauses):
        return None
    expected_marks = ["，" if index % 2 == 0 else None for index in range(len(clauses))]
    for actual, expected in zip(marks, expected_marks):
        if expected == "，" and actual != "，":
            return None
        if expected is None and actual not in "。！？":
            return None
    widths = {len(clause) for clause in clauses}
    if len(widths) != 1:
        return None
    width = next(iter(widths))
    if (len(clauses), width) not in FORMS:
        return None
    if not all(clause and all(is_han_character(char) for char in clause) for clause in clauses):
        return None
    return FORMS[(len(clauses), width)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create strict labeled classical-poetry splits")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    source_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_counts: Counter[str] = Counter()
    lines = ["严格体裁标注报告 / Strict poetic-form labeling report", "=" * 62]
    for split in ("train", "validation", "test"):
        counts: Counter[str] = Counter()
        total = kept = 0
        with (source_dir / f"{split}.jsonl").open(encoding="utf-8") as src, (
            output_dir / f"{split}.jsonl"
        ).open("w", encoding="utf-8") as dst:
            for line in src:
                total += 1
                record = json.loads(line)
                form = classify(record["text"])
                if form is None:
                    continue
                record["form"] = form
                dst.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts[form] += 1
                kept += 1
        all_counts.update(counts)
        lines.append(f"{split}: kept={kept:,}/{total:,}; " + ", ".join(f"{k}={counts[k]:,}" for k in FORMS.values()))
    lines.extend(
        [
            "",
            "合计 / Total: " + f"{sum(all_counts.values()):,}",
            ", ".join(f"{key}={all_counts[key]:,}" for key in FORMS.values()),
            "",
            "仅保留严格的四句/八句、逐句五字/七字、逗号句号交替的记录。",
            "Only strict 4/8-line records with uniform 5/7-character clauses and alternating punctuation are retained.",
        ]
    )
    report = "\n".join(lines) + "\n"
    Path(args.report).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
