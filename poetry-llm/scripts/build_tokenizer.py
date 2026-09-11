#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poetry_llm.tokenizer import CharTokenizer, SPECIAL_TOKENS, format_record
from poetry_llm.runtime import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a character tokenizer from training JSONL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    records = 0
    for line in Path(args.input).open(encoding="utf-8"):
        record = json.loads(line)
        counts.update(format_record(record))
        records += 1
    characters = sorted(counts)
    tokenizer = CharTokenizer(SPECIAL_TOKENS + characters)
    tokenizer.save(
        args.output,
        {
            "training_records": records,
            "training_source_sha256": sha256(args.input),
            "character_count": sum(counts.values()),
        },
    )
    rare = sum(1 for value in counts.values() if value == 1)
    report = f"""字符级 Tokenizer 报告 / Character tokenizer report
=================================================
训练文件 / Training source: {args.input}
训练记录 / Training records: {records:,}
训练字符 / Training characters: {sum(counts.values()):,}
普通字符 / Regular characters: {len(characters):,}
特殊 token / Special tokens: {len(SPECIAL_TOKENS)}
词表大小 / Vocabulary size: {tokenizer.vocab_size:,}
仅出现一次的字符 / Characters occurring once: {rare:,}
未知字符策略 / Unknown-character policy: <unk>
Tokenizer SHA-256: {sha256(args.output)}

词表只使用训练集构建，验证集和测试集不会影响词表。
The vocabulary is built from the training split only; validation and test do not influence it.
"""
    Path(args.report).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
