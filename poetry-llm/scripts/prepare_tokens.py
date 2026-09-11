#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poetry_llm.runtime import sha256
from poetry_llm.tokenizer import CharTokenizer, format_record


def encode_split(source: Path, destination: Path, tokenizer: CharTokenizer) -> dict:
    token_chunks: list[np.ndarray] = []
    records = 0
    unknown = 0
    for line in source.open(encoding="utf-8"):
        record = json.loads(line)
        ids = tokenizer.encode(format_record(record), add_bos=True, add_eos=True)
        unknown += ids.count(tokenizer.unk_id)
        token_chunks.append(np.asarray(ids, dtype=np.uint16))
        records += 1
    tokens = np.concatenate(token_chunks) if token_chunks else np.empty(0, dtype=np.uint16)
    tokens.tofile(destination)
    return {
        "source": str(source),
        "output": str(destination),
        "records": records,
        "tokens": int(tokens.size),
        "unknown_tokens": unknown,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode JSONL splits as packed uint16 token files")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = CharTokenizer.load(args.tokenizer)
    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError("vocabulary is too large for uint16 storage")
    results = {}
    for split in ("train", "validation", "test"):
        results[split] = encode_split(
            data_dir / f"{split}.jsonl", output_dir / f"{split}.bin", tokenizer
        )
    metadata = {
        "dtype": "uint16",
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_sha256": sha256(args.tokenizer),
        "splits": results,
    }
    (output_dir / "meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "Token 数据报告 / Token data report",
        "=====================================",
        f"词表大小 / Vocabulary size: {tokenizer.vocab_size:,}",
        "存储格式 / Storage format: packed uint16",
        "",
    ]
    for split, item in results.items():
        lines.append(
            f"{split}: records={item['records']:,}, tokens={item['tokens']:,}, "
            f"unknown={item['unknown_tokens']:,}, bytes={item['bytes']:,}, sha256={item['sha256']}"
        )
    report = "\n".join(lines) + "\n"
    Path(args.report).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
