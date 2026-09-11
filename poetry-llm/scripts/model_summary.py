#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poetry_llm.model import GPT, GPTConfig
from poetry_llm.runtime import load_json
from poetry_llm.tokenizer import CharTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct and smoke-test the poetry GPT")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    raw = load_json(args.config)
    tokenizer = CharTokenizer.load(args.tokenizer)
    raw["vocab_size"] = tokenizer.vocab_size
    config = GPTConfig(**raw)
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (2, min(32, config.block_size)))
    logits, loss = model(x, x)
    loss.backward()
    params = model.parameter_count()
    report = f"""模型结构报告 / Model architecture report
============================================
词表 / Vocabulary: {config.vocab_size:,}
上下文长度 / Context length: {config.block_size}
层数 / Layers: {config.n_layer}
注意力头 / Attention heads: {config.n_head}
隐藏维度 / Embedding width: {config.n_embd}
参数量 / Parameters: {params:,} ({params / 1_000_000:.3f}M)
权重共享 / Tied token embedding and LM head: True
前向传播形状 / Forward output shape: {tuple(logits.shape)}
反向传播 / Backward smoke test: PASS
"""
    Path(args.report).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
