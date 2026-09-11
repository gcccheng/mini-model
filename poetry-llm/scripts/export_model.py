#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poetry_llm.runtime import sha256
from poetry_llm.model import GPT, GPTConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an inference-only poetry model bundle")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    inference = {
        "model": checkpoint["model"],
        "model_config": checkpoint["model_config"],
        "step": checkpoint["step"],
        "training_step": checkpoint["step"],
        "best_validation_loss": checkpoint["best_validation_loss"],
    }
    model_path = output / "model.pt"
    tokenizer_path = output / "tokenizer.json"
    torch.save(inference, model_path)
    shutil.copy2(args.tokenizer, tokenizer_path)
    manifest = {
        "format": "poetry-llm-inference-v1",
        "model_sha256": sha256(model_path),
        "tokenizer_sha256": sha256(tokenizer_path),
        "parameters": model.parameter_count(),
        "training_step": checkpoint["step"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    readme = """# Poetry LLM export

这是仅用于学习的约 10M 参数古诗字符级语言模型。它可能生成错误、重复或虚构内容。
This is an approximately 10M-parameter character-level poetry model for learning.
It can generate incorrect, repetitive, or fabricated text.

Files:
- `model.pt`: inference weights and model configuration
- `tokenizer.json`: character vocabulary
- `manifest.json`: version, hashes, and training metadata
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
