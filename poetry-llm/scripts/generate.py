#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poetry_llm.runtime import choose_device, load_model_from_checkpoint, seed_everything
from poetry_llm.tokenizer import CharTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with the trained poetry GPT")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", default="朝代：唐朝\n作者：李白\n题目：秋夜\n正文：")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    args = parser.parse_args()
    device = choose_device(args.device)
    tokenizer = CharTokenizer.load(args.tokenizer)
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    model.eval()
    prompt_ids = tokenizer.encode(args.prompt, add_bos=True)
    if tokenizer.unk_id in prompt_ids:
        raise ValueError("prompt contains a character absent from the training vocabulary")
    checkpoint_step = checkpoint.get("step", checkpoint.get("training_step", "unknown"))
    rendered = [
        f"生成样本 / Generated samples (checkpoint step {checkpoint_step}, device {device})",
        "=" * 72,
    ]
    for index in range(args.samples):
        seed_everything(args.seed + index)
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        y = model.generate(
            x,
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_id=tokenizer.eos_id,
        )
        text = tokenizer.decode(y[0].tolist())
        rendered.extend([f"\n--- sample {index + 1} ---", text])
    output = "\n".join(rendered) + "\n"
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
