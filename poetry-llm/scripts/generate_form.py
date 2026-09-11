#!/usr/bin/env python3
from __future__ import annotations

import argparse
import secrets
import sys
import unicodedata
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poetry_llm.runtime import choose_device, load_model_from_checkpoint, seed_everything
from poetry_llm.tokenizer import CharTokenizer


FORM_SPECS = {
    "五言绝句": (4, 5),
    "七言绝句": (4, 7),
    "五言律诗": (8, 5),
    "七言律诗": (8, 7),
}


def is_han_character(token: str) -> bool:
    if len(token) != 1:
        return False
    name = unicodedata.name(token, "")
    return "CJK UNIFIED IDEOGRAPH" in name or "CJK COMPATIBILITY IDEOGRAPH" in name


@torch.inference_mode()
def sample_character(model, ids, allowed_ids, temperature, top_k, repetition_penalty):
    context = ids[:, -model.config.block_size :]
    logits, _ = model(context)
    logits = logits[:, -1, :] / temperature
    if repetition_penalty > 1:
        for token_id in set(ids[0].tolist()):
            value = logits[0, token_id]
            logits[0, token_id] = value / repetition_penalty if value > 0 else value * repetition_penalty
    mask = torch.full_like(logits, -float("inf"))
    mask[:, allowed_ids] = logits[:, allowed_ids]
    logits = mask
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, len(allowed_ids)))
        logits[logits < values[:, [-1]]] = -float("inf")
    return torch.multinomial(F.softmax(logits, dim=-1), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate poetry with exact form constraints")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--form", choices=FORM_SPECS, default="七言绝句")
    parser.add_argument("--dynasty", default="唐朝")
    parser.add_argument("--author", default="佚名")
    parser.add_argument("--title", default="月夜")
    parser.add_argument("--temperature", type=float, default=0.70)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.20)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional reproducible seed; omitted means a fresh random seed each run",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    args = parser.parse_args()
    device = choose_device(args.device)
    tokenizer = CharTokenizer.load(args.tokenizer)
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    model.eval()
    allowed_ids = [index for index, token in enumerate(tokenizer.tokens) if is_han_character(token)]
    comma_id = tokenizer.token_to_id["，"]
    stop_id = tokenizer.token_to_id["。"]
    line_count, width = FORM_SPECS[args.form]
    prompt = f"体裁：{args.form}\n朝代：{args.dynasty}\n作者：{args.author}\n题目：{args.title}\n正文："
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    if tokenizer.unk_id in prompt_ids:
        raise ValueError("prompt contains a character absent from the tokenizer")
    step = checkpoint.get("step", checkpoint.get("training_step", "unknown"))
    base_seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    output_lines = [
        f"格律生成 / Form-constrained generation (step {step}, device {device}, seed {base_seed})",
        "=" * 72,
    ]
    for sample_index in range(args.samples):
        seed_everything(base_seed + sample_index)
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        poem_lines = []
        for line_index in range(line_count):
            chars = []
            for _ in range(width):
                next_id = sample_character(
                    model, ids, allowed_ids, args.temperature, args.top_k, args.repetition_penalty
                )
                ids = torch.cat((ids, next_id), dim=1)
                chars.append(tokenizer.decode(next_id[0].tolist()))
            punctuation_id = comma_id if line_index % 2 == 0 else stop_id
            punctuation = "，" if line_index % 2 == 0 else "。"
            ids = torch.cat(
                (ids, torch.tensor([[punctuation_id]], dtype=torch.long, device=device)), dim=1
            )
            poem_lines.append("".join(chars) + punctuation)
        output_lines.extend([f"\n--- sample {sample_index + 1} ---", prompt + "".join(poem_lines)])
    rendered = "\n".join(output_lines) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
