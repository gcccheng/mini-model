#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poetry_llm.runtime import choose_device, load_model_from_checkpoint, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a poetry checkpoint on packed test tokens")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    model.eval()
    data = np.memmap(args.tokens, dtype=np.uint16, mode="r")
    block = model.config.block_size
    losses = []
    with torch.inference_mode():
        for _ in range(args.batches):
            starts = np.random.randint(0, len(data) - block - 1, size=args.batch_size)
            x_np = np.stack([np.asarray(data[i : i + block], dtype=np.int64) for i in starts])
            y_np = np.stack([np.asarray(data[i + 1 : i + block + 1], dtype=np.int64) for i in starts])
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            _, loss = model(x, y)
            losses.append(float(loss))
    mean_loss = sum(losses) / len(losses)
    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint["step"],
        "device": str(device),
        "test_batches": args.batches,
        "batch_size": args.batch_size,
        "block_size": block,
        "test_loss": mean_loss,
        "perplexity": math.exp(mean_loss),
    }
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
