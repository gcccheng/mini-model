#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poetry_llm.model import GPT, GPTConfig
from poetry_llm.runtime import choose_device, load_json, seed_everything
from poetry_llm.tokenizer import CharTokenizer


def batch_from_memmap(
    data: np.memmap,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([np.asarray(data[i : i + block_size], dtype=np.int64) for i in starts])
    y = np.stack([np.asarray(data[i + 1 : i + block_size + 1], dtype=np.int64) for i in starts])
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.inference_mode()
def estimate_loss(
    model: GPT,
    data: np.memmap,
    batches: int,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = batch_from_memmap(data, batch_size, block_size, device)
        _, loss = model(x, y)
        losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def cpu_state_dict(model: GPT) -> dict:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def learning_rate(step: int, cfg: dict) -> float:
    if step < cfg["warmup_steps"]:
        return cfg["learning_rate"] * (step + 1) / cfg["warmup_steps"]
    progress = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"])
    coefficient = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return cfg["min_learning_rate"] + coefficient * (
        cfg["learning_rate"] - cfg["min_learning_rate"]
    )


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    model_config: GPTConfig,
    train_config: dict,
    step: int,
    best_validation_loss: float,
    elapsed_seconds: float,
) -> None:
    payload = {
        "model": cpu_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "model_config": model_config.to_dict(),
        "train_config": train_config,
        "step": step,
        "best_validation_loss": best_validation_loss,
        "elapsed_seconds": elapsed_seconds,
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the small poetry GPT from scratch")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokens-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument(
        "--reset-best-validation",
        action="store_true",
        help="Reset best validation loss when resuming on a different dataset",
    )
    parser.add_argument("--max-steps", type=int, help="Override max_steps for a smoke test")
    args = parser.parse_args()

    train_cfg = load_json(args.train_config)
    if args.max_steps is not None:
        train_cfg["max_steps"] = args.max_steps
        train_cfg["warmup_steps"] = min(train_cfg["warmup_steps"], max(1, args.max_steps // 10))
        train_cfg["eval_interval"] = min(train_cfg["eval_interval"], args.max_steps)
        train_cfg["checkpoint_interval"] = min(train_cfg["checkpoint_interval"], args.max_steps)
    seed_everything(train_cfg["seed"])
    device = choose_device(train_cfg["device"])
    print(f"设备 / Device: {device}", flush=True)

    tokenizer = CharTokenizer.load(args.tokenizer)
    model_raw = load_json(args.model_config)
    model_raw["vocab_size"] = tokenizer.vocab_size
    model_cfg = GPTConfig(**model_raw)
    model = GPT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        weight_decay=train_cfg["weight_decay"],
    )
    start_step = 0
    best_val = float("inf")
    prior_elapsed = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        best_val = float(checkpoint["best_validation_loss"])
        prior_elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
        print(f"继续训练 / Resuming after step: {start_step}", flush=True)
        if args.reset_best_validation:
            best_val = float("inf")
            print("重置最佳验证指标 / Reset best validation metric", flush=True)

    tokens_dir = Path(args.tokens_dir)
    train_data = np.memmap(tokens_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(tokens_dir / "validation.bin", dtype=np.uint16, mode="r")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.jsonl"
    started = time.perf_counter()
    model.train()

    for step_index in range(start_step, train_cfg["max_steps"]):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(train_cfg["gradient_accumulation_steps"]):
            x, y = batch_from_memmap(
                train_data, train_cfg["batch_size"], model_cfg.block_size, device
            )
            _, loss = model(x, y)
            (loss / train_cfg["gradient_accumulation_steps"]).backward()
            accumulated_loss += float(loss.detach()) / train_cfg["gradient_accumulation_steps"]
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        lr = learning_rate(step_index, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        completed_step = step_index + 1

        if completed_step % train_cfg["log_interval"] == 0 or completed_step == 1:
            elapsed = prior_elapsed + time.perf_counter() - started
            tokens_seen = (
                completed_step
                * train_cfg["batch_size"]
                * model_cfg.block_size
                * train_cfg["gradient_accumulation_steps"]
            )
            item = {
                "step": completed_step,
                "train_loss": accumulated_loss,
                "learning_rate": lr,
                "elapsed_seconds": elapsed,
                "tokens_seen": tokens_seen,
                "tokens_per_second": tokens_seen / max(elapsed, 1e-9),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(
                f"step {completed_step:4d}/{train_cfg['max_steps']} | "
                f"loss {accumulated_loss:.4f} | lr {lr:.2e} | "
                f"{item['tokens_per_second']:.0f} tok/s",
                flush=True,
            )

        validation_due = completed_step % train_cfg["eval_interval"] == 0
        final_step = completed_step == train_cfg["max_steps"]
        if validation_due or final_step:
            val_loss = estimate_loss(
                model,
                val_data,
                train_cfg["eval_batches"],
                train_cfg["batch_size"],
                model_cfg.block_size,
                device,
            )
            elapsed = prior_elapsed + time.perf_counter() - started
            print(f"验证 / Validation | step {completed_step} | loss {val_loss:.4f}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    model_cfg,
                    train_cfg,
                    completed_step,
                    best_val,
                    elapsed,
                )

        checkpoint_due = completed_step % train_cfg["checkpoint_interval"] == 0
        if checkpoint_due or final_step:
            elapsed = prior_elapsed + time.perf_counter() - started
            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer,
                model_cfg,
                train_cfg,
                completed_step,
                best_val,
                elapsed,
            )

    elapsed = prior_elapsed + time.perf_counter() - started
    summary = {
        "device": str(device),
        "parameters": model.parameter_count(),
        "steps": train_cfg["max_steps"],
        "best_validation_loss": best_val,
        "elapsed_seconds": elapsed,
        "tokens_seen": train_cfg["max_steps"]
        * train_cfg["batch_size"]
        * model_cfg.block_size
        * train_cfg["gradient_accumulation_steps"],
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
