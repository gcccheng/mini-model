#!/usr/bin/env python3
"""Create deterministic, body-group-aware train/validation/test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLIT_RATIOS = {"train": 0.90, "validation": 0.05, "test": 0.05}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def body_fingerprint(body: str) -> str:
    normalized = re.sub(r"\s+", "", body)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")


def choose_split(
    counts: dict[str, int], targets: dict[str, int], group_size: int
) -> str:
    # Select the split with the largest remaining proportional deficit. This
    # keeps same-body groups intact while staying close to the requested ratios.
    def score(name: str) -> tuple[float, int]:
        target = targets[name]
        remaining = target - counts[name]
        proportional_deficit = remaining / target if target else float("-inf")
        return proportional_deficit, remaining

    candidates = [name for name in SPLIT_RATIOS if counts[name] < targets[name]]
    if not candidates:
        candidates = list(SPLIT_RATIOS)
    return max(candidates, key=score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    with args.source.open(encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[body_fingerprint(str(record["text"]))].append(record)

    grouped_items = sorted(groups.items(), key=lambda item: item[0])
    rng = random.Random(args.seed)
    rng.shuffle(grouped_items)

    total = len(records)
    targets = {
        "train": int(total * SPLIT_RATIOS["train"]),
        "validation": int(total * SPLIT_RATIOS["validation"]),
    }
    targets["test"] = total - targets["train"] - targets["validation"]

    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    assigned_fingerprints: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    counts = {name: 0 for name in splits}

    for fingerprint, group in grouped_items:
        split_name = choose_split(counts, targets, len(group))
        splits[split_name].extend(group)
        assigned_fingerprints[split_name].add(fingerprint)
        counts[split_name] += len(group)

    output_paths: dict[str, Path] = {}
    for name, split_records in splits.items():
        rng.shuffle(split_records)
        output_path = args.output_directory / f"{name}.jsonl"
        write_jsonl(output_path, split_records)
        output_paths[name] = output_path

    all_ids = {record["id"] for record in records}
    split_ids = {
        name: {record["id"] for record in split_records}
        for name, split_records in splits.items()
    }
    assert split_ids["train"].isdisjoint(split_ids["validation"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["validation"].isdisjoint(split_ids["test"])
    assert set().union(*split_ids.values()) == all_ids
    assert assigned_fingerprints["train"].isdisjoint(
        assigned_fingerprints["validation"]
    )
    assert assigned_fingerprints["train"].isdisjoint(assigned_fingerprints["test"])
    assert assigned_fingerprints["validation"].isdisjoint(
        assigned_fingerprints["test"]
    )

    report_lines = [
        "数据集划分报告 / Dataset split report",
        "=" * 43,
        f"源文件 / Source: {args.source}",
        f"输出目录 / Output directory: {args.output_directory}",
        f"随机种子 / Random seed: {args.seed}",
        f"源文件 SHA-256 / Source SHA-256: {sha256_file(args.source)}",
        "",
        "划分规则 / Split policy",
        "-" * 29,
        "训练/验证/测试比例 / Train/validation/test ratios: 90% / 5% / 5%",
        "相同正文指纹始终进入同一划分，避免跨集合泄漏。",
        "Identical normalized-body fingerprints always stay in the same split.",
        f"正文指纹组数 / Body-fingerprint groups: {len(groups):,}",
        f"多记录正文组 / Multi-record body groups: {sum(len(g) > 1 for g in groups.values()):,}",
        "",
        "划分结果 / Split results",
        "-" * 27,
    ]

    for name in ("train", "validation", "test"):
        split_records = splits[name]
        characters = sum(len(str(record["text"])) for record in split_records)
        dynasty_counts = Counter(str(record["dynasty"]) for record in split_records)
        report_lines.extend(
            [
                f"{name}:",
                f"  目标记录 / Target records: {targets[name]:,}",
                f"  实际记录 / Actual records: {len(split_records):,}",
                f"  实际比例 / Actual ratio: {len(split_records) / total:.4%}",
                f"  正文字符 / Body characters: {characters:,}",
                f"  正文组 / Body groups: {len(assigned_fingerprints[name]):,}",
                f"  朝代种类 / Dynasty values: {len(dynasty_counts):,}",
                f"  SHA-256: {sha256_file(output_paths[name])}",
            ]
        )

    report_lines.extend(
        [
            "",
            f"总记录 / Total records: {sum(len(value) for value in splits.values()):,}",
            "ID 集合互斥 / ID sets disjoint: True",
            "正文指纹集合互斥 / Body-fingerprint sets disjoint: True",
            "记录守恒 / Record accounting passed: True",
        ]
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
