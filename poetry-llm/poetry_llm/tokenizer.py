from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def format_record(record: dict) -> str:
    """Turn one JSONL record into the exact text seen by the model."""
    form_prefix = f"体裁：{record['form']}\n" if record.get("form") else ""
    return form_prefix + (
        f"朝代：{record['dynasty']}\n"
        f"作者：{record['author']}\n"
        f"题目：{record['title']}\n"
        f"正文：{record['text']}"
    )


class CharTokenizer:
    """A deterministic UTF-8 character tokenizer with four special tokens."""

    def __init__(self, tokens: Iterable[str]):
        tokens = list(tokens)
        if tokens[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise ValueError("token list must start with the standard special tokens")
        if len(tokens) != len(set(tokens)):
            raise ValueError("token list contains duplicates")
        self.tokens = tokens
        self.token_to_id = {token: idx for idx, token in enumerate(tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.token_to_id.get(char, self.unk_id) for char in text]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        pieces: list[str] = []
        special = set(SPECIAL_TOKENS)
        for idx in ids:
            token = self.tokens[int(idx)]
            if skip_special and token in special:
                continue
            pieces.append(token)
        return "".join(pieces)

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        payload = {
            "type": "character",
            "version": 1,
            "special_tokens": SPECIAL_TOKENS,
            "tokens": self.tokens,
            "metadata": metadata or {},
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["tokens"])
