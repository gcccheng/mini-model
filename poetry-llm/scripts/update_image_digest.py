#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pin the Kubernetes image to a GHCR digest")
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--digest", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.digest):
        raise SystemExit("invalid sha256 digest")
    path = Path(args.deployment)
    text = path.read_text(encoding="utf-8")
    replacement = f"          image: {args.image}@{args.digest}"
    updated, count = re.subn(r"^          image: .+$", replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit("expected exactly one container image line")
    path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
