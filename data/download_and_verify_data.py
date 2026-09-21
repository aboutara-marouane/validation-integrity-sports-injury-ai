#!/usr/bin/env python3
"""Verify manually downloaded source data against the release manifest."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("source_manifest.csv")


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> int:
    failures = 0
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        for item in csv.DictReader(stream):
            target = ROOT / item["local_path"]
            if not target.is_file():
                print(f"MISSING  {item['local_path']}\n         Obtain: {item['source_doi']}")
                failures += 1
                continue
            actual = digest(target, "sha256")
            if actual != item["sha256"]:
                print(f"MISMATCH {item['local_path']}\n         expected SHA-256: {item['sha256']}\n         actual SHA-256:   {actual}")
                failures += 1
                continue
            print(f"OK       {item['local_path']}")
    if failures:
        print(f"\nVerification failed for {failures} file(s).")
        return 1
    print("\nAll source files match the recorded provenance manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
