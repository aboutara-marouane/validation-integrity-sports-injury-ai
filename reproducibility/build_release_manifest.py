#!/usr/bin/env python3
"""Create a deterministic SHA-256 manifest for submission artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reproducibility" / "release_manifest.json"
INCLUDE = [
    ROOT / "CITATION.cff",
    ROOT / "LICENSE",
    ROOT / "README.md",
    ROOT / "DATA_GOVERNANCE.md",
    ROOT / ".gitignore",
    ROOT / ".github",
    ROOT / "environment.yml",
    ROOT / "requirements.txt",
    ROOT / "data",
    ROOT / "manuscript",
    ROOT / "experiments" / "outcome_sampling_audit",
    ROOT / "experiments" / "validation_integrity_audit",
    ROOT / "experiments" / "leakage_safe_validation",
    ROOT / "experiments" / "confirmatory_weekly_analysis",
    ROOT / "notebooks" / "outcome_sampling_audit.ipynb",
    ROOT / "notebooks" / "build_provisional_outcome_audit_data.ipynb",
    ROOT / "notebooks" / "confirmatory_weekly_validation.ipynb",
    ROOT / "reproducibility",
]
EXCLUDED_PARTS = {"__pycache__", "smoke_external", "smoke_protocol_ladder", "smoke_synthetic"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


files = []
missing_entries = [entry.relative_to(ROOT) for entry in INCLUDE if not entry.exists()]
if missing_entries:
    missing = "\n".join(f"- {entry}" for entry in missing_entries)
    raise FileNotFoundError(f"Release manifest cannot be built; required path(s) missing:\n{missing}")

for entry in INCLUDE:
    candidates = [entry] if entry.is_file() else entry.rglob("*")
    for path in candidates:
        if (
            not path.is_file() or path == OUTPUT or path.name.endswith(".tar.gz")
            or any(part in EXCLUDED_PARTS for part in path.parts)
        ):
            continue
        files.append({
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        })
payload = {
    "schema": "sports-injury-validation-integrity-release-manifest-v1",
    "file_count": len(files),
    "files": sorted(files, key=lambda item: item["path"]),
}
OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
print(f"Wrote {OUTPUT} with {len(files)} files")
