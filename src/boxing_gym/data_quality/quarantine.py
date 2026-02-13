"""Quarantine management for corrupted/invalid result files.

Moves files to .quarantine/ with audit trail instead of deleting.
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ValidationLevel


class QuarantineManager:
    """Manages quarantined result files with audit logging."""

    SUBDIRS = {
        ValidationLevel.L0_SCHEMA: "L0_schema",
        ValidationLevel.L1_NON_FINITE: "L1_non_finite",
        ValidationLevel.L1_5_CONSISTENCY: "L1_5_consistency",
        ValidationLevel.L2_HARD_INVARIANTS: "L2_hard_invariants",
    }

    def __init__(self, quarantine_dir: str = ".quarantine"):
        self.root = Path(quarantine_dir)
        self._audit_path = self.root / "audit_log.jsonl"

    def setup(self) -> None:
        """Create quarantine directory structure."""
        self.root.mkdir(exist_ok=True)
        for subdir in self.SUBDIRS.values():
            (self.root / subdir).mkdir(exist_ok=True)

    def quarantine(
        self,
        source: Path,
        layer: ValidationLevel,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> Path:
        """Move a file to quarantine and log the decision.

        Returns the new path of the quarantined file.
        """
        self.setup()
        subdir = self.SUBDIRS.get(layer, "other")
        dest_dir = self.root / subdir
        dest_dir.mkdir(exist_ok=True)
        # use timestamp + uuid to avoid race conditions with parallel agents
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        dest = dest_dir / f"{source.stem}_{ts}_{uuid.uuid4().hex[:8]}{source.suffix}"

        shutil.move(str(source), str(dest))

        self._log_audit(
            {
                "action": "quarantine",
                "source": str(source),
                "destination": str(dest),
                "layer": layer.name,
                "reason": reason,
                "details": details,
            }
        )

        return dest

    def restore(
        self, quarantined_path: Path, original_dir: Path, results_root: Path | None = None
    ) -> Path:
        """Restore a quarantined file to its original directory.

        Args:
            quarantined_path: Path to the quarantined file.
            original_dir: Target directory to restore into.
            results_root: If provided, original_dir must be under this root
                (prevents path traversal).

        Returns the restored path.
        """
        if results_root is not None:
            if not original_dir.resolve().is_relative_to(results_root.resolve()):
                raise ValueError(
                    f"restore path {original_dir} is outside results root {results_root}"
                )

        dest = original_dir / quarantined_path.name
        if dest.exists():
            raise FileExistsError(f"cannot restore: {dest} already exists")

        original_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(quarantined_path), str(dest))

        self._log_audit(
            {
                "action": "restore",
                "source": str(quarantined_path),
                "destination": str(dest),
            }
        )

        return dest

    def list_quarantined(self, layer: ValidationLevel | None = None) -> list[Path]:
        """List quarantined files, optionally filtered by layer."""
        if not self.root.exists():
            return []

        if layer is not None:
            subdir = self.SUBDIRS.get(layer)
            if subdir is None:
                return []
            target = self.root / subdir
            return sorted(target.rglob("*.json")) if target.exists() else []

        all_subdirs = list(self.SUBDIRS.values()) + ["other"]
        files = []
        for subdir in all_subdirs:
            target = self.root / subdir
            if target.exists():
                files.extend(target.rglob("*.json"))
        return sorted(files)

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Read full audit log."""
        if not self._audit_path.exists():
            return []
        entries = []
        with open(self._audit_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries

    def _log_audit(self, entry: dict[str, Any]) -> None:
        entry["timestamp"] = datetime.now(UTC).isoformat()
        self.root.mkdir(exist_ok=True)
        with open(self._audit_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
