#!/usr/bin/env python
"""CLI for managing quarantined BoxingGym result files.

Usage:
    uv run python scripts/manage_quarantine.py --list
    uv run python scripts/manage_quarantine.py --list --layer L0_SCHEMA
    uv run python scripts/manage_quarantine.py --restore <filename> --to results/death_process/
    uv run python scripts/manage_quarantine.py --audit
    uv run python scripts/manage_quarantine.py --stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from boxing_gym.data_quality.config import ValidationLevel
from boxing_gym.data_quality.quarantine import QuarantineManager


def main():
    parser = argparse.ArgumentParser(description="Manage quarantined result files")
    parser.add_argument("--list", action="store_true", help="List quarantined files")
    parser.add_argument(
        "--layer", type=str, default=None, help="Filter by layer (L0_SCHEMA, L1_NON_FINITE, etc.)"
    )
    parser.add_argument(
        "--restore", type=str, default=None, help="Filename to restore from quarantine"
    )
    parser.add_argument(
        "--to", type=str, default=None, help="Directory to restore file to (used with --restore)"
    )
    parser.add_argument("--audit", action="store_true", help="Show audit log")
    parser.add_argument("--stats", action="store_true", help="Show quarantine statistics")
    parser.add_argument("--quarantine-dir", type=str, default=".quarantine")
    parser.add_argument("--results-root", type=str, default="results/")
    args = parser.parse_args()

    qm = QuarantineManager(args.quarantine_dir)

    if args.list:
        layer = None
        if args.layer:
            try:
                layer = ValidationLevel[args.layer]
            except KeyError:
                print(f"error: unknown layer '{args.layer}'", file=sys.stderr)
                print(f"valid layers: {[l.name for l in ValidationLevel]}", file=sys.stderr)
                sys.exit(1)

        files = qm.list_quarantined(layer)
        if not files:
            print("no quarantined files" + (f" for layer {args.layer}" if args.layer else ""))
            return

        for f in files:
            rel = f.relative_to(qm.root)
            print(f"  {rel}")
        print(f"\ntotal: {len(files)} files")

    elif args.restore:
        if not args.to:
            print("error: --to is required with --restore", file=sys.stderr)
            sys.exit(1)

        # find the file in quarantine
        all_files = qm.list_quarantined()
        matches = [f for f in all_files if f.name == args.restore]

        if not matches:
            print(f"error: '{args.restore}' not found in quarantine", file=sys.stderr)
            sys.exit(1)

        if len(matches) > 1:
            print(f"multiple matches for '{args.restore}':")
            for m in matches:
                print(f"  {m}")
            print("specify the full path to disambiguate")
            sys.exit(1)

        dest_dir = Path(args.to)
        results_root = Path(args.results_root)
        restored = qm.restore(matches[0], dest_dir, results_root=results_root)
        print(f"restored: {restored}")

    elif args.audit:
        entries = qm.get_audit_log()
        if not entries:
            print("no audit entries")
            return

        for entry in entries:
            ts = entry.get("timestamp", "?")[:19]
            action = entry.get("action", "?")
            source = Path(entry.get("source", "?")).name
            layer = entry.get("layer", "")
            reason = entry.get("reason", "")
            print(f"[{ts}] {action:12s} {source:50s} {layer:20s} {reason[:60]}")

        print(f"\ntotal: {len(entries)} audit entries")

    elif args.stats:
        entries = qm.get_audit_log()
        if not entries:
            print("no quarantine data")
            return

        by_layer = {}
        by_action = {}
        for e in entries:
            layer = e.get("layer", "unknown")
            action = e.get("action", "unknown")
            by_layer[layer] = by_layer.get(layer, 0) + 1
            by_action[action] = by_action.get(action, 0) + 1

        print("by layer:")
        for layer, count in sorted(by_layer.items()):
            print(f"  {layer}: {count}")

        print("\nby action:")
        for action, count in sorted(by_action.items()):
            print(f"  {action}: {count}")

        # count files on disk (rglob to find files in timestamp subdirs)
        all_subdirs = list(QuarantineManager.SUBDIRS.values()) + ["other"]
        for subdir in all_subdirs:
            d = qm.root / subdir
            if d.exists():
                n = len(list(d.rglob("*.json")))
                if n > 0:
                    print(f"\n{subdir}/: {n} files")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
