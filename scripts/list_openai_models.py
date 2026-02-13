#!/usr/bin/env python3
"""Utility to print the OpenAI models that your API key can access."""

import argparse
import json
import sys
from collections.abc import Iterable

from openai import OpenAI


def _to_dict(model) -> dict:
    """Convert an OpenAI model object into a serialisable dict."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "to_dict"):
        return model.to_dict()
    if isinstance(model, dict):
        return model
    # best effort fallback
    return {attr: getattr(model, attr) for attr in dir(model) if not attr.startswith("_")}


def _iter_models(client: OpenAI, limit: int | None) -> Iterable[dict]:
    """Yield model dicts from the OpenAI SDK, respecting an optional limit."""
    seen = 0
    page = client.models.list()
    for model in page:
        yield _to_dict(model)
        seen += 1
        if limit is not None and seen >= limit:
            return


def _format_rows(models: list[dict], contains: str | None) -> list[dict]:
    filtered = []
    for m in models:
        model_id = m.get("id", "<unknown>")
        if contains and contains.lower() not in model_id.lower():
            continue
        filtered.append(
            {
                "id": model_id,
                "owned_by": m.get("owned_by"),
                "context_window": m.get("context_window"),
                "created": m.get("created"),
            }
        )
    return filtered


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("No models matched your filters.")
        return

    headers = ["id", "owned_by", "created", "context_window"]
    col_widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            value = row.get(header)
            text = "" if value is None else str(value)
            col_widths[header] = max(col_widths[header], len(text))

    def fmt(row: dict) -> str:
        cells = []
        for header in headers:
            value = row.get(header)
            text = "" if value is None else str(value)
            cells.append(text.ljust(col_widths[header]))
        return "  ".join(cells)

    header_line = "  ".join(h.upper().ljust(col_widths[h]) for h in headers)
    print(header_line)
    print("  ".join("-" * col_widths[h] for h in headers))
    for row in rows:
        print(fmt(row))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "List the OpenAI models that are currently available to your API key.\n"
            "Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL / OPENAI_PROJECT) before running."
        )
    )
    parser.add_argument(
        "--contains",
        help="Only display model IDs containing this substring (case-insensitive).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Stop after retrieving N models (default: no limit).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON instead of a formatted table.",
    )
    parser.add_argument(
        "--base-url",
        help="Override the OpenAI base URL (defaults to OPENAI_BASE_URL or the SDK default).",
    )
    parser.add_argument(
        "--project",
        help="Override the OpenAI project ID (defaults to OPENAI_PROJECT).",
    )
    args = parser.parse_args(argv)

    client_kwargs: dict[str, str | None] = {}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    if args.project:
        client_kwargs["project"] = args.project

    try:
        client = OpenAI(**client_kwargs)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Failed to initialise OpenAI client: {exc}", file=sys.stderr)
        return 1

    try:
        models = list(_iter_models(client, args.limit))
    except Exception as exc:
        print(f"Error while listing models: {exc}", file=sys.stderr)
        return 1

    rows = _format_rows(models, args.contains)

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        _print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
