#!/usr/bin/env python3
"""
Prepare Tinker Training Data from BoxingGym Results

Converts BoxingGym experiment results (JSON) into Tinker-compatible JSONL format.
Extracts from best-performing models per environment based on sweep results.

Usage:
    # Extract from VPS results (best models per env)
    uv run python scripts/prepare_tinker_data.py \
        --results-dir results/ \
        --output data/tinker_sft.jsonl

    # Dry run to inspect format
    uv run python scripts/prepare_tinker_data.py --results-dir results/ --dry-run

    # Filter by specific model
    uv run python scripts/prepare_tinker_data.py --results-dir results/ --model "deepseek"

    # Filter by environment
    uv run python scripts/prepare_tinker_data.py --results-dir results/ --env hyperbolic

Best models per environment (from 1,546 valid runs):
    hyperbolic_temporal_discount: GLM-4.7 (z=-1.065)
    death_process: GPT-5.1-Codex-mini (z=-1.030)
    peregrines: MiniMax-M2.1 (z=-0.662)
    morals: Kimi-for-coding (z=-0.615)
    lotka_volterra: GPT-5.1-Codex-mini (z=-0.449)
    irt: Qwen3-32B (z=-0.360)
    location_finding: Qwen3-32B (z=-0.144)
    dugongs: GPT-5.1-Codex-mini (z=-0.095)
    survival: DeepSeek (z=-0.092)
    emotion: GPT-4o (z=+0.781)
"""

import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Best models per environment from sweep analysis (budget≥10, |z|<100)
BEST_MODELS_PER_ENV = {
    "hyperbolic_temporal_discount": ["glm-4.7", "glm-4.6"],
    "death_process": ["gpt-5.1-codex-mini", "codex-mini"],
    "peregrines": ["minimax-m2.1", "minimax"],
    "morals": ["kimi-for-coding", "kimi"],
    "lotka_volterra": ["gpt-5.1-codex-mini", "codex-mini"],
    "irt": ["qwen3-32b", "qwen3"],
    "location_finding": ["qwen3-32b", "qwen3"],
    "dugongs": ["gpt-5.1-codex-mini", "codex-mini"],
    "survival": ["deepseek", "deepseek-chat"],
    "emotion": ["gpt-4o"],
}


def model_matches(model_name: str, patterns: List[str]) -> bool:
    """Check if model name matches any of the patterns (case-insensitive)."""
    model_lower = model_name.lower()
    return any(p.lower() in model_lower for p in patterns)


def extract_env_from_filename(filename: str) -> Optional[str]:
    """Extract environment name from result filename."""
    match = re.search(r"env=([a-z_]+)", filename)
    return match.group(1) if match else None


def extract_model_from_config(config: Dict[str, Any]) -> str:
    """Extract model name from config."""
    llms = config.get("llms", {})
    if isinstance(llms, dict):
        return llms.get("model_name", "unknown")
    return str(llms)


def parse_scientist_messages(messages: List[str]) -> List[Dict[str, str]]:
    """Parse scientist_messages strings into role/content dicts."""
    parsed = []
    for msg in messages:
        # Format: "role:X, message:Y"
        match = re.match(r"role:(\w+),\s*message:(.*)", msg, re.DOTALL)
        if match:
            role = match.group(1).strip()
            content = match.group(2).strip()
            parsed.append({"role": role, "content": content})
    return parsed


def extract_conversations_from_result(
    result_data: Dict[str, Any],
    env_name: str,
) -> List[Dict[str, Any]]:
    """
    Extract training conversations from a BoxingGym result JSON.

    Returns list of Tinker-format messages:
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    """
    conversations = []

    # Get scientist_messages (new format)
    scientist_messages = result_data.get("scientist_messages", [])
    if scientist_messages:
        parsed_msgs = parse_scientist_messages(scientist_messages)
    else:
        parsed_msgs = []

    # Extract system message
    system_msg = ""
    for msg in parsed_msgs:
        if msg["role"] == "system":
            system_msg = msg["content"]
            break

    if not system_msg:
        return []

    # Find user-assistant pairs with thought/observe tags
    i = 0
    while i < len(parsed_msgs):
        msg = parsed_msgs[i]

        if msg["role"] == "assistant":
            content = msg["content"]

            # Parse thought and observe tags
            thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
            observe_match = re.search(r"<observe>(.*?)</observe>", content, re.DOTALL)

            if thought_match and observe_match:
                thought = thought_match.group(1).strip()
                observe = observe_match.group(1).strip()

                # Skip smoke test examples
                if "offline smoke test" in thought.lower():
                    i += 1
                    continue

                # Build Tinker format (ChatML-compatible)
                messages = [
                    {"role": "system", "content": system_msg},
                    {
                        "role": "user",
                        "content": "Think about where to observe next. Articulate your strategy for choosing measurements in <thought>.\nProvide a new measurement point in the format:\n<thought>your thought</thought>\n<observe>your observation</observe>",
                    },
                    {
                        "role": "assistant",
                        "content": f"<thought>{thought}</thought>\n<observe>{observe}</observe>",
                    },
                ]

                conversations.append({"messages": messages, "env": env_name})

        i += 1

    return conversations


def process_results_directory(
    results_dir: Path,
    model_filter: Optional[str] = None,
    env_filter: Optional[str] = None,
    use_best_per_env: bool = True,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Process all JSON files in the results directory.

    Args:
        results_dir: Directory containing BoxingGym result JSON files
        model_filter: Only include results from models matching this pattern
        env_filter: Only include results from environments matching this pattern
        use_best_per_env: If True, only use best model per environment
        limit: Maximum number of examples to extract

    Returns:
        List of all training examples
    """
    all_conversations = []
    stats = defaultdict(lambda: defaultdict(int))

    # Find all JSON files recursively
    json_files = list(results_dir.rglob("*.json"))
    print(f"Found {len(json_files)} result files in {results_dir}")

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                result_data = json.load(f)

            # Extract metadata
            config = result_data.get("config", {})
            model_name = extract_model_from_config(config)
            env_name = extract_env_from_filename(json_file.name)

            if not env_name:
                continue

            # Apply environment filter
            if env_filter and env_filter.lower() not in env_name.lower():
                continue

            # Apply model filter
            if model_filter and model_filter.lower() not in model_name.lower():
                continue

            # Apply best-per-env filter
            if use_best_per_env:
                best_patterns = BEST_MODELS_PER_ENV.get(env_name, [])
                if best_patterns and not model_matches(model_name, best_patterns):
                    continue

            # Extract conversations
            conversations = extract_conversations_from_result(result_data, env_name)
            all_conversations.extend(conversations)
            stats[env_name][model_name] += len(conversations)

            if limit and len(all_conversations) >= limit:
                all_conversations = all_conversations[:limit]
                break

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    # Print stats
    print("\nExtraction stats:")
    for env, models in sorted(stats.items()):
        total = sum(models.values())
        print(f"  {env}: {total} examples")
        for model, count in sorted(models.items(), key=lambda x: -x[1]):
            print(f"    - {model}: {count}")

    return all_conversations


def write_jsonl(conversations: List[Dict[str, Any]], output_path: Path):
    """Write conversations to JSONL file (Tinker format)."""
    # Remove env field before writing (used only for stats)
    clean_conversations = [{"messages": c["messages"]} for c in conversations]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for conv in clean_conversations:
            f.write(json.dumps(conv) + "\n")

    print(f"\nWrote {len(clean_conversations)} training examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Tinker training data from BoxingGym results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/"),
        help="Directory containing BoxingGym result JSON files",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tinker_sft.jsonl"),
        help="Output JSONL file path",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter by model name (e.g., 'deepseek', 'gpt-4o')",
    )

    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Filter by environment name (e.g., 'hyperbolic', 'dugongs')",
    )

    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Use all models, not just best per environment",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of training examples to extract",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sample output without writing file",
    )

    args = parser.parse_args()

    # Validate results directory
    if not args.results_dir.exists():
        print(f"Error: Results directory does not exist: {args.results_dir}")
        return 1

    # Process results
    conversations = process_results_directory(
        args.results_dir,
        model_filter=args.model,
        env_filter=args.env,
        use_best_per_env=not args.all_models,
        limit=args.limit,
    )

    if len(conversations) == 0:
        print("\nWarning: No training examples extracted")
        print("Try --all-models to include all models, not just best per environment")
        return 1

    # Dry run: print sample
    if args.dry_run:
        print("\n=== Sample Training Example ===")
        print(json.dumps(conversations[0], indent=2))
        print(f"\nTotal examples: {len(conversations)}")
        return 0

    # Write output
    write_jsonl(conversations, args.output)

    print(f"\n✅ Success! Tinker training data ready at: {args.output}")
    print("\nNext steps:")
    print("  1. Set TINKER_API_KEY in .env")
    print("  2. Run: uv run python scripts/tinker_sft.py")

    return 0


if __name__ == "__main__":
    exit(main())
