#!/usr/bin/env python3
"""
Prepare MLX Training Data from BoxingGym Results

Converts BoxingGym experiment results (JSON) into MLX-compatible JSONL format.
Each line contains a single training example with the full conversation context.

Usage:
    uv run python scripts/prepare_mlx_training_data.py \
        --results_dir results/microfluidics \
        --output mlx_training_data.jsonl \
        --mode poisson

Date: 2025-11-14
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import re


def extract_conversations_from_result(result_data: Dict[str, Any], mode: str) -> List[Dict[str, str]]:
    """
    Extract training conversations from a single BoxingGym result JSON.

    Args:
        result_data: Parsed JSON data from a BoxingGym result file
        mode: Environment mode (poisson/stretch) for context

    Returns:
        List of training examples, each with 'text' field
    """
    conversations = []

    # Get system message (contains environment context)
    system_msg = result_data.get('system_message', '')

    # Get experiment history
    history = result_data.get('history', [])

    # Build conversational context
    for i, entry in enumerate(history):
        # Extract agent's thought process and observation
        agent_msg = entry.get('agent_message', '')
        env_response = entry.get('environment_response', '')

        # Parse thought and observe tags
        thought_match = re.search(r'<thought>(.*?)</thought>', agent_msg, re.DOTALL)
        observe_match = re.search(r'<observe>(.*?)</observe>', agent_msg, re.DOTALL)

        if not (thought_match and observe_match):
            continue

        thought = thought_match.group(1).strip()
        observe = observe_match.group(1).strip()

        # Build training example
        # Format: System context + Human query → Assistant reasoning + action
        conversation_text = f"""Human: {system_msg}

Based on your current understanding of the system, propose the next experiment.
Assistant: <thought>{thought}</thought>
<observe>{observe}</observe>"""

        conversations.append({'text': conversation_text})

    return conversations


def process_results_directory(results_dir: Path, mode: str, limit: int = None) -> List[Dict[str, str]]:
    """
    Process all JSON files in the results directory.

    Args:
        results_dir: Directory containing BoxingGym result JSON files
        mode: Environment mode (poisson/stretch)
        limit: Maximum number of examples to extract (None = all)

    Returns:
        List of all training examples
    """
    all_conversations = []

    # Find all JSON files
    json_files = list(results_dir.glob('*.json'))

    print(f'Found {len(json_files)} result files in {results_dir}')

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                result_data = json.load(f)

            conversations = extract_conversations_from_result(result_data, mode)
            all_conversations.extend(conversations)

            if limit and len(all_conversations) >= limit:
                all_conversations = all_conversations[:limit]
                break

        except Exception as e:
            print(f'Error processing {json_file}: {e}')
            continue

    return all_conversations


def write_jsonl(conversations: List[Dict[str, str]], output_path: Path):
    """
    Write conversations to JSONL file (MLX format).

    Args:
        conversations: List of training examples
        output_path: Output JSONL file path
    """
    with open(output_path, 'w') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')

    print(f'Wrote {len(conversations)} training examples to {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Prepare MLX training data from BoxingGym results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--results_dir',
        type=Path,
        default=Path('results/microfluidics'),
        help='Directory containing BoxingGym result JSON files'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('mlx_training_data.jsonl'),
        help='Output JSONL file path'
    )

    parser.add_argument(
        '--mode',
        choices=['poisson', 'stretch', 'both'],
        default='poisson',
        help='Environment mode filter'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of training examples to extract'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print sample output without writing file'
    )

    args = parser.parse_args()

    # Validate results directory
    if not args.results_dir.exists():
        print(f'Error: Results directory does not exist: {args.results_dir}')
        return 1

    # Process results
    conversations = process_results_directory(args.results_dir, args.mode, args.limit)

    if len(conversations) == 0:
        print('Warning: No training examples extracted')
        return 1

    # Dry run: print sample
    if args.dry_run:
        print('\n=== Sample Training Example ===')
        print(json.dumps(conversations[0], indent=2))
        print(f'\nTotal examples: {len(conversations)}')
        return 0

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(conversations, args.output)

    print(f'\n✅ Success! MLX training data ready at: {args.output}')
    print('\nNext steps:')
    print('  1. Create lora-config.yaml (see docs/QWEN3_UNSLOTH_VLLM_BOXINGGYM.md)')
    print(f'  2. Run: mlx_lm.lora --model Qwen/Qwen3-4B-Instruct-2507 --data {args.output} --train --config lora-config.yaml')

    return 0


if __name__ == '__main__':
    exit(main())
