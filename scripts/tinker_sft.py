#!/usr/bin/env python3
"""
Tinker SFT Training for BoxingGym

Supervised fine-tuning on BoxingGym experiment traces using Tinker LoRA.

Usage:
    uv run python scripts/tinker_sft.py                              # full training
    uv run python scripts/tinker_sft.py --smoke-test                 # 128 examples, 1 epoch
    uv run python scripts/tinker_sft.py --data data/tinker_sft.jsonl # custom data
    uv run python scripts/tinker_sft.py --model "Qwen/Qwen3-30B-A3B" # different model

Requires TINKER_API_KEY in .env and training data from prepare_tinker_data.py.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_tinker_api_key():
    """Verify Tinker API key is configured."""
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        print("Error: TINKER_API_KEY not set in environment")
        print("Add to .env: TINKER_API_KEY=your_key_here")
        print("Get API key from: https://tinker-console.thinkingmachines.ai/")
        sys.exit(1)
    return api_key


def prepare_smoke_test_data(data_path: Path, output_path: Path, num_examples: int = 128):
    """Create a small subset of training data for smoke testing."""
    with open(data_path) as f:
        lines = f.readlines()

    # take first N examples (default 128 to ensure at least 1-2 batches with batch_size=64)
    subset = lines[:num_examples]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(subset)

    print(f"Created smoke test data: {output_path} ({len(subset)} examples)")
    return output_path


def build_config(args):
    """Build Tinker training configuration."""
    import chz
    from tinker_cookbook.model_info import get_recommended_renderer_name
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised import train
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    model_name = args.model
    renderer_name = get_recommended_renderer_name(model_name)

    # for smoke tests, create a small subset (128 examples for 2 batches at batch_size=64)
    data_path = args.data
    if args.smoke_test:
        smoke_data_path = args.data.parent / "tinker_sft_smoke.jsonl"
        data_path = prepare_smoke_test_data(args.data, smoke_data_path, num_examples=128)

    print(f"Model: {model_name}")
    print(f"Renderer: {renderer_name}")
    print(f"Data: {data_path}")

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=str(data_path),
    )

    # for smoke tests, use 1 epoch
    num_epochs = 1 if args.smoke_test else args.num_epochs

    config_dict = {
        "model_name": model_name,
        "log_path": str(args.log_path),
        "dataset_builder": dataset_builder,
        "learning_rate": args.learning_rate,
        "lr_schedule": args.lr_schedule,
        "num_epochs": num_epochs,
        "lora_rank": args.lora_rank,
        "eval_every": args.eval_every,
    }

    return chz.Blueprint(train.Config).apply(config_dict)


async def run_training(args):
    """Run the training loop."""
    from tinker_cookbook.supervised import train

    blueprint = build_config(args)
    config = blueprint.make()

    actual_epochs = 1 if args.smoke_test else args.num_epochs

    print("\n" + "=" * 60)
    print("Starting Tinker SFT Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_length}")
    print(f"Epochs: {actual_epochs}" + (" (smoke test)" if args.smoke_test else ""))
    print(f"Log path: {args.log_path}")
    print("=" * 60 + "\n")

    await train.main(config)


def main():
    parser = argparse.ArgumentParser(
        description="Tinker SFT training for BoxingGym",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/tinker_sft.jsonl"),
        help="Path to training data JSONL",
    )

    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("/tmp/boxing-gym-sft"),
        help="Directory for logs and checkpoints",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base model to fine-tune",
    )

    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank (8-64, higher = more capacity)",
    )

    # Training arguments
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (LoRA typically needs 20-100x higher than full FT)",
    )

    parser.add_argument(
        "--lr-schedule",
        type=str,
        choices=["linear", "constant"],
        default="linear",
        help="Learning rate schedule",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="Evaluate every N steps",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test (50 examples, 1 epoch)",
    )

    args = parser.parse_args()

    # Check prerequisites
    check_tinker_api_key()

    if not args.data.exists():
        print(f"Error: Training data not found: {args.data}")
        print("Run: uv run python scripts/prepare_tinker_data.py")
        sys.exit(1)

    # Run training
    asyncio.run(run_training(args))


if __name__ == "__main__":
    main()
