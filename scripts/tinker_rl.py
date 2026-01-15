#!/usr/bin/env python3
"""
Tinker RL Training for BoxingGym (GRPO-style)

Reinforcement learning fine-tuning using BoxingGym evaluation as reward.
Uses Group Relative Policy Optimization with importance sampling.

Usage:
    # Start from SFT checkpoint
    uv run python scripts/tinker_rl.py \
        --checkpoint /tmp/boxing-gym-sft/checkpoint-final

    # Smoke test
    uv run python scripts/tinker_rl.py --max-batches 2 --group-size 4

    # Custom environment
    uv run python scripts/tinker_rl.py --env hyperbolic_temporal_discount

Prerequisites:
    1. TINKER_API_KEY in .env
    2. SFT model trained: uv run python scripts/tinker_sft.py
"""

import os
import sys
import asyncio
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional
from dotenv import load_dotenv

import numpy as np

# Load environment variables
load_dotenv()

# Import BoxingGym components
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from boxing_gym.envs.registry import get_environment_registry


def check_tinker_api_key():
    """Verify Tinker API key is configured."""
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        print("Error: TINKER_API_KEY not set in environment")
        sys.exit(1)
    return api_key


def load_boxing_gym_env(env_name: str, goal_name: str = "direct"):
    """Load a BoxingGym environment and goal."""
    nametoenv, nameenvtogoal = get_environment_registry()

    if env_name not in nametoenv:
        available = list(nametoenv.keys())
        print(f"Error: Unknown environment '{env_name}'")
        print(f"Available: {available}")
        sys.exit(1)

    # first create the environment instance
    env_class = nametoenv[env_name]
    env = env_class()

    # get the environment module
    module = sys.modules[env_class.__module__]

    # look for goal class and instantiate with env
    goal_class_name = "DirectGoal" if goal_name == "direct" else f"{goal_name.title()}Goal"
    if hasattr(module, goal_class_name):
        goal_class = getattr(module, goal_class_name)
        goal = goal_class(env)
    else:
        goal = module.Goal(env) if hasattr(module, "Goal") else None

    if goal is None:
        print(f"Error: Could not create goal for {env_name}")
        sys.exit(1)

    return goal


def compute_reward(
    prediction: str,
    ground_truth: str,
    goal,
) -> float:
    """
    Compute reward from BoxingGym evaluation.

    Lower z-score = better, so we negate for reward.
    """
    try:
        # Evaluate prediction
        score, _ = goal.evaluate_predictions([prediction], [ground_truth])

        # Normalize to z-score
        norm_mu = getattr(goal, "norm_mu", 0.5)
        norm_sigma = getattr(goal, "norm_sigma", 0.5)

        if norm_sigma > 0:
            z = (score - norm_mu) / norm_sigma
        else:
            z = score

        # Negate: lower z = better = higher reward
        return -z
    except Exception as e:
        print(f"Warning: Reward computation failed: {e}")
        return 0.0


async def run_rl_training(args):
    """Run GRPO-style RL training loop."""
    import tinker
    from tinker import types
    from tinker.types.tensor_data import TensorData
    import torch
    from tinker_cookbook import model_info, renderers

    print("\n" + "=" * 60)
    print("Starting Tinker RL Training (GRPO)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Environment: {args.env}")
    print(f"Batch size: {args.batch_size}")
    print(f"Group size: {args.group_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max batches: {args.max_batches or 'unlimited'}")
    print("=" * 60 + "\n")

    # Initialize Tinker clients
    service_client = tinker.ServiceClient()

    # Load checkpoint or create fresh LoRA
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        training_client = service_client.create_lora_training_client(
            base_model=args.model,
            rank=args.lora_rank,
        )
        training_client.load_state(args.checkpoint)
    else:
        print("Starting from base model (no checkpoint)")
        training_client = service_client.create_lora_training_client(
            base_model=args.model,
            rank=args.lora_rank,
        )

    tokenizer = training_client.get_tokenizer()
    renderer_name = model_info.get_recommended_renderer_name(args.model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Sampling parameters
    sampling_params = types.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.7,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = types.AdamParams(learning_rate=args.learning_rate)

    # Load BoxingGym environment
    goal = load_boxing_gym_env(args.env)
    print(f"Loaded environment: {args.env}")

    # Generate evaluation questions
    eval_questions = []
    for _ in range(args.batch_size):
        q, gt = goal.get_goal_eval_question(include_prior=True)
        eval_questions.append((q, gt))

    print(f"Generated {len(eval_questions)} evaluation questions")

    # Training loop
    log_path = Path(args.log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    metrics_file = log_path / "metrics.jsonl"

    batch_idx = 0
    while args.max_batches is None or batch_idx < args.max_batches:
        print(f"\n--- Batch {batch_idx} ---")

        # Save weights for sampling
        sampling_path = training_client.save_weights_for_sampler(
            name=f"{batch_idx:06d}"
        ).result().path
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)

        datums = []
        all_rewards = []

        for question, ground_truth in eval_questions[:args.batch_size]:
            # Build prompt
            convo = [{"role": "user", "content": question}]
            model_input = renderer.build_generation_prompt(convo)
            prompt_tokens = model_input.to_ints()

            # Sample group_size rollouts
            result = sampling_client.sample(
                prompt=model_input,
                num_samples=args.group_size,
                sampling_params=sampling_params,
            ).result()

            # Compute rewards
            rewards = []
            for seq in result.sequences:
                parsed, success = renderer.parse_response(seq.tokens)
                if success:
                    prediction = parsed.get("content", "")
                else:
                    prediction = tokenizer.decode(seq.tokens)

                reward = compute_reward(prediction, ground_truth, goal)
                rewards.append(reward)

            all_rewards.extend(rewards)

            # GRPO-style advantage centering
            mean_reward = sum(rewards) / len(rewards) if rewards else 0
            advantages = [r - mean_reward for r in rewards]

            # Skip if all advantages are zero (no variance)
            if all(a == 0 for a in advantages):
                continue

            # Create training data
            for seq, advantage in zip(result.sequences, advantages):
                tokens = prompt_tokens + list(seq.tokens)
                ob_len = len(prompt_tokens) - 1

                # Build logprobs tensor
                logprobs_list = [0.0] * ob_len + list(seq.logprobs)
                advantages_list = [0.0] * ob_len + [advantage] * (len(tokens) - 1 - ob_len)

                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(
                            torch.tensor(tokens[1:], dtype=torch.long)
                        ),
                        "logprobs": TensorData.from_torch(
                            torch.tensor(logprobs_list, dtype=torch.float32)
                        ),
                        "advantages": TensorData.from_torch(
                            torch.tensor(advantages_list, dtype=torch.float32)
                        ),
                    },
                )
                datums.append(datum)

        if not datums:
            print("Warning: No training data this batch, skipping")
            batch_idx += 1
            continue

        # Training step
        print(f"Training on {len(datums)} examples...")
        fwd_bwd = training_client.forward_backward(datums, loss_fn="importance_sampling")
        optim = training_client.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd.result()
        optim.result()

        # Compute metrics
        mean_reward = np.mean(all_rewards) if all_rewards else 0
        std_reward = np.std(all_rewards) if all_rewards else 0

        metrics = {
            "batch_idx": batch_idx,
            "num_examples": len(datums),
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "min_reward": float(min(all_rewards)) if all_rewards else 0,
            "max_reward": float(max(all_rewards)) if all_rewards else 0,
        }

        print(f"Mean reward: {mean_reward:.4f} (std: {std_reward:.4f})")

        # Log metrics
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # Save checkpoint periodically
        if batch_idx > 0 and batch_idx % args.save_every == 0:
            checkpoint_path = training_client.save_state(
                name=f"checkpoint-{batch_idx:06d}"
            ).result().path
            print(f"Saved checkpoint: {checkpoint_path}")

        batch_idx += 1

    # Save final checkpoint
    final_path = training_client.save_state(name="checkpoint-final").result().path
    print(f"\nTraining complete! Final checkpoint: {final_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Tinker RL training for BoxingGym (GRPO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to SFT checkpoint to start from",
    )

    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("/tmp/boxing-gym-rl"),
        help="Directory for logs and checkpoints",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base model for RL",
    )

    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (smaller for RL)",
    )

    # Environment arguments
    parser.add_argument(
        "--env",
        type=str,
        default="hyperbolic_temporal_discount",
        help="BoxingGym environment name",
    )

    # Training arguments
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=4e-5,
        help="Learning rate (lower for RL)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of problems per batch",
    )

    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="Number of rollouts per problem (for variance reduction)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum generation tokens",
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum training batches (for smoke tests)",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N batches",
    )

    args = parser.parse_args()

    # Check prerequisites
    check_tinker_api_key()

    # Run training
    asyncio.run(run_rl_training(args))


if __name__ == "__main__":
    main()
