#!/usr/bin/env python3
"""
Tinker RL Training for BoxingGym (GRPO-style)

RL fine-tuning using BoxingGym evaluation as reward. Uses GRPO with importance
sampling. Generates fresh questions each batch to prevent memorization, tracks
question uniqueness via hashing, and uses held-out validation for early stopping.

Usage:
    uv run python scripts/tinker_rl.py --checkpoint /tmp/boxing-gym-sft/checkpoint-final
    uv run python scripts/tinker_rl.py --max-batches 5 --group-size 4  # smoke test

Requires TINKER_API_KEY in .env and an SFT checkpoint from tinker_sft.py.
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

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


def hash_question(question: str) -> str:
    """Hash a question for uniqueness tracking."""
    return hashlib.md5(question.encode()).hexdigest()[:8]


def evaluate_on_held_out(
    questions: list[tuple[str, str]],
    sampling_client,
    goal,
    renderer,
    tokenizer,
    sampling_params,
    num_samples: int = 8,
) -> tuple[float, float]:
    """
    Evaluate model on held-out validation questions.

    Args:
        num_samples: Number of samples per question for variance reduction (default: 8, matches training group_size)

    Returns:
        (mean_reward, mean_z_score)
    """
    rewards = []
    z_scores = []

    for question, ground_truth in questions:
        # Build prompt
        convo = [{"role": "user", "content": question}]
        model_input = renderer.build_generation_prompt(convo)

        # Sample num_samples times to match training's group_size (variance reduction)
        result = sampling_client.sample(
            prompt=model_input,
            num_samples=num_samples,
            sampling_params=sampling_params,
        ).result()

        # Average over all samples (same as training)
        question_rewards = []
        for seq in result.sequences:
            parsed, success = renderer.parse_response(seq.tokens)
            if success:
                prediction = parsed.get("content", "")
            else:
                prediction = tokenizer.decode(seq.tokens)

            # Compute reward
            reward = compute_reward(prediction, ground_truth, goal)
            question_rewards.append(reward)

        # Use mean reward across 8 samples
        mean_reward = np.mean(question_rewards)
        rewards.append(mean_reward)
        z_scores.append(-mean_reward)

    return np.mean(rewards), np.mean(z_scores)


async def run_rl_training(args):
    """Run GRPO-style RL training loop with fresh questions each batch."""
    import tinker
    import torch
    from tinker import types
    from tinker.types.tensor_data import TensorData
    from tinker_cookbook import model_info, renderers

    print("\n" + "=" * 60)
    print("Starting Tinker RL Training (GRPO with Fresh Questions)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Environment: {args.env}")
    print(f"Batch size: {args.batch_size}")
    print(f"Group size: {args.group_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max batches: {args.max_batches or 'unlimited'}")
    print(f"Validation size: {args.val_size}")
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

    # validation set generated once, before training loop
    print(f"\nGenerating validation set ({args.val_size} questions)...")
    val_questions = []
    for _ in range(args.val_size):
        q, gt = goal.get_goal_eval_question(include_prior=True)
        val_questions.append((q, gt))
    print(f"Validation set created: {len(val_questions)} questions")

    # Track question hashes globally to verify no reuse
    global_question_hashes = set()

    # Training loop setup
    log_path = Path(args.log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    metrics_file = log_path / "metrics.jsonl"

    best_val_reward = float("-inf")
    patience_counter = 0

    batch_idx = 0
    while args.max_batches is None or batch_idx < args.max_batches:
        print(f"\n{'=' * 60}")
        print(f"Batch {batch_idx}")
        print(f"{'=' * 60}")

        # fresh questions each batch to prevent memorization
        train_questions = []
        batch_hashes = []
        for _ in range(args.batch_size):
            q, gt = goal.get_goal_eval_question(include_prior=True)
            train_questions.append((q, gt))
            qhash = hash_question(q)
            batch_hashes.append(qhash)
            global_question_hashes.add(qhash)

        unique_in_batch = len(set(batch_hashes))
        uniqueness_rate = unique_in_batch / args.batch_size
        print(
            f"Question generation: {unique_in_batch}/{args.batch_size} unique in batch ({uniqueness_rate * 100:.0f}%)"
        )
        print(f"Global question pool: {len(global_question_hashes)} unique questions seen")

        # warn if question diversity is low
        if uniqueness_rate < 0.80:
            print(f"WARNING: Low question diversity ({uniqueness_rate * 100:.0f}%) - may overfit")
            if uniqueness_rate < 0.50:
                print("CRITICAL: Severe question collapse - consider stopping")

        # Save weights for sampling
        sampling_path = (
            training_client.save_weights_for_sampler(name=f"{batch_idx:06d}").result().path
        )
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)

        datums = []
        all_rewards = []

        for question, ground_truth in train_questions:
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

        # Compute training metrics
        mean_train_reward = np.mean(all_rewards) if all_rewards else 0
        std_train_reward = np.std(all_rewards) if all_rewards else 0
        mean_train_z = -mean_train_reward  # z-score is negative of reward

        print(
            f"Training - Mean reward: {mean_train_reward:.4f} (z: {mean_train_z:.4f}, std: {std_train_reward:.4f})"
        )

        metrics = {
            "batch_idx": batch_idx,
            "num_examples": len(datums),
            "train_reward": float(mean_train_reward),
            "train_z_score": float(mean_train_z),
            "train_std": float(std_train_reward),
            "unique_questions_in_batch": unique_in_batch,
            "global_unique_questions": len(global_question_hashes),
        }

        if batch_idx % args.val_every == 0:
            print(f"\nRunning validation on {len(val_questions)} held-out questions...")
            val_reward, val_z = evaluate_on_held_out(
                val_questions,
                sampling_client,
                goal,
                renderer,
                tokenizer,
                sampling_params,
                num_samples=args.group_size,
            )
            print(f"Validation - Mean reward: {val_reward:.4f} (z: {val_z:.4f})")

            metrics["val_reward"] = float(val_reward)
            metrics["val_z_score"] = float(val_z)

            # Early stopping logic
            if val_reward > best_val_reward:
                best_val_reward = val_reward
                patience_counter = 0
                print(f"New best: {val_reward:.4f}")

                # Save best checkpoint
                best_checkpoint_path = (
                    training_client.save_state(name=f"checkpoint-best-{batch_idx:06d}")
                    .result()
                    .path
                )
                print(f"Saved best checkpoint: {best_checkpoint_path}")
            else:
                patience_counter += 1
                print(f"Validation not improving (patience: {patience_counter}/{args.patience})")

                if patience_counter >= args.patience:
                    print(f"\nEarly stopping after {args.patience} checks without improvement")
                    break

        # Log metrics
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # Save checkpoint periodically
        if batch_idx > 0 and batch_idx % args.save_every == 0:
            checkpoint_path = (
                training_client.save_state(name=f"checkpoint-{batch_idx:06d}").result().path
            )
            print(f"Saved checkpoint: {checkpoint_path}")

        batch_idx += 1

    # Save final checkpoint
    final_path = training_client.save_state(name="checkpoint-final").result().path
    print(f"\nTraining complete! Final checkpoint: {final_path}")
    print(f"Best validation reward: {best_val_reward:.4f}")
    print(f"Total unique questions generated: {len(global_question_hashes)}")


def main():
    parser = argparse.ArgumentParser(
        description="Tinker RL training for BoxingGym (GRPO with fresh questions)",
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
        default="lotka_volterra",
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

    # NEW: Validation arguments
    parser.add_argument(
        "--val-size",
        type=int,
        default=50,
        help="Number of held-out validation questions",
    )

    parser.add_argument(
        "--val-every",
        type=int,
        default=5,
        help="Run validation every N batches",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience (validation checks)",
    )

    args = parser.parse_args()

    # Check prerequisites
    check_tinker_api_key()

    # Run training
    asyncio.run(run_rl_training(args))


if __name__ == "__main__":
    main()
