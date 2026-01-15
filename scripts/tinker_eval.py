#!/usr/bin/env python3
"""
Tinker Evaluation for BoxingGym

Evaluate a Tinker checkpoint on BoxingGym environments by sampling predictions
and computing z-scores against ground truth.

Usage:
    # Evaluate SFT checkpoint on all environments
    uv run python scripts/tinker_eval.py \
        --checkpoint "tinker://288631d4-1ad2-5ce9-9548-a95535741497:train:0/weights/final"

    # Evaluate on specific environments
    uv run python scripts/tinker_eval.py \
        --checkpoint "tinker://..." \
        --envs hyperbolic_temporal_discount,dugongs

    # More samples for better statistics
    uv run python scripts/tinker_eval.py \
        --checkpoint "tinker://..." \
        --num-samples 20

Prerequisites:
    1. TINKER_API_KEY in .env
    2. Trained checkpoint from tinker_sft.py or tinker_rl.py
"""

import os
import sys
import asyncio
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

import numpy as np

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from boxing_gym.envs.registry import get_environment_registry

# all BoxingGym environments
ALL_ENVIRONMENTS = [
    "hyperbolic_temporal_discount",
    "death_process",
    "peregrines",
    "morals",
    "lotka_volterra",
    "irt",
    "location_finding",
    "dugongs",
    "survival",
    "emotion",
]


def check_tinker_api_key():
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        print("Error: TINKER_API_KEY not set in environment")
        sys.exit(1)
    return api_key


def load_boxing_gym_env(env_name: str, goal_name: str = "direct"):
    nametoenv, nameenvtogoal = get_environment_registry()

    if env_name not in nametoenv:
        available = list(nametoenv.keys())
        print(f"Error: Unknown environment '{env_name}'")
        print(f"Available: {available}")
        return None

    env_class = nametoenv[env_name]
    env = env_class()

    module = sys.modules[env_class.__module__]
    goal_class_name = "DirectGoal" if goal_name == "direct" else f"{goal_name.title()}Goal"

    if hasattr(module, goal_class_name):
        goal_class = getattr(module, goal_class_name)
        goal = goal_class(env)
    else:
        goal = module.Goal(env) if hasattr(module, "Goal") else None

    return goal


def extract_answer(text: str) -> str:
    """Extract answer from model output, handling various formats."""
    import re

    # try <answer>...</answer> tags first
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()

    # try to find a list pattern like [1, 2] or [1.5, 2.3]
    list_match = re.search(r"\[[\d.,\s]+\]", text)
    if list_match:
        return list_match.group(0)

    # try to find a number
    num_match = re.search(r"[-+]?\d*\.?\d+", text)
    if num_match:
        return num_match.group(0)

    return text


def compute_z_score(prediction: str, ground_truth: str, goal) -> Optional[float]:
    try:
        # extract answer from prediction
        answer = extract_answer(prediction)
        score, _ = goal.evaluate_predictions([answer], [ground_truth])
        norm_mu = getattr(goal, "norm_mu", 0.5)
        norm_sigma = getattr(goal, "norm_sigma", 0.5)

        if norm_sigma > 0:
            z = (score - norm_mu) / norm_sigma
        else:
            z = score

        return z
    except Exception as e:
        print(f"  Warning: Evaluation failed: {e}")
        return None


async def run_evaluation(args):
    import tinker
    from tinker import types
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    print("\n" + "=" * 60)
    print("Tinker Checkpoint Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model}")
    print(f"Environments: {args.envs}")
    print(f"Samples per env: {args.num_samples}")
    print("=" * 60 + "\n")

    service_client = tinker.ServiceClient()

    # get tokenizer and renderer first
    print("Loading tokenizer and renderer...")
    tokenizer = get_tokenizer(args.model)
    renderer_name = model_info.get_recommended_renderer_name(args.model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # load checkpoint for sampling
    print("Loading checkpoint...")
    sampling_client = service_client.create_sampling_client(model_path=args.checkpoint)

    sampling_params = types.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.7,
        stop=renderer.get_stop_sequences(),
    )

    # parse environment list
    env_names = args.envs.split(",") if args.envs else ALL_ENVIRONMENTS

    results: Dict[str, Dict] = {}

    for env_name in env_names:
        env_name = env_name.strip()
        print(f"\n--- Evaluating: {env_name} ---")

        goal = load_boxing_gym_env(env_name)
        if goal is None:
            print(f"  Skipping {env_name}: Could not load goal")
            continue

        z_scores = []

        for i in range(args.num_samples):
            # generate evaluation question
            try:
                question, ground_truth = goal.get_goal_eval_question(include_prior=True)
            except NotImplementedError:
                print(f"  Skipping {env_name}: get_goal_eval_question not implemented")
                break

            # build prompt
            convo = [{"role": "user", "content": question}]
            model_input = renderer.build_generation_prompt(convo)

            # sample from model
            result = sampling_client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
            ).result()

            # parse response
            if result.sequences:
                seq = result.sequences[0]
                parsed, success = renderer.parse_response(seq.tokens)
                if success:
                    prediction = parsed.get("content", "")
                else:
                    prediction = tokenizer.decode(seq.tokens)
            else:
                prediction = ""

            # compute z-score
            z = compute_z_score(prediction, ground_truth, goal)
            if z is not None:
                z_scores.append(z)
                print(f"  Sample {i+1}/{args.num_samples}: z={z:.3f}")

        if z_scores:
            results[env_name] = {
                "mean_z": float(np.mean(z_scores)),
                "std_z": float(np.std(z_scores)),
                "min_z": float(np.min(z_scores)),
                "max_z": float(np.max(z_scores)),
                "n": len(z_scores),
            }
            print(f"  Mean z-score: {results[env_name]['mean_z']:.3f} (std: {results[env_name]['std_z']:.3f})")
        else:
            print(f"  No valid samples for {env_name}")

    # compute aggregate
    all_means = [r["mean_z"] for r in results.values()]
    aggregate = {
        "mean_z": float(np.mean(all_means)) if all_means else 0.0,
        "std_z": float(np.std(all_means)) if all_means else 0.0,
        "n_envs": len(results),
    }

    # compile output
    output = {
        "checkpoint": args.checkpoint,
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "num_samples_per_env": args.num_samples,
        "results": results,
        "aggregate": aggregate,
    }

    # print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Environment':<35} {'Mean z':<10} {'Std z':<10} {'n':<5}")
    print("-" * 60)
    for env, r in sorted(results.items()):
        print(f"{env:<35} {r['mean_z']:<10.3f} {r['std_z']:<10.3f} {r['n']:<5}")
    print("-" * 60)
    print(f"{'AGGREGATE':<35} {aggregate['mean_z']:<10.3f} {aggregate['std_z']:<10.3f} {aggregate['n_envs']:<5}")
    print("=" * 60)

    # save output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Tinker checkpoint on BoxingGym environments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Tinker checkpoint path (tinker://...)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base model name (for tokenizer)",
    )

    parser.add_argument(
        "--envs",
        type=str,
        default=None,
        help="Comma-separated environment names (default: all)",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples per environment",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum generation tokens",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )

    args = parser.parse_args()

    check_tinker_api_key()
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
