#!/usr/bin/env python3
"""
Test question diversity for BoxingGym environments.

Generates N questions and reports uniqueness statistics.
"""

import sys
import hashlib
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from boxing_gym.envs.registry import get_environment_registry


def hash_question(question: str) -> str:
    """Hash a question for uniqueness tracking."""
    return hashlib.md5(question.encode()).hexdigest()[:8]


def test_diversity(env_name: str, n_samples: int = 100, goal_name: str = "direct"):
    """Test question generation diversity."""
    nametoenv, _ = get_environment_registry()

    if env_name not in nametoenv:
        print(f"Error: Unknown environment '{env_name}'")
        print(f"Available: {list(nametoenv.keys())}")
        sys.exit(1)

    # Create environment and goal
    env_class = nametoenv[env_name]
    env = env_class()
    module = sys.modules[env_class.__module__]

    goal_class_name = "DirectGoal" if goal_name == "direct" else f"{goal_name.title()}Goal"
    if hasattr(module, goal_class_name):
        goal_class = getattr(module, goal_class_name)
        goal = goal_class(env)
    else:
        goal = module.Goal(env) if hasattr(module, "Goal") else None

    if goal is None:
        print(f"Error: Could not create goal for {env_name}")
        sys.exit(1)

    print(f"Testing diversity for: {env_name}")
    print(f"Generating {n_samples} questions...\n")

    questions = []
    hashes = []

    for i in range(n_samples):
        q, gt = goal.get_goal_eval_question(include_prior=True)
        questions.append(q)
        hashes.append(hash_question(q))

        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1}/{n_samples}...")

    # Analyze uniqueness
    unique_hashes = set(hashes)
    hash_counts = Counter(hashes)

    print(f"\n{'='*60}")
    print("DIVERSITY REPORT")
    print(f"{'='*60}")
    print(f"Total questions generated: {n_samples}")
    print(f"Unique questions: {len(unique_hashes)}")
    print(f"Uniqueness rate: {len(unique_hashes) / n_samples * 100:.1f}%")
    print(f"\nDuplicate statistics:")
    print(f"  Questions appearing once: {sum(1 for c in hash_counts.values() if c == 1)}")
    print(f"  Questions appearing 2+ times: {sum(1 for c in hash_counts.values() if c > 1)}")
    print(f"  Max repetitions: {max(hash_counts.values())}")

    # Show most repeated questions
    most_common = hash_counts.most_common(5)
    print(f"\nMost repeated question hashes:")
    for hash_val, count in most_common:
        print(f"  {hash_val}: {count} times")
        # Find one example question with this hash
        for q, h in zip(questions, hashes):
            if h == hash_val:
                preview = q[:100] + "..." if len(q) > 100 else q
                print(f"    Example: {preview}")
                break

    # Assessment
    print(f"\n{'='*60}")
    print("ASSESSMENT")
    print(f"{'='*60}")

    uniqueness_pct = len(unique_hashes) / n_samples * 100

    if uniqueness_pct >= 90:
        print("✅ EXCELLENT: High diversity, suitable for training")
    elif uniqueness_pct >= 70:
        print("⚠️ MODERATE: Some repetition, may limit training")
    elif uniqueness_pct >= 50:
        print("❌ LOW: Significant repetition, training will overfit")
    else:
        print("🚨 CRITICAL: Severe repetition, generator collapsed")

    print(f"\nFor 30-batch training (batch_size=32):")
    print(f"  Total questions needed: 960")
    print(f"  With {uniqueness_pct:.0f}% uniqueness: ~{int(960 * uniqueness_pct / 100)} unique")
    print(f"  Average repetitions per question: {960 / len(unique_hashes):.1f}x")

    return len(unique_hashes), n_samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test question diversity")
    parser.add_argument("--env", default="lotka_volterra", help="Environment name")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of questions to generate")
    parser.add_argument("--goal", default="direct", help="Goal type")

    args = parser.parse_args()

    test_diversity(args.env, args.n_samples, args.goal)
