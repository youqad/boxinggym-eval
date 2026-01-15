#!/usr/bin/env python3
"""
Plot budget progression for completed experiments
Shows how error changes with budget (0 → 5 → 10)

Usage:
    uv run python scripts/plot_budget_progression.py [--root results] [--model MODEL] [--env ENV]
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict


def iter_json_files(root: Path):
    """Recursively find all JSON files in the results directory."""
    for filepath in root.rglob("*.json"):
        yield filepath


def load_results(root: str = "results", model_filter: str = None, env_filter: str = None):
    """Load all result files, optionally filtering by model or environment.
    
    Args:
        root: Results directory path
        model_filter: Optional model name substring to filter by
        env_filter: Optional environment name to filter by
    """
    results_dir = Path(root)
    if not results_dir.exists():
        return defaultdict(lambda: defaultdict(list))

    data = defaultdict(lambda: defaultdict(list))

    for filepath in iter_json_files(results_dir):
        try:
            with open(filepath) as f:
                result = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        # Extract config from JSON content (not filename)
        config = result.get('config', {})
        env = config.get('envs', {}).get('env_name')
        model = config.get('llms', {}).get('model_name', '')
        budgets = config.get('exp', {}).get('num_experiments')

        if not env:
            continue

        # Apply filters if specified
        if env_filter and env != env_filter:
            continue
        if model_filter and model_filter.lower() not in model.lower():
            continue

        # Extract error metric
        results_data = result['data'].get('results', [])
        if not isinstance(results_data, list):
            continue

        for i, entry in enumerate(results_data):
            if not entry or not isinstance(entry, list) or len(entry) < 1:
                continue
            metrics = entry[0]
            if not isinstance(metrics, list) or len(metrics) < 2:
                continue
            error = abs(metrics[0]) if metrics[0] is not None else None
            if error is None:
                continue

            budget = None
            if isinstance(budgets, list) and i < len(budgets):
                budget = budgets[i]
            elif isinstance(budgets, (int, float)):
                budget = int(budgets)
            else:
                budget = i
            data[env][int(budget)].append(error)

    return data

def plot_progression(data, output_file="outputs/budget_progression.png", title="Budget Progression Analysis"):
    """Create budget progression plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    axes = axes.flatten()

    environments = sorted(data.keys())

    for idx, env in enumerate(environments):
        if idx >= len(axes):
            break

        ax = axes[idx]
        env_data = data[env]

        budgets = sorted(env_data.keys())
        means = [np.mean(env_data[b]) for b in budgets]
        stds = [np.std(env_data[b]) if len(env_data[b]) > 1 else 0 for b in budgets]

        # Plot with error bars
        ax.errorbar(budgets, means, yerr=stds, marker='o', linewidth=2,
                   markersize=8, capsize=5, capthick=2, label=env)

        # Add value labels
        for b, m in zip(budgets, means):
            ax.text(b, m, f'{m:.4f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Budget (# experiments)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
        ax.set_title(f'{env.upper()}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(budgets)

        # Add improvement annotations
        if len(budgets) > 1:
            for i in range(len(budgets)-1):
                improvement = means[i] - means[i+1]
                pct = (improvement / means[i] * 100) if means[i] != 0 else 0
                mid_x = (budgets[i] + budgets[i+1]) / 2
                mid_y = (means[i] + means[i+1]) / 2

                color = 'green' if improvement > 0 else 'red'
                symbol = '↓' if improvement > 0 else '↑'
                ax.annotate(f'{symbol} {pct:.1f}%',
                          xy=(mid_x, mid_y),
                          fontsize=9, color=color, fontweight='bold',
                          ha='center', va='bottom')

    # Hide unused subplots
    for idx in range(len(environments), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Ensure output directory exists
    Path("outputs").mkdir(exist_ok=True)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_file}")

    return output_file

def main():
    parser = argparse.ArgumentParser(description="Plot budget progression for benchmark results")
    parser.add_argument("--root", default="results", help="Results directory (default: results)")
    parser.add_argument("--model", default=None, help="Filter by model name (substring match)")
    parser.add_argument("--env", default=None, help="Filter by environment name")
    parser.add_argument("--output", default="outputs/budget_progression.png", help="Output file path")
    parser.add_argument("--title", default="Budget Progression Analysis", help="Plot title")
    args = parser.parse_args()

    print("📊 Creating Budget Progression Plots...")
    if args.model:
        print(f"   Filtering by model: {args.model}")
    if args.env:
        print(f"   Filtering by env: {args.env}")

    data = load_results(root=args.root, model_filter=args.model, env_filter=args.env)

    if not data:
        print("⚠️  No results found!")
        return

    print(f"✅ Loaded data for {len(data)} environments")

    output_file = plot_progression(data, output_file=args.output, title=args.title)

    print("\n🎨 Visualization complete!")
    print(f"   View: {output_file}")

if __name__ == "__main__":
    main()
