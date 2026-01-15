#!/usr/bin/env python3
"""Comprehensive OED analysis for all local results."""

import json
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console(width=140)

def load_results():
    """Load all JSON results with flexible parsing."""
    results = []
    for f in Path('results').rglob('*.json'):
        if f.name.startswith('llm_calls'):
            continue
        try:
            with open(f) as fp:
                data = json.load(fp)

                config = data.get('config', {})
                z_results = data.get('data', {}).get('z_results', [])

                if not z_results:
                    continue

                # Get final z_mean (highest budget)
                final_z = z_results[-1]

                # Extract exp mode
                exp_mode = config.get('exp', 'unknown')
                if isinstance(exp_mode, dict):
                    exp_mode = exp_mode.get('experiment_type', exp_mode.get('type', 'unknown'))

                # Extract use_ppl
                use_ppl = config.get('use_ppl', False)

                results.append({
                    'exp': str(exp_mode),
                    'use_ppl': bool(use_ppl),
                    'model': str(config.get('llms', 'unknown')),
                    'env': str(config.get('envs', 'unknown')),
                    'seed': config.get('seed', -1),
                    'z_mean': float(final_z.get('z_mean', float('inf'))),
                    'budget': int(final_z.get('budget', 0)),
                })
        except Exception as e:
            continue

    return pd.DataFrame(results)

def main():
    df = load_results()

    # Filter valid
    valid_df = df[
        (df['z_mean'].notna()) &
        (df['z_mean'].abs() < 100) &
        (df['budget'] >= 0) &
        (df['exp'] != 'unknown')
    ].copy()

    console.print(f'[bold green]✓ Loaded {len(valid_df)} valid runs[/bold green]')
    console.print(f'Exp modes: {dict(valid_df["exp"].value_counts())}')
    console.print(f'PPL usage: {dict(valid_df["use_ppl"].value_counts())}')
    console.print(f'Environments: {len(valid_df["env"].unique())} unique\n')

    # 2x2 Matrix
    table = Table(title='2×2 Matrix: Experiment Mode × PPL Usage', show_header=True, width=140)
    table.add_column('Mode', style='cyan', width=15)
    table.add_column('use_ppl=False', justify='center', width=35)
    table.add_column('use_ppl=True', justify='center', width=35)
    table.add_column('Δ (PPL effect)', justify='center', width=35)

    for exp in sorted(valid_df['exp'].unique()):
        exp_df = valid_df[valid_df['exp'] == exp]

        no_ppl_df = exp_df[exp_df['use_ppl'] == False]
        yes_ppl_df = exp_df[exp_df['use_ppl'] == True]

        no_ppl_mean = no_ppl_df['z_mean'].mean()
        no_ppl_n = len(no_ppl_df)

        yes_ppl_mean = yes_ppl_df['z_mean'].mean()
        yes_ppl_n = len(yes_ppl_df)

        delta = yes_ppl_mean - no_ppl_mean if pd.notna(no_ppl_mean) and pd.notna(yes_ppl_mean) else float('nan')

        no_ppl_str = f'{no_ppl_mean:+.3f} (n={no_ppl_n})' if pd.notna(no_ppl_mean) else 'N/A'
        yes_ppl_str = f'{yes_ppl_mean:+.3f} (n={yes_ppl_n})' if pd.notna(yes_ppl_mean) else 'N/A'

        if pd.notna(delta):
            delta_symbol = '✓ PPL helps' if delta < 0 else '✗ PPL hurts'
            delta_str = f'{delta:+.3f} {delta_symbol}'
        else:
            delta_str = 'N/A'

        table.add_row(exp, no_ppl_str, yes_ppl_str, delta_str)

    console.print(table)

    # Configuration rankings
    console.print()
    rank_table = Table(title='Configuration Rankings (Best to Worst)', show_header=True, width=140)
    rank_table.add_column('Rank', width=8)
    rank_table.add_column('Mode', width=15)
    rank_table.add_column('PPL', width=8)
    rank_table.add_column('Mean z', justify='right', width=12)
    rank_table.add_column('Count', justify='right', width=10)

    configs = []
    for exp in valid_df['exp'].unique():
        for ppl in [False, True]:
            subset = valid_df[(valid_df['exp'] == exp) & (valid_df['use_ppl'] == ppl)]
            if len(subset) > 0:
                configs.append({
                    'exp': exp,
                    'ppl': ppl,
                    'mean_z': subset['z_mean'].mean(),
                    'count': len(subset)
                })

    if not configs:
        console.print('[yellow]No valid configurations found[/yellow]')
        return

    configs_df = pd.DataFrame(configs).sort_values('mean_z')

    for i, row in enumerate(configs_df.itertuples(), 1):
        medal = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else '  '
        color = 'green' if row.mean_z < 0 else 'red' if row.mean_z > 0.5 else 'yellow'

        rank_table.add_row(
            f'{medal}{i}',
            row.exp,
            '✓' if row.ppl else '✗',
            f'[{color}]{row.mean_z:+.3f}[/{color}]',
            str(row.count)
        )

    console.print(rank_table)

    # Per-environment winners
    console.print()
    env_table = Table(title='Best Configuration Per Environment', show_header=True, width=140)
    env_table.add_column('Environment', width=30)
    env_table.add_column('Winner', width=25)
    env_table.add_column('Mean z', justify='right', width=15)
    env_table.add_column('Runs', justify='right', width=10)

    for env in sorted(valid_df['env'].unique()):
        env_df = valid_df[valid_df['env'] == env]

        best_config = None
        best_z = float('inf')

        for exp in env_df['exp'].unique():
            for ppl in [False, True]:
                subset = env_df[(env_df['exp'] == exp) & (env_df['use_ppl'] == ppl)]
                if len(subset) > 0:
                    mean_z = subset['z_mean'].mean()
                    if mean_z < best_z:
                        best_z = mean_z
                        ppl_str = 'PPL' if ppl else 'no-PPL'
                        best_config = (f'{exp} + {ppl_str}', mean_z, len(subset))

        if best_config:
            color = 'green' if best_config[1] < 0 else 'red' if best_config[1] > 0.5 else 'yellow'
            env_table.add_row(
                env,
                best_config[0],
                f'[{color}]{best_config[1]:+.3f}[/{color}]',
                str(best_config[2])
            )

    console.print(env_table)

if __name__ == '__main__':
    main()
