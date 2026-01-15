#!/usr/bin/env python3
import argparse
import os
import sys

import pandas as pd

try:
    import seaborn as sns
except Exception:
    print("Seaborn is required. Please install with: pip install seaborn==0.13.2", file=sys.stderr)
    raise

import matplotlib.pyplot as plt


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Clean and coerce numeric fields that may be blank
    for col in ["z_mean", "z_std", "delta_vs_paper", "paper_discovery10_mean", "paper_discovery10_se"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def style():
    sns.set_theme(style="whitegrid", context="talk")
    sns.set_palette("deep")


def plot_discovery_at_10(df: pd.DataFrame, outdir: str):
    filt = (
        (df["goal"] == "direct_naive")
        & (df["include_prior"] == True)
        & (df["budget"] == 10)
        & df["env"].isin(["dugongs", "peregrines", "lotka_volterra"])
    )
    d = df.loc[filt].copy()
    if d.empty:
        print("No Discovery@10 rows found; skipping discovery plot.")
        return

    # Sort envs to a nice order
    env_order = ["dugongs", "peregrines", "lotka_volterra"]
    d["env"] = pd.Categorical(d["env"], categories=env_order, ordered=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    # Bar or point plot without internal aggregation
    sns.barplot(data=d, x="env", y="z_mean", hue="model", errorbar=None, ax=ax)

    # Add our own error bars using z_std
    # Compute x positions by grouping like seaborn (per env per model)
    # We can iterate over the patches in the bar container
    # But we need to align to rows: build a lookup keyed by (env, model)
    lookup = {(row.env, row.model): (row.z_mean, row.z_std) for row in d.itertuples()}
    for container in ax.containers:
        # Each container corresponds to one hue level across x positions
        for patch in container:
            # x center of bar
            x = patch.get_x() + patch.get_width() / 2
            env_index = int(round(x))  # tick positions are 0,1,2...
            if env_index < 0 or env_index >= len(env_order):
                continue
            env = env_order[env_index]
            model = container.get_label()
            if (env, model) in lookup:
                mean, std = lookup[(env, model)]
                if pd.notna(std):
                    ax.errorbar(x, mean, yerr=std, fmt="none", ecolor="black", elinewidth=1.2, capsize=4, capthick=1.2)

    # Reference lines for paper values
    paper = d.dropna(subset=["paper_discovery10_mean"]).drop_duplicates("env")[
        ["env", "paper_discovery10_mean", "paper_discovery10_se"]
    ]
    for i, row in paper.iterrows():
        x = env_order.index(row.env)
        ax.axhline(y=row.paper_discovery10_mean, xmin=(x - 0.4) / (len(env_order) - 1 if len(env_order) > 1 else 1), xmax=(x + 0.4) / (len(env_order) - 1 if len(env_order) > 1 else 1), color="gray", linestyle="--", linewidth=2, alpha=0.7)

    ax.set_title("Discovery@10 (with prior): Standardized Error vs Paper")
    ax.set_xlabel("Environment")
    ax.set_ylabel("Standardized Error (z)")
    ax.legend(title="Model", ncols=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "discovery_at_10.png"), dpi=200)
    fig.savefig(os.path.join(outdir, "discovery_at_10.pdf"))
    plt.close(fig)


def plot_delta_vs_paper(df: pd.DataFrame, outdir: str):
    filt = (
        (df["goal"] == "direct_naive")
        & (df["include_prior"] == True)
        & (df["budget"] == 10)
        & df["env"].isin(["dugongs", "peregrines", "lotka_volterra"])
        & df["delta_vs_paper"].notna()
    )
    d = df.loc[filt].copy()
    if d.empty:
        print("No delta_vs_paper rows found; skipping delta plot.")
        return

    env_order = ["dugongs", "peregrines", "lotka_volterra"]
    d["env"] = pd.Categorical(d["env"], categories=env_order, ordered=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=d, x="env", y="delta_vs_paper", hue="model", errorbar=None, ax=ax)
    ax.axhline(0.0, color="black", linewidth=1.5)
    ax.set_title("Difference from Paper Discovery@10 (ours − paper)")
    ax.set_xlabel("Environment")
    ax.set_ylabel("Δ standardized error (z)")
    ax.legend(title="Model", ncols=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "delta_vs_paper.png"), dpi=200)
    fig.savefig(os.path.join(outdir, "delta_vs_paper.pdf"))
    plt.close(fig)


def plot_budget_curves(df: pd.DataFrame, outdir: str):
    # For any env/goal, plot z_mean over budget per model; facet by env
    d = df.copy()
    d = d[d["budget"].notna()]
    if d.empty:
        print("No budgeted runs found; skipping budget curves plot.")
        return

    # Limit to a small set of envs if many exist
    env_order = sorted(d["env"].unique())

    g = sns.FacetGrid(d, col="env", hue="model", col_wrap=3, sharey=False, height=4, aspect=1.3)
    g.map(sns.lineplot, "budget", "z_mean")
    g.add_legend(title="Model")
    g.set_axis_labels("Budget (# experiments)", "Standardized Error (z)")
    for ax in g.axes.flatten():
        sns.despine(ax=ax)
    g.figure.suptitle("Standardized Error vs Budget", y=1.02)
    g.figure.tight_layout()
    out_png = os.path.join(outdir, "budget_curves.png")
    g.figure.savefig(out_png, dpi=200, bbox_inches="tight")
    g.figure.savefig(os.path.join(outdir, "budget_curves.pdf"), bbox_inches="tight")
    plt.close(g.figure)


def main():
    ap = argparse.ArgumentParser(description="Plot standardized BoxingGym results with seaborn.")
    ap.add_argument("--csv", default="outputs/standardized_results.csv", help="Input CSV produced by aggregate_results.py")
    ap.add_argument("--outdir", default="outputs/plots", help="Output directory for figures")
    args = ap.parse_args()

    ensure_outdir(args.outdir)
    style()
    df = load_csv(args.csv)

    plot_discovery_at_10(df, args.outdir)
    plot_delta_vs_paper(df, args.outdir)
    plot_budget_curves(df, args.outdir)
    print(f"Wrote figures to {args.outdir}")


if __name__ == "__main__":
    main()

