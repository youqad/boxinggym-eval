---
title: BoxingGym Dashboard
emoji: 🥊
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.39.0"
app_file: scripts/streamlit_app/app.py
pinned: false
license: mit
---

# 🥊 BoxingGym Dashboard

Interactive dashboard for analyzing LLM agent performance on the BoxingGym benchmark.

## Features

- **Leaderboard**: Model rankings with bootstrap 95% CIs and significance testing (works out-of-the-box with bundled data)
- **Sweep Analysis**: Compare multiple W&B sweeps with auto-discovery
- **Model Rankings**: See all models ranked by performance
- **Parameter Importance**: Which hyperparameters matter most
- **Heatmaps**: Environment × Model performance visualization
- **Paper Comparison**: Compare against baseline results

## Setup

The **Leaderboard** page works with bundled canonical snapshot data — no secrets needed.

<!-- CANONICAL_HF_SNAPSHOT:START -->
## Canonical Snapshot

- Generated: **2026-02-13T21:14:31Z**
- Source: `.boxing-gym-cache/runs.parquet`
- Filters: `budget >= 10`, `exclude_outliers=true`
- Valid runs: **1,948** across **7 models** and **10 environments**
- Current #1 model: **MiniMax-M2.1** (`z=+0.185`)

The Leaderboard page defaults to this pinned snapshot in HF demo mode.
Live W&B views are exploratory and may differ from the published snapshot.
<!-- CANONICAL_HF_SNAPSHOT:END -->

### Optional Secrets (for Sweep Analysis)

To fetch live sweep data, add a WandB API key in Space settings:

1. Go to Space Settings → Variables and secrets
2. Add a new secret:
   - Name: `WANDB_API_KEY`
   - Value: Your WandB API key (from https://wandb.ai/authorize)

### Optional Configuration

- `WANDB_ENTITY`: Your W&B entity
- `WANDB_PROJECT`: Your W&B project (default: "boxing-gym")

## About

Dashboard for [**BoxingGym**](https://github.com/kanishkg/boxing-gym) (Gandhi et al., Stanford).

**Paper:** [arXiv:2501.01540](https://arxiv.org/abs/2501.01540)
**Original:** [kanishkg/boxing-gym](https://github.com/kanishkg/boxing-gym)
**Fork:** [youqad/boxing-gym](https://github.com/youqad/boxing-gym)
