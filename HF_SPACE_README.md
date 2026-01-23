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

The **Leaderboard** page works with bundled demo data — no secrets needed.

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
