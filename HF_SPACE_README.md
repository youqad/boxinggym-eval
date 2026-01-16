---
title: BoxingGym Dashboard
emoji: 🥊
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.39.0"
app_file: hf_space_app.py
pinned: false
license: mit
---

# 🥊 BoxingGym Dashboard

Interactive dashboard for analyzing LLM agent performance on the BoxingGym benchmark.

## Features

- **Sweep Analysis**: Compare multiple W&B sweeps with auto-discovery
- **Model Rankings**: See all models ranked by performance
- **Parameter Importance**: Understand what hyperparameters matter
- **Heatmaps**: Environment × Model performance visualization
- **Paper Comparison**: Compare against baseline results

## Setup

### Required Secrets

This Space requires a WandB API key to fetch sweep data. Add it in the Space settings:

1. Go to Space Settings → Variables and secrets
2. Add a new secret:
   - Name: `WANDB_API_KEY`
   - Value: Your WandB API key (from https://wandb.ai/authorize)

### Optional Configuration

- `WANDB_ENTITY`: Your W&B entity
- `WANDB_PROJECT`: Your W&B project (default: "boxing-gym")

## Usage

1. Go to **Sweep Analysis** in the sidebar
2. Dashboard auto-discovers available sweeps
3. Select sweeps from the dropdown
4. View charts and tables

## About

Dashboard for [**BoxingGym**](https://github.com/kanishkg/boxing-gym) (Gandhi et al., Stanford).

**Paper:** [arXiv:2501.01540](https://arxiv.org/abs/2501.01540)
**Original:** [kanishkg/boxing-gym](https://github.com/kanishkg/boxing-gym)
**Fork:** [youqad/boxing-gym](https://github.com/youqad/boxing-gym)
