# 🥊 BoxingGym: Multi-Model Evaluation & Analysis

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-3670A0?style=flat-square&logo=python&logoColor=ffdd54" alt="Python 3.11+"></a>&nbsp;
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/badge/uv-enabled-blue?style=flat-square" alt="uv"></a>&nbsp;
  <a href="https://wandb.ai/site"><img src="https://img.shields.io/badge/W%26B-000000?style=flat-square&logo=weightsandbiases&logoColor=white" alt="Weights & Biases"></a>&nbsp;
  <a href="https://streamlit.io"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"></a>&nbsp;
  <a href="https://www.pymc.io/"><img src="https://img.shields.io/badge/PyMC-3498DB?style=flat-square" alt="PyMC"></a>&nbsp;
  <a href="https://github.com/BerriAI/litellm"><img src="https://img.shields.io/badge/LiteLLM-enabled-blueviolet?style=flat-square" alt="LiteLLM"></a>&nbsp;
  <a href="https://arxiv.org/abs/2501.01540"><img src="https://img.shields.io/badge/arXiv-2501.01540-b31b1b.svg?style=flat-square" alt="arXiv"></a>
</p>

<p align="center">
  <em>Fork of <a href="https://github.com/kanishkg/boxing-gym">Stanford's BoxingGym</a> with multi-model evaluation and analysis dashboards</em>
</p>

---

## What's New

This fork adds multi-model evaluation tooling to Stanford's BoxingGym:

- 7 LLM providers (DeepSeek, MiniMax-M2.1, GLM-4.7, Kimi-K2, GPT-4o, GPT-5.1-Codex-mini, Qwen3-32B)
- TUI + Streamlit dashboards (rankings, heatmaps, parameter importance, PPL diagnostics)
- WandB sweep orchestration with parallel agents
- Cost tracking per model
- 522 tests
- `uv` for dependency management
- 2,001 completed sweep runs across 10 environments

---

## 🎯 Evaluation Results

> **2,001 valid runs** (budget ≥ 10, |z| < 100) across **7 models** and **10 environments**

### Overall Model Rankings

Mean z-score across all environments (lower is better):

| Rank | Model | Mean z | 95% CI | Runs |
|------|-------|--------|--------|------|
| 1 | **MiniMax-M2.1** | **+0.059** | [+0.000, +0.119] | 327 |
| 2 | DeepSeek | +0.068 | [+0.002, +0.144] | 376 |
| 3 | Kimi-K2 | +0.126 | [+0.047, +0.212] | 306 |
| 4 | GLM-4.7 | +0.137 | [+0.041, +0.245] | 326 |
| 5 | Qwen3-32B | +0.179 | [+0.083, +0.290] | 190 |
| 6 | GPT-4o | +0.257 | [+0.123, +0.446] | 230 |
| 7 | GPT-5.1-Codex-mini | +0.265 | [+0.147, +0.398] | 246 |

### Per-Environment Champions

Best mean performance per environment (lower z-score = better):

| Environment | Description | Best Model | Mean z |
|-------------|-------------|------------|--------|
| **death_process** | Disease spread modeling | GPT-5.1-Codex-mini | **-0.639** ✓ |
| **hyperbolic_temporal_discount** | Intertemporal choice | DeepSeek | **-0.430** ✓ |
| **lotka_volterra** | Predator-prey dynamics | DeepSeek | -0.306 ✓ |
| **moral_machines** | Autonomous vehicle ethics | GPT-5.1-Codex-mini | -0.140 ✓ |
| **irt** | Item response theory | GPT-4o | -0.120 ✓ |
| **location_finding** | Signal source localization | GPT-4o | -0.106 ✓ |
| **peregrines** | Falcon population dynamics | DeepSeek | -0.084 ✓ |
| **dugongs** | Sea cow growth modeling | GPT-4o | -0.041 ✓ |
| **survival** | Breast cancer survival | GLM-4.7 | +0.429 |
| **emotion** | Emotion from gambling | Qwen3-32B | +1.286 |

✓ = Beats baseline (negative z-score)

### Takeaways

**MiniMax-M2.1** wins overall (+0.059 mean z), statistically tied with DeepSeek.

**DeepSeek** dominates temporal/dynamic tasks: hyperbolic (-0.430), lotka_volterra (-0.306), peregrines (-0.084).

**GPT-4o with PPL** wins dugongs, irt, location_finding. PPL helps on some tasks despite overall negative effect.

**Hard environments**: emotion (+1.286) and survival (+0.429) remain above baseline for all models.

**PPL finding**: OED without PPL (z=+0.009) significantly beats OED+PPL (z=+0.702). Effect size: d=0.70 (medium).

**Only GPT-5.1-Codex-mini** is significantly worse than #1 (p=0.02 after FDR correction).

**Environment difficulty**: death_process easiest (z=-0.34), emotion hardest (z=+1.29). Diminishing returns after budget ~20.

**Filtering**: 2,001 runs after budget ≥ 10, |z| < 100 (outlier removal). Rankings above include only models with n ≥ 100.

### Statistical Methods

All queries use proper statistical testing:

- **Welch's t-test**: Comparing groups with unequal variances
- **Bootstrap CI**: 95% confidence intervals via 10,000 resamples
- **Cohen's d**: Effect size interpretation (negligible < 0.2 < small < 0.5 < medium < 0.8 < large)
- **Benjamini-Hochberg**: FDR correction for multiple comparisons

Run `box query all` for full statistics.

---

## Quick Start

```bash
git clone https://github.com/youqad/boxing-gym-wip.git
cd boxing-gym-wip

# Install with uv
uv sync
uv run python -c "import boxing_gym; print('✓ Import OK')"

# Set API keys (create .env with your keys)
echo "OPENAI_API_KEY=sk-..." >> .env
echo "DEEPSEEK_API_KEY=sk-..." >> .env

# Run single experiment
uv run python run_experiment.py seeds=null seed=1 exp=oed envs=hyperbolic_direct llms=gpt-4o

# Run full test suite
uv run pytest tests/ -v
```

**Requirements**: Python 3.11+, API keys for desired LLM providers

---

## Analysis Interfaces

The `box` CLI handles result analysis. Sync once, query as needed.

```
┌─────────────────┐    ┌──────────────────────────┐    ┌─────────────────┐
│  Local JSON     │───▶│  box sync --local        │───▶│  Parquet Cache  │
│  results/*.json │    │  (aggregates & caches)   │    │  .boxing-gym-   │
└─────────────────┘    └──────────────────────────┘    │   cache/        │
                                                       └────────┬────────┘
                                                                │
┌─────────────────┐    ┌──────────────────────────┐             │
│  Analysis       │◀───│  box query <name>        │◀────────────┘
│  (tables, stats)│    │  (runs pre-built queries)│
└─────────────────┘    └──────────────────────────┘
```

Cache results first (run once, or after adding new results):
```bash
uv run box sync --local results/    # parse JSON → .boxing-gym-cache/runs.parquet
uv run box sync --status            # check cache: 6,068 runs, 16 models, 10 envs
```

Then run queries (instant):
```bash
uv run box query --list             # list available queries
uv run box query leaderboard        # model rankings with significance tests
uv run box query oed-discovery      # OED vs Discovery 2×2 comparison
uv run box query ppl-impact         # PPL effect analysis
uv run box query env-difficulty     # environment difficulty rankings
uv run box query best-configs       # best config per environment
uv run box query paper-comparison   # comparison with paper baselines
uv run box query all                # full report (all queries)
```

Filter and format options:
```bash
uv run box query leaderboard --min-budget 10     # filter by minimum budget
uv run box query leaderboard --env dugongs       # filter by environment
uv run box query leaderboard --format md         # markdown output
uv run box query leaderboard --format json       # JSON output
uv run box query all --include-outliers          # include flagged outliers
```

### CLI, TUI, and Web

**CLI** — quick answers, scripting, CI/CD:
```bash
uv run box query leaderboard
uv run box query all
```

**TUI** — interactive terminal:

```bash
uv run box results --tui
```

| View | Description |
|------|-------------|
| model-rankings | Leaderboard: z-scores, bootstrap CIs, significance |
| heatmap | Environment × Model z-mean matrix |
| best-configs | Top configuration per environment |
| budget-progression | Performance trends across budget values |
| parameter-importance | Permutation importance analysis |
| seed-stability | Cross-seed variance, unstable configs |
| local-summary | Quick summary of local results |

**Web** — Streamlit dashboard:
```bash
uv run box results --web
```

| Page | Description |
|------|-------------|
| Leaderboard | Model rankings with bootstrap 95% CIs and significance testing |
| Benchmark Dashboard | Paper baselines (GPT-4o, BOX), filtering, delta coloring |
| Sweep Analysis | Parameter importance, model rankings, heatmaps, best configs |
| Paper Comparison | Gandhi et al. baseline comparison |
| PPL Examples | Generated PyMC models with sampling diagnostics |
| LLM Call Logs | Cost breakdown by model, latency histograms, token counts |

### Make shortcuts

```bash
make aggregate   # box sync --local results/
make plot        # box query all
make clean       # rm outputs/ .boxing-gym-cache/
```

---

## Configuration System

Hydra-based config with 10 environments, 15+ LLMs, and 3 experiment types:

```bash
# Override any config parameter
uv run python run_experiment.py \
  envs=hyperbolic_direct \
  exp=oed \
  llms=deepseek-v3.2 \
  seeds=[1,2,3,4,5] \
  +wandb=true
```

**Configs**:
- Environments: `conf/envs/*.yaml` (10 environments × 2 goal types)
- LLMs: `conf/llms/*.yaml` (19 models across 9 providers)
- Experiments: `conf/exp/{oed,oed_box,discovery}.yaml`
- Sweeps: `sweeps/*.yaml` (WandB sweep configs)

---

## Upstream Docs

Original BoxingGym documentation (environment details, API, metrics): **[README_upstream.md](README_upstream.md)**

---

## Attribution

This is a fork of **BoxingGym** by Gandhi et al. (Stanford University).

**Original Paper**: [BoxingGym: Benchmarking Progress in Automated Experimental Design and Model Discovery](https://arxiv.org/abs/2501.01540)
*Kanishk Gandhi, Michael Y. Li, Dorsa Sadigh, Noah D. Goodman (Stanford University)*

**Original Repository**: [github.com/kanishkg/boxing-gym](https://github.com/kanishkg/boxing-gym)

---

## License

MIT (same as upstream)

---

## Citation

Cite the original paper:

```bibtex
@article{gandhi2025boxinggym,
  title={BoxingGym: Benchmarking Progress in Automated Experimental Design and Model Discovery},
  author={Gandhi, Kanishk and Li, Michael Y and Sadigh, Dorsa and Goodman, Noah D},
  journal={arXiv preprint arXiv:2501.01540},
  year={2025}
}
```

