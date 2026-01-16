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
- 393 tests
- `uv` for dependency management
- 1,554 completed sweep runs across 10 environments

---

## 🎯 Evaluation Results

> **1,554 valid runs** (budget ≥ 10, |z| < 100) across **7 models** and **10 environments**

### Overall Model Rankings

Mean z-score across all environments (lower is better):

| Rank | Model | Mean z | Std | Runs |
|------|-------|--------|-----|------|
| 1 | **MiniMax-M2.1** | **+0.062** | **0.54** | 236 |
| 2 | DeepSeek | +0.080 | 0.73 | 302 |
| 3 | Kimi-for-coding | +0.176 | 0.84 | 205 |
| 4 | GLM-4.7 | +0.212 | 1.02 | 236 |
| 5 | GPT-4o | +0.257 | 1.28 | 230 |
| 6 | GPT-5.1-Codex-mini | +0.260 | 1.00 | 249 |
| 7 | Qwen3-32B | +0.311 | 0.82 | 96 |

### Per-Environment Champions

Best mean performance per environment (lower z-score = better):

| Environment | Description | Best Model | Mean z |
|-------------|-------------|------------|--------|
| **death_process** | Disease spread modeling in populations | GPT-5.1-Codex-mini | **-0.629** ✓ |
| **hyperbolic_temporal_discount** | Human decision-making between immediate/delayed rewards | DeepSeek | **-0.438** ✓ |
| **lotka_volterra** | Predator-prey population dynamics | DeepSeek | -0.297 ✓ |
| **peregrines** | Falcon population dynamics over time | DeepSeek | -0.140 ✓ |
| **morals** | Ethical decision-making in autonomous vehicles | GPT-5.1-Codex-mini | -0.140 ✓ |
| **irt** | Student-question performance modeling (Item Response Theory) | GPT-5.1-Codex-mini | -0.050 ✓ |
| **dugongs** | Sea cow growth modeling (age vs length) | Qwen3-32B | -0.035 ✓ |
| **location_finding** | Signal source localization in n-dimensional space | MiniMax-M2.1 | +0.211 |
| **survival** | Breast cancer patient survival prediction | GLM-4.7 | +0.415 |
| **emotion** | Emotion prediction from gambling outcomes | Qwen3-32B | +1.286 |

✓ = Beats baseline (negative z-score)

### Takeaways

**MiniMax-M2.1** wins overall (+0.062 mean z, lowest variance). Most consistent across environments.

**DeepSeek** dominates temporal/dynamic tasks: hyperbolic (-0.438), lotka_volterra (-0.297), peregrines (-0.140).

**GPT-5.1-Codex-mini** excels at causal reasoning: death_process (-0.629), morals (-0.140), irt (-0.050).

**Hard environments**: emotion (+1.286) and survival (+0.415) remain above baseline for all models.

**PPL finding**: OED without PPL beats OED+PPL in 6/7 comparisons. LLMs may already do implicit Bayesian updating.

**Model sensitivity varies 10×**: hyperbolic (importance: 1.82) vs survival (0.18). Some tasks are model-agnostic.

Run `box query all` for full statistics.

---

## Quick Start

```bash
git clone https://github.com/youqad/boxing-gym.git
cd boxing-gym

# Install with uv
uv sync
uv run python -c "import boxing_gym; print('✓ Import OK')"

# Set API keys
cp .env.example .env
# Edit .env with your keys: OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, etc.

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
uv run box sync --status            # check cache: 4,864 runs, 16 models, 10 envs
```

Then run queries (instant):
```bash
uv run box query leaderboard        # model rankings with 95% CIs
uv run box query oed-discovery      # OED vs Discovery 2×2 with p-values
uv run box query ppl-impact         # PPL effect (Welch's t-test)
uv run box query env-difficulty     # environment difficulty rankings
uv run box query best-configs       # best config per environment
uv run box query all                # full report
uv run box query --list             # list available queries
```

Filter and format options:
```bash
uv run box query leaderboard --min-budget 10    # filter by budget
uv run box query leaderboard --env hyperbolic   # filter by environment
uv run box query leaderboard --format md        # markdown output
uv run box query leaderboard --format json      # JSON output
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

Keyboard navigation through model rankings, heatmaps, parameter importance, budget progression, best configs, call logs.

**Web** — Streamlit dashboard:
```bash
uv run box results --web
```

Five pages: benchmark dashboard (vs paper baselines), sweep analysis, paper comparison, PPL diagnostics, LLM call logs with cost tracking.

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
- LLMs: `conf/llms/*.yaml` (15+ models across 7 providers)
- Experiments: `conf/exp/{oed,discovery,naive}.yaml`
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

