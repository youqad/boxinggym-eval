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
  <em>Fork of <a href="https://github.com/kanishkg/boxing-gym">Stanford's BoxingGym</a> with multi-model evaluation and comprehensive dashboards</em>
</p>

---

## What's New in This Fork

This fork extends Stanford's BoxingGym benchmark with production-ready tooling for large-scale LLM evaluation:

- **7 LLM Providers**: DeepSeek, MiniMax-M2.1, GLM-4.7, Kimi-K2, GPT-4o, GPT-5.1-Codex-mini, Qwen3-32B
- **Comprehensive Dashboards**: TUI + Streamlit with 5 analysis views (model rankings, parameter importance, heatmaps, budget progression, PPL diagnostics)
- **WandB Sweep Orchestration**: Distributed sweep management with parallel agents
- **Usage Tracking**: Real-time cost estimation and token counting per model
- **Test Suite**: 393 tests covering experiment loop, agents, and evaluation
- **Analysis Utilities**: 12 scripts for sweep analysis, outlier detection, and result aggregation
- **Migrated to `uv`**: Modern Python dependency management (from pip/requirements.txt)
- **1,554 Sweep Runs**: Completed experiments across 10 environments with 7 models

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

### Key Insights

1. **Overall Champion: MiniMax-M2.1**
   - Lowest mean z-score (+0.062) across all environments
   - Tightest variance (Std=0.54) → most consistent performance
   - Strong on location_finding, morals, dugongs

2. **Environment Specialist: DeepSeek**
   - Dominates 3 environments by mean: hyperbolic (-0.438), lotka_volterra (-0.297), peregrines (-0.140)
   - Second-best overall (+0.080 mean z)
   - 302 runs (most data)

3. **Complex Tasks: GPT-5.1-Codex-mini**
   - Best on death_process (-0.629), morals (-0.140), irt (-0.050)
   - Strong on structured causal reasoning

4. **Challenging Environments**
   - **emotion** (+1.286 best score): Hardest for all models
   - **survival** (+0.415 best score): Above-baseline performance still difficult

5. **PPL Impact**
   - OED without PPL wins **6/7** comparisons vs OED with PPL
   - **Hypothesis**: LLMs may implicitly perform Bayesian updating, making explicit probabilistic programs redundant

6. **Environment Sensitivity**
   - LLM choice matters **10x more** on hyperbolic (importance: 1.82) vs survival (0.18)
   - Some tasks are model-agnostic, others highly sensitive to model capabilities

See [CLAUDE.md](CLAUDE.md) for full sweep results and analysis.

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

Three ways to explore results:

### 1. CLI Queries (non-interactive)

```bash
uv run box sync --local results/       # Cache local results first
uv run box query leaderboard           # Model rankings with 95% CIs
uv run box query all                   # Full statistical report
uv run box query --list                # See all available queries
```

### 2. TUI (interactive terminal)

```bash
uv run box results --tui
```

Textual-based terminal UI with keyboard navigation:
- Model rankings, environment × model heatmaps
- Parameter importance, budget progression
- Seed stability, best configs, call logs

### 3. Streamlit (web dashboard)

```bash
uv run box results --web
```

**5 Analysis Pages**:
1. **Benchmark Dashboard**: Compare results against paper baselines (GPT-4o, BOX)
2. **Sweep Analysis**: Parameter importance, model rankings, best configs
3. **Paper Comparison**: Replicate paper figures with your data overlay
4. **PPL Examples**: View generated PyMC models with diagnostics (Rhat, ESS, divergences)
5. **LLM Call Logs**: Cost tracking, latency analysis, token counts

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

## Upstream Documentation

For the original BoxingGym documentation (environment details, API reference, metrics), see:

📄 **[README_upstream.md](README_upstream.md)** (Stanford's original README)

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

If you use this fork in your research, please cite both the original BoxingGym paper and acknowledge this fork:

```bibtex
@article{gandhi2025boxinggym,
  title={BoxingGym: Benchmarking Progress in Automated Experimental Design and Model Discovery},
  author={Gandhi, Kanishk and Li, Michael Y and Sadigh, Dorsa and Goodman, Noah D},
  journal={arXiv preprint arXiv:2501.01540},
  year={2025}
}
```

