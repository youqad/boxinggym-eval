# 🥊 SynthStats Multi-Model Evaluation and Analysis of BoxingGym

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-3670A0?style=flat-square&logo=python&logoColor=ffdd54" alt="Python 3.11+"></a>&nbsp;
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/badge/uv-enabled-blue?style=flat-square" alt="uv"></a>&nbsp;
  <a href="https://wandb.ai/site"><img src="https://img.shields.io/badge/W%26B-000000?style=flat-square&logo=weightsandbiases&logoColor=white" alt="Weights & Biases"></a>&nbsp;
  <a href="https://streamlit.io"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"></a>&nbsp;
  <a href="https://www.pymc.io/"><img src="https://img.shields.io/badge/PyMC-3498DB?style=flat-square" alt="PyMC"></a>&nbsp;
  <a href="https://github.com/BerriAI/litellm"><img src="https://img.shields.io/badge/LiteLLM-enabled-blueviolet?style=flat-square" alt="LiteLLM"></a>&nbsp;
  <a href="https://arxiv.org/abs/2501.01540"><img src="https://img.shields.io/badge/arXiv-2501.01540-b31b1b.svg?style=flat-square" alt="arXiv"></a>&nbsp;
  <a href="https://huggingface.co/spaces/youkad/boxing-gym-dashboard"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Space-Live%20Demo-yellow?style=flat-square" alt="HF Space"></a>&nbsp;
  <a href="https://github.com/youqad/boxing-gym-wip/actions/workflows/ci.yml"><img src="https://github.com/youqad/boxing-gym-wip/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

<p align="center">
  <em>Fork of <a href="https://github.com/kanishkg/boxing-gym">Stanford's BoxingGym</a> with multi-model evaluation and analysis dashboards</em>
</p>

---

## What's New

Stanford's BoxingGym is a benchmarking framework designed to evaluate the capabilities of language-based agents in experimental design and model discovery. The original benchmark was primarily designed for evaluation through OpenAI endpoints.

This fork adds multi-model evaluation tooling to Stanford's BoxingGym:

- 7 LLM providers (DeepSeek, MiniMax-M2.1, GLM-4.7, Kimi-K2, GPT-4o, GPT-5.1-Codex-mini, Qwen3-32B)
- TUI + Streamlit dashboards (rankings, heatmaps, parameter importance, PPL diagnostics)
- WandB sweep orchestration with parallel agents
- Cost tracking per model
- 522 tests
- `uv` for dependency management
- Canonical snapshot + generated README/HF summaries from one shared parquet source

---

## 🎯 Evaluation Results

<!-- CANONICAL_METADATA:START -->
> Canonical snapshot generated: **2026-02-13T21:14:31Z**
> Source: `.boxing-gym-cache/runs.parquet` @ `a0211d7c6e83300cb1216d4789b084a910523083`
> Filters: `budget >= 10`, `exclude_outliers=true`
> Valid runs: **1,948** across **7 models** and **10 environments**
<!-- CANONICAL_METADATA:END -->

<!-- CANONICAL_RESULTS:START -->
### Overall Model Rankings

Mean z-score across all filtered runs (lower is better):

| Rank | Model | Mean z | 95% CI | vs #1 (FDR p) | Runs |
| --- | --- | --- | --- | --- | --- |
| 1 | MiniMax-M2.1 | +0.185 | [+0.071, +0.315] | — | 311 |
| 2 | GPT-4o | +0.256 | [+0.138, +0.388] | 0.424 | 226 |
| 3 | Kimi-K2 | +0.262 | [+0.130, +0.408] | 0.424 | 302 |
| 4 | Qwen3-32B | +0.275 | [+0.115, +0.464] | 0.424 | 188 |
| 5 | DeepSeek-V3.2 | +0.296 | [+0.122, +0.532] | 0.424 | 369 |
| 6 | GLM-4.7 | +0.546 | [+0.257, +0.904] | 0.120 | 314 |
| 7 | GPT-5.1-Codex-Mini | +0.586 | [+0.310, +0.966] | 0.120 | 238 |

### Per-Environment Champions (Best Model Only)

_Definition: mean over all runs for each model within environment (ignores `use_ppl`)._

| Environment | Best Model | Mean z | Status |
| --- | --- | --- | --- |
| death_process | GPT-4o | -0.903 | ✓ Beats baseline |
| lotka_volterra | DeepSeek-V3.2 | -0.301 | ✓ Beats baseline |
| hyperbolic_temporal_discount | DeepSeek-V3.2 | -0.152 | ✓ Beats baseline |
| moral_machines | GPT-5.1-Codex-Mini | -0.140 | ✓ Beats baseline |
| irt | GPT-5.1-Codex-Mini | -0.050 | ✓ Beats baseline |
| dugongs | Qwen3-32B | -0.040 | ✓ Beats baseline |
| peregrines | DeepSeek-V3.2 | +0.094 | Above baseline |
| survival | GLM-4.7 | +0.434 | Above baseline |
| emotion | Qwen3-32B | +1.286 | Above baseline |
| location_finding | GPT-4o | +1.396 | Above baseline |

### Per-Environment Champions (Best Model + PPL)

_Definition: mean over each `(model, use_ppl)` combination within environment._

| Environment | Best Model | use_ppl | Mean z | Status |
| --- | --- | --- | --- | --- |
| death_process | Qwen3-32B | true | -1.170 | ✓ Beats baseline |
| lotka_volterra | DeepSeek-V3.2 | false | -0.301 | ✓ Beats baseline |
| hyperbolic_temporal_discount | DeepSeek-V3.2 | false | -0.159 | ✓ Beats baseline |
| moral_machines | GPT-5.1-Codex-Mini | false | -0.140 | ✓ Beats baseline |
| irt | GPT-4o | true | -0.120 | ✓ Beats baseline |
| dugongs | GPT-4o | true | -0.047 | ✓ Beats baseline |
| location_finding | GPT-4o | true | -0.041 | ✓ Beats baseline |
| peregrines | DeepSeek-V3.2 | false | +0.094 | Above baseline |
| survival | GLM-4.7 | false | +0.429 | Above baseline |
| emotion | Qwen3-32B | false | +1.286 | Above baseline |

### Key Findings

- **MiniMax-M2.1** leads overall with z=+0.185
- **6/10** environments beat baseline under model-only definition
- Models significantly worse than #1 (FDR < 0.05): None

### Statistical Methods

- **Welch's t-test**: Group comparison with unequal variances
- **Bootstrap CI**: 95% confidence intervals from resampling
- **Benjamini-Hochberg**: FDR correction for multiple comparisons

_Snapshot version: `2026-02-13`_
<!-- CANONICAL_RESULTS:END -->

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
uv run box sync --status            # check current cache counts
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

### Canonical Snapshot Refresh

Regenerate the pinned snapshot + README/HF summary blocks from the local cache:

```bash
uv run box sync --local results/
uv run python scripts/refresh_canonical_results.py
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
