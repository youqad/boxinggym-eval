# BoxingGym Sweeps

WandB sweep configurations for benchmarking LLMs on Optimal Experimental Design (OED) tasks.

## Models (7)

| Model | Provider | Config |
|-------|----------|--------|
| gpt-4o | OpenAI | `conf/llms/gpt-4o.yaml` |
| gpt-5.1-codex-mini | OpenAI | `conf/llms/gpt-5.1-codex-mini.yaml` |
| deepseek-v3.2 | DeepSeek | `conf/llms/deepseek-v3.2.yaml` |
| glm-4.7 | Zhipu AI | `conf/llms/glm-4.7.yaml` |
| minimax-m2.1 | MiniMax | `conf/llms/minimax-m2.1.yaml` |
| kimi-k2-thinking | Moonshot | `conf/llms/kimi-k2-thinking.yaml` |
| bedrock-qwen3-32b | AWS Bedrock | `conf/llms/bedrock-qwen3-32b.yaml` |

## Environments

### OED (13 environments)
Standard prediction tasks using `*_direct` configs:

| Environment | Goal | Config |
|-------------|------|--------|
| dugongs_direct | length prediction | Core trio |
| peregrines_direct | population prediction | Core trio |
| lotka_volterra_direct | population dynamics | Core trio |
| hyperbolic_direct | choice prediction | |
| hyperbolic_discount | discount rate | Variant goal |
| location_finding_direct | signal prediction | |
| location_finding_source | source location | Variant goal |
| death_process_direct | infection count | |
| death_process_infection | infection rate | Variant goal |
| irt_direct | correctness | |
| survival_direct | survival | |
| emotion_direct | emotion prediction | |
| morals_direct | moral judgement | |

### Discovery (10 environments)
Communication tasks using `*_naive` configs (scientist explains to naive agent):

- dugongs_direct_naive
- peregrines_direct_naive
- lotka_volterra_direct_naive
- hyperbolic_direct_naive
- location_finding_direct_naive
- death_process_direct_naive
- irt_direct_naive
- survival_direct_naive
- emotion_direct_naive
- morals_direct_naive

## Sweep Categories

### Paper Replication (Production)

Citable, reproducible benchmarks with isolated sweep IDs per experimental condition.

| Sweep | Runs | prior | ppl | Description |
|-------|------|-------|-----|-------------|
| `paper_replication_oed` | 455 | ✓ | ✗ | Standard OED with prior |
| `paper_replication_oed_ppl` | 455 | ✓ | ✓ | Box's Loop with prior |
| `paper_replication_oed_no_prior` | 455 | ✗ | ✗ | OED without prior |
| `paper_replication_oed_ppl_no_prior` | 455 | ✗ | ✓ | Box's Loop without prior |
| `paper_replication_discovery` | 350 | ✓ | ✗ | Communication metric |

**Config:** 7 models × 13 envs × 5 seeds, budgets [0,1,3,5,7,10]

**Total: 2,170 runs**

### Lite Validation (Fast)

Quick validation before expensive runs. 96% compute savings.

| Sweep | Runs | Description |
|-------|------|-------------|
| `paper_replication_lite_oed` | 56 | 7 models × 1 env × 2 seeds × 4 modes |
| `paper_replication_lite_discovery` | 14 | 7 models × 1 env × 2 seeds |

**Environment:** location_finding_direct (OED) / location_finding_direct_naive (Discovery)

**Total: 70 runs**

### Trio (Core 3 Environments)

Focus on the paper's main environments: dugongs, peregrines, lotka_volterra.

| Sweep | Runs | Budgets | Description |
|-------|------|---------|-------------|
| `full_oed_trio` | 105 | [0,1,3,5,7,10] | Full budget curve |
| `trio_oed_budget_0_5` | 105 | [0,5] | Quick validation |
| `trio_discovery_budget10` | 105 | [10] | Discovery on trio |

**Total: 315 runs**

### Full Sweeps (Exploration)

All modes combined in single sweeps. Use for development, not paper submission.

| Sweep | Runs | Method | Description |
|-------|------|--------|-------------|
| `full_oed` | 1,820 | random | 7 × 13 × 5 × 4 modes |
| `full_discovery` | 350 | random | 7 × 10 × 5 |

**Note:** These use `method: random` (no fixed count), so agents sample from the full space. For exhaustive coverage, use the paper_replication sweeps which use `method: grid`.

**Total: up to 2,170 runs**

### Test Sweeps (Minimal)

Pipeline validation and smoke tests.

| Sweep | Runs | Description |
|-------|------|-------------|
| `test_oed` | 56 | Validate OED pipeline |
| `test_discovery` | 14 | Validate discovery pipeline |
| `test_metrics_quick` | 1 | Single run smoke test |

**Total: 71 runs**

## Execution Order

```bash
# 1. Smoke test (1 run)
wandb sweep sweeps/test_metrics_quick.yaml

# 2. Pipeline validation (70 runs)
wandb sweep sweeps/test_oed.yaml
wandb sweep sweeps/test_discovery.yaml

# 3. Model validation (70 runs)
wandb sweep sweeps/paper_replication_lite_oed.yaml
wandb sweep sweeps/paper_replication_lite_discovery.yaml

# 4. Core environments (105 runs)
wandb sweep sweeps/trio_oed_budget_0_5.yaml

# 5. Full paper benchmark (455+ runs each)
wandb sweep sweeps/paper_replication_oed.yaml
wandb sweep sweeps/paper_replication_oed_ppl.yaml
# ... remaining as needed
```

## Usage

### Create and run a sweep

```bash
# Create sweep (returns SWEEP_ID)
wandb sweep sweeps/paper_replication_oed.yaml

# Run agent
uv run wandb agent <SWEEP_ID>

# Run multiple agents in parallel
for i in {1..5}; do
  uv run wandb agent <SWEEP_ID> &
done
```

### Local single run (no sweep)

```bash
uv run python run_experiment.py \
  llms=gpt-4o \
  envs=dugongs_direct \
  exp=oed \
  seed=1 \
  include_prior=true \
  use_ppl=false
```

## Design Decisions

### Why 4 separate paper_replication_oed files?

The 2×2 grid of `include_prior` × `use_ppl` could be a single sweep, but separate files provide:

1. **Provenance** - Each sweep ID maps to one experimental condition
2. **Selective reruns** - Rerun 455 runs, not 1,820
3. **Failure isolation** - Bug in one condition doesn't contaminate others
4. **Cost control** - Budget per condition at sweep level

Use `full_oed.yaml` for exploration; use `paper_replication_*` for citable results.

### Why `*_naive` suffix for discovery?

Discovery mode tests if a "scientist" LLM can explain findings to a "naive" agent. The `*_naive` environment configs set up this two-agent communication task.

### Parameter details

| Parameter | Values | Description |
|-----------|--------|-------------|
| `include_prior` | true/false | Whether to include domain prior in prompt |
| `use_ppl` | true/false | Whether to use Box's Loop (PPL-based refinement) |
| `exp` | oed/discovery | Experiment type |
| `seed` | 1-5 | Random seed for reproducibility |
| `num_experiments` | [0,1,3,5,7,10] | Budget (number of experiments to run) |

## Cost Estimation

| Category | Runs | Est. Cost |
|----------|------|-----------|
| Test sweeps | 71 | ~$5 |
| Lite validation | 70 | ~$10-20 |
| Trio sweeps | 315 | ~$50-100 |
| Paper replication (all 5) | 2,170 | ~$300-600 |
| Full sweeps | 2,170 | ~$300-600 |

Costs vary by model. DeepSeek and GLM are cheapest; GPT-4o is most expensive.
