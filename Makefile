.PHONY: all venv deps aggregate plot clean bench-fast bench

VENV := .venv
PY := $(VENV)/bin/python
UV := uv

CSV := outputs/standardized_results.csv
PLOTS := outputs/plots

all: aggregate plot

venv:
	@$(UV) venv $(VENV) >/dev/null

deps: venv
	@($(UV) pip sync --python $(PY) requirements.txt >/dev/null) || echo "uv pip sync failed; falling back to editable install"
	@$(UV) pip install --python $(PY) -e . >/dev/null

aggregate: deps
	@$(UV) run python scripts/aggregate_results.py --out $(CSV) --skip-reproducibility-check

plot: deps
	@$(UV) run python scripts/plot_results.py --csv $(CSV) --outdir $(PLOTS)

clean:
	@rm -rf $(CSV) $(PLOTS)

bench-fast: deps
	@BOXINGGYM_FAST_ENV=1 $(UV) run python scripts/run_comparative_benchmarks.py \
		--models deepseek/deepseek-chat gpt-5-mini \
		--envs dugongs \
		--runs 1 \
		--fast-env

# Full benchmark suite (5 models × 3 envs × 3 seeds)
# Uses scripts/run_full_benchmark.sh
bench: deps
	@bash scripts/run_full_benchmark.sh
