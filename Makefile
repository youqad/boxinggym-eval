.PHONY: all venv deps aggregate plot clean bench-fast bench

VENV := .venv
PY := $(VENV)/bin/python
UV := uv

all: aggregate plot

venv:
	@$(UV) venv $(VENV) >/dev/null

deps: venv
	@($(UV) pip sync --python $(PY) requirements.txt >/dev/null) || echo "uv pip sync failed; falling back to editable install"
	@$(UV) pip install --python $(PY) -e . >/dev/null

# was: aggregate_results.py
aggregate: deps
	@$(UV) run box sync --local results/

# was: plot_results.py (try: box results --tui)
plot: deps
	@$(UV) run box query all

# clear cache and outputs
clean:
	@rm -rf outputs/ .boxing-gym-cache/

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
