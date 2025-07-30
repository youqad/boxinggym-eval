#!/usr/bin/env bash
#SBATCH --job-name=llm-eig-miss20
#SBATCH --partition=cocoflops
#SBATCH --account=cocoflops
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --array=0-19%20
#SBATCH --output=logs_missing/llm_eig_miss20_%A_%a.log

set -euo pipefail
mkdir -p logs_missing

# ─────────────────────────────────────────────────────────────────────────────
# 0) Resolve index (Slurm array or local CLI arg)
# ─────────────────────────────────────────────────────────────────────────────
IDX=${SLURM_ARRAY_TASK_ID:-${1:-}}
if [[ -z $IDX ]]; then
  echo "Usage:"
  echo "  sbatch llm_missing_remaining.sh          # full 20-job array"
  echo "  ./llm_missing_remaining.sh <index>       # run one job locally (0–19)"
  exit 1
fi
shift || true; set --

# ─────────────────────────────────────────────────────────────────────────────
# 1) Fixed grid of 20 missing combos
# ─────────────────────────────────────────────────────────────────────────────
ENVIRONMENTS=( irt dugongs survival peregrines )
SEEDS=( 1 2 3 4 5 )
MODEL="qwen2.5-32b-instruct"
PRIOR="False"

# derive env & seed from linear index: idx = env_idx*5 + (seed-1)
ENV_IDX=$(( IDX / 5 ))
SEED=${SEEDS[$(( IDX % 5 ))]}
ENV=${ENVIRONMENTS[$ENV_IDX]}

# Hydra config name for each env
declare -A CFG_MAP=(
  [irt]=irt_direct
  [dugongs]=dugongs_direct
  [survival]=survival_direct
  [peregrines]=peregrines_direct
)
CFG=${CFG_MAP[$ENV]}

echo "▶ IDX=$IDX → env=$ENV  cfg=$CFG  seed=$SEED  prior=$PRIOR"

# ─────────────────────────────────────────────────────────────────────────────
# 2) Activate conda (argument-safe)
# ─────────────────────────────────────────────────────────────────────────────
source /scr/$USER/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate boxing-gym-env || { echo "❌  cannot activate env"; exit 1; }

export PYTENSOR_FLAGS="compiledir=/tmp/pytensor_${IDX}"

# ─────────────────────────────────────────────────────────────────────────────
# 3) Run experiment (direct goal)
# ─────────────────────────────────────────────────────────────────────────────
python run_eig_regret.py \
    seed="$SEED" \
    llms.model_name="$MODEL" \
    exp=oed \
    envs="$CFG" \
    include_prior="$PRIOR" \
    num_random=100 \
    redo=true \
    box=false
