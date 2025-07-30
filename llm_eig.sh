#!/usr/bin/env bash
###############################################################################
# llm_eig.sh  —  Slurm array for prior-only OED regret runs
#
# Grid:
#   4  models  ×  7  environments  ×  5  seeds  =  140  jobs
#   (include_prior is hard-wired to true)
###############################################################################
#SBATCH --job-name=llm-eig
#SBATCH --partition=cocoflops
#SBATCH --account=cocoflops
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --array=0-250%50                 # 140 tasks, max 20 running at once
#SBATCH --output=logs/llm_eig_%A_%a.log

# --------------------------------------------------------------------------- #
# 0.  Strict error handling & per-task setup
# --------------------------------------------------------------------------- #
set -euo pipefail
IDX=${SLURM_ARRAY_TASK_ID}

mkdir -p logs .progress_tracking
export PYTENSOR_FLAGS="compiledir=/tmp/pytensor_${IDX},compile.timeout=1800"

# --------------------------------------------------------------------------- #
# 1.  Activate conda env (abort if python missing)
# --------------------------------------------------------------------------- #
source /scr/$USER/miniconda3/bin/activate
conda deactivate 2>/dev/null || true
conda activate boxing-gym-env || { echo "❌ Cannot activate boxing-gym-env"; exit 1; }
command -v python >/dev/null || { echo "❌ python not on PATH"; exit 127; }

# --------------------------------------------------------------------------- #
# 2.  Experiment grid
# --------------------------------------------------------------------------- #
MODELS=( "qwen2.5-7b-instruct" "OpenThinker-7B"
         "OpenThinker-32B" "claude-3-7-sonnet-20250219" )

ENVIRONMENTS=( "hyperbolic_direct" "location_finding_direct" "death_process_direct"
               "irt_direct"        "survival_direct"        "dugongs_direct"
               "peregrines_direct" )

SEEDS=(3 4 5 1 2)

NUM_M=${#MODELS[@]}     # 4
NUM_E=${#ENVIRONMENTS[@]} # 7
NUM_S=${#SEEDS[@]}      # 5
TOTAL=$((NUM_M*NUM_E*NUM_S))  # 140

# --------------------------------------------------------------------------- #
# 3.  Decode array index  →  (model, env, seed)
# --------------------------------------------------------------------------- #
(( IDX < TOTAL )) || { echo "⚠️  Index $IDX ≥ $TOTAL — nothing to do"; exit 0; }

seed_idx=$(( IDX / (NUM_M*NUM_E) ))           # 0‥4
env_idx=$(( (IDX / NUM_M) % NUM_E ))          # 0‥6
model_idx=$(( IDX % NUM_M ))                  # 0‥3

SEED=${SEEDS[$seed_idx]}
ENV=${ENVIRONMENTS[$env_idx]}
MODEL=${MODELS[$model_idx]}
PRIOR=true
EXP=oed
NUM_RANDOM=100
REDO=true
BOX=false

# --------------------------------------------------------------------------- #
# 4.  Logging
# --------------------------------------------------------------------------- #
MODEL_DIR="logs/${MODEL}"
mkdir -p "$MODEL_DIR"
LOGFILE="$MODEL_DIR/run_${IDX}.log"
STATUSFILE="$MODEL_DIR/status_${IDX}.txt"

{
echo "=== JOB CONFIGURATION ==="
echo "Array ID     : $IDX / $TOTAL"
echo "Model        : $MODEL"
echo "Environment  : $ENV"
echo "Seed         : $SEED"
echo "Prior        : $PRIOR"
echo "Experiment   : $EXP"
echo "========================="
echo "Start: $(date '+%F %T')"
} | tee  "$LOGFILE"

# --------------------------------------------------------------------------- #
# 5.  Run the experiment
# --------------------------------------------------------------------------- #
python run_eig_regret.py \
      seed=$SEED \
      llms.model_name=$MODEL \
      exp=$EXP \
      envs=$ENV \
      include_prior=$PRIOR \
      num_random=$NUM_RANDOM \
      redo=$REDO \
      box=$BOX 2>&1 | tee -a "$LOGFILE"

EXIT=$?
echo "End  : $(date '+%F %T')"          | tee -a "$LOGFILE"
echo "Exit : $EXIT"                    | tee -a "$LOGFILE"

# Write compact status line for dashboards / grep
printf "RUN_ID=%s_%s_seed%s_prior%s,EXIT=%d,STATUS=%s,START=%s,END=%s\n" \
       "$MODEL" "${ENV/_direct/}" "$SEED" "$PRIOR" \
       "$EXIT"  "$([ $EXIT -eq 0 ] && echo SUCCESS || echo FAILED)" \
       "$(grep '^Start:' "$LOGFILE" | cut -d' ' -f2-)" \
       "$(grep '^End  :'  "$LOGFILE" | cut -d' ' -f2-)" \
       > "$STATUSFILE"

exit "$EXIT"
