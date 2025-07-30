#!/bin/bash
#SBATCH --job-name=claude-exps-array
#SBATCH --partition=cocoflops
#SBATCH --account=cocoflops
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --array=0-199%10
#SBATCH --output=logs/claude_exps_%A_%a.log

# Create logs directory if it doesn't exist
mkdir -p logs

cd /sailhome/agam/boxing-gym

# 2. Activate the right Conda env
source /scr/agam/miniconda3/bin/activate
conda activate boxing-gym-env

# 3. Make sure Python can see the source tree
export PYTHONPATH="/sailhome/agam/boxing-gym/src:${PYTHONPATH}"

# 4. (optional) echo for sanity
echo "PYTHONPATH=$PYTHONPATH"
echo "PWD=$(pwd)"
# Configuration
SEEDS=(1 2 3 4 5)
MODEL="claude"
EXPERIMENTS=("oed")

# Define environments
ENVIRONMENTS=(
    "irt_direct"
    "survival_direct"
    "peregrines_direct"
    "emotion_direct"
    "morals_direct"
    "lotka_volterra_direct"
    "death_process_direct"
    "hyperbolic_direct"
    "location_finding_direct"
    "dugongs_direct"
)

# Create a progress tracking directory (separate from experiment results)
PROGRESS_DIR=".progress_tracking"
PROGRESS_FILE="${PROGRESS_DIR}/${MODEL}_experiments_progress.txt"
LOG_FILE="${PROGRESS_DIR}/${MODEL}_experiments_run.log"

# Create progress directory if it doesn't exist
mkdir -p "$PROGRESS_DIR"

# Initialize or read progress file
if [ ! -f "$PROGRESS_FILE" ]; then
    touch "$PROGRESS_FILE"
fi

# Calculate parameters for this job based on array index
IDX=$SLURM_ARRAY_TASK_ID

# Calculate total combinations per experiment type
COMBS_PER_EXP=$((${#SEEDS[@]} * 2 * ${#ENVIRONMENTS[@]}))  # seeds × prior settings × environments

# Determine which experiment type to run
EXP_IDX=$((IDX / COMBS_PER_EXP))
EXPERIMENT=${EXPERIMENTS[$EXP_IDX]}

# Calculate index within experiment type
LOCAL_IDX=$((IDX % COMBS_PER_EXP))

# Determine seed
SEED_IDX=$((LOCAL_IDX / (2 * ${#ENVIRONMENTS[@]})))
SEED=${SEEDS[$SEED_IDX]}

# Determine prior setting
PRIOR_IDX=$(((LOCAL_IDX / ${#ENVIRONMENTS[@]}) % 2))
if [ $PRIOR_IDX -eq 0 ]; then
    PRIOR="true"
else
    PRIOR="false"
fi

# Determine environment
ENV_IDX=$((LOCAL_IDX % ${#ENVIRONMENTS[@]}))
ENV=${ENVIRONMENTS[$ENV_IDX]}

# Function to log progress
log() {
    local message="$1"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[${timestamp}] $message" | tee -a "$LOG_FILE"
}

# Function to check if an experiment is already completed
is_completed() {
    local experiment_key="$1"
    grep -q "^${experiment_key}$" "$PROGRESS_FILE"
    return $?
}

# Function to mark an experiment as completed
mark_completed() {
    local experiment_key="$1"
    echo "$experiment_key" >> "$PROGRESS_FILE"
}

# Log job information
echo "=== JOB CONFIGURATION ==="
echo "Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Experiment Type: $EXPERIMENT"
echo "Seed: $SEED"
echo "Prior: $PRIOR"
echo "Environment: $ENV"
echo "Working directory: $(pwd)"
echo "Conda Environment: $(conda info --envs | grep '*')"
echo "=========================="

# Create experiment key for tracking
experiment_key="${SEED}_${MODEL}_${PRIOR}_${ENV}_${EXPERIMENT}"

# Check if environment is emotion_direct or morals_direct and prior is false
if [[ "$ENV" == "emotion_direct" || "$ENV" == "morals_direct" ]] && [[ "$PRIOR" == "false" ]]; then
    log "SKIPPING incompatible: seed=$SEED, prior=$PRIOR, env=$ENV, exp=$EXPERIMENT (requires prior=true)"
    exit 0
fi

# If experiment already completed, skip
if is_completed "$experiment_key"; then
    log "SKIPPING already completed: seed=$SEED, prior=$PRIOR, env=$ENV, exp=$EXPERIMENT"
    exit 0
fi

log "STARTING: seed=$SEED, prior=$PRIOR, env=$ENV, exp=$EXPERIMENT"

# Run the experiment directly
python /sailhome/agam/boxing-gym/run_experiment.py seed=$SEED llms=$MODEL include_prior=$PRIOR exp=$EXPERIMENT envs=$ENV

# Check if experiment was successful
exit_code=$?
if [ $exit_code -eq 0 ]; then
    mark_completed "$experiment_key"
    log "COMPLETED: seed=$SEED, prior=$PRIOR, env=$ENV, exp=$EXPERIMENT"
else
    log "FAILED: seed=$SEED, prior=$PRIOR, env=$ENV, exp=$EXPERIMENT (exit code: $exit_code)"
fi

# FIXED: Calculate progress with simplified arithmetic
standard_envs=8
prior_only_envs=2
total_exps=180  # Manually computed: 5 seeds × 2 exp types × (2×8 + 2) envs

# Get completion count safely
completed=0
if [ -f "$PROGRESS_FILE" ]; then
    completed=$(cat "$PROGRESS_FILE" | wc -l)
fi

# Fixed progress calculation - no variable within calculation
if [ "$completed" -gt 0 ]; then
    # Add 1 to denominator to avoid division by zero
    percent=$(( 100 * completed / total_exps ))
    log "Progress: $completed/$total_exps ($percent%)"
else
    log "Progress: 0/$total_exps (0%)"
fi