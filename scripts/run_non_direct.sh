#!/bin/bash
#SBATCH --job-name=gpt4o-non-direct
#SBATCH --partition=cocoflops
#SBATCH --account=cocoflops
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --array=0-19%10  # Reduced to 18 jobs (removing prior=false where not applicable)
#SBATCH --output=logs/gpt4o_non_direct_%A_%a.log

# Create logs directory if it doesn't exist
mkdir -p logs
export PYTHONPATH="/sailhome/agam/boxing-gym/src:${PYTHONPATH}"
# Load your conda environment
source /scr/$USER/miniconda3/bin/activate
conda activate boxing-gym-env

# Configuration
SEEDS=(1 2 3 4 5)
MODEL="gpt-4o"
EXPERIMENTS=("oed")

# Define environment-goal pairs with non-direct goals
# Format: "environment config_name requires_prior"
ENV_GOAL_PAIRS=(
    "hyperbolic_temporal_discount hyperbolic_discount true"
    "location_finding location_finding_source true"
    "death_process death_process_infection true"
)

# Create a progress tracking directory (separate from experiment results)
PROGRESS_DIR=".progress_tracking"
PROGRESS_FILE="${PROGRESS_DIR}/${MODEL}_non_direct_goals_progress.txt"
LOG_FILE="${PROGRESS_DIR}/${MODEL}_non_direct_goals_run.log"

# Create progress directory if it doesn't exist
mkdir -p "$PROGRESS_DIR"

# Initialize or read progress file
if [ ! -f "$PROGRESS_FILE" ]; then
    touch "$PROGRESS_FILE"
fi

# Calculate parameters for this job based on array index
IDX=$SLURM_ARRAY_TASK_ID

# Calculate total combinations per experiment type
# Now each env-goal pair has only one prior setting (true)
COMBS_PER_EXP=$((${#SEEDS[@]} * ${#ENV_GOAL_PAIRS[@]}))  # seeds × env-goal pairs

# Determine which experiment type to run
EXP_IDX=$((IDX / COMBS_PER_EXP))
EXPERIMENT=${EXPERIMENTS[$EXP_IDX]}

# Calculate index within experiment type
LOCAL_IDX=$((IDX % COMBS_PER_EXP))

# Determine seed index
SEED_IDX=$((LOCAL_IDX / ${#ENV_GOAL_PAIRS[@]}))
SEED=${SEEDS[$SEED_IDX]}

# Determine environment-goal pair
ENV_GOAL_IDX=$((LOCAL_IDX % ${#ENV_GOAL_PAIRS[@]}))
ENV_GOAL=${ENV_GOAL_PAIRS[$ENV_GOAL_IDX]}

# Split the environment, config name, and prior requirement
read -r ENV CONFIG_NAME REQUIRES_PRIOR <<< "$ENV_GOAL"

# For all these non-direct goals, we'll always use prior=true
PRIOR="true"
GOAL=$(echo "$CONFIG_NAME" | sed -e "s/${ENV}_//g" -e "s/${ENV}//g")
if [ -z "$GOAL" ]; then
    # If goal extraction failed, use the config name as fallback
    GOAL="$CONFIG_NAME"
fi

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
echo "Goal: $GOAL"
echo "Config Name: $CONFIG_NAME"
echo "=========================="

# Create experiment key for tracking
experiment_key="${SEED}_${MODEL}_${PRIOR}_${ENV}_${GOAL}_${EXPERIMENT}"

# If experiment already completed, skip
if is_completed "$experiment_key"; then
    log "SKIPPING already completed: seed=$SEED, prior=$PRIOR, env=$ENV, goal=$GOAL, experiment=$EXPERIMENT"
    exit 0
fi

# Log start of experiment
log "STARTING: seed=$SEED, prior=$PRIOR, env=$ENV, goal=$GOAL, experiment=$EXPERIMENT"

# Run the experiment using Hydra's config system with the correct config name
python run_experiment.py seed=$SEED llms.model_name=$MODEL include_prior=$PRIOR exp.experiment_type=$EXPERIMENT envs=$CONFIG_NAME

# Check if experiment was successful
if [ $? -eq 0 ]; then
    mark_completed "$experiment_key"
    log "COMPLETED: seed=$SEED, prior=$PRIOR, env=$ENV, goal=$GOAL, experiment=$EXPERIMENT"
else
    log "FAILED: seed=$SEED, prior=$PRIOR, env=$ENV, goal=$GOAL, experiment=$EXPERIMENT"
fi

# Update progress information
completed=$(grep -c "" "$PROGRESS_FILE" || echo 0)
total_experiments=$((${#SEEDS[@]} * ${#ENV_GOAL_PAIRS[@]} * ${#EXPERIMENTS[@]}))
if [ "$total_experiments" -eq 0 ]; then
    log "Progress: $completed/$total_experiments (0%)"
else
    log "Progress: $completed/$total_experiments ($(( completed * 100 / total_experiments ))%)"
fi