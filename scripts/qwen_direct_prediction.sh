# SEEDS=(1 2 3 4 5)
# for seed in "${SEEDS[@]}"; do
#     # OED w/ prior = True
#     python run_experiment.py seed=$seed llms=qwen include_prior=true exp=oed envs=irt_direct 
#     python run_experiment.py seed=$seed llms=qwen include_prior=true exp=oed envs=survival_direct 
#     python run_experiment.py seed=$seed llms=qwen include_prior=true exp=oed envs=peregrines_direct 
#     python run_experiment.py seed=$seed llms=qwen include_prior=true exp=oed envs=emotion_direct 
#     python run_experiment.py seed=$seed llms=qwen include_prior=true exp=oed envs=morals_direct 
#     python run_experiment.py seed=$seed llms=qwen include_prior=true exp=oed envs=lotka_volterra_direct 
#     python run_experiment.py seed=$seed llms=qwen include_prior=true exp=oed envs=death_process_direct
#     python run_experiment.py seed=$seed llms=qwen include_prior=true exp=oed envs=hyperbolic_direct
#     python run_experiment.py seed=$seed llms=qwen include_prior=true exp=oed envs=location_finding_direct
#     python run_experiment.py seed=$seed llms=qwen include_prior=true exp=oed envs=dugongs_direct
#     # OED w/ prior=False
#     python run_experiment.py seed=$seed llms=qwen include_prior=false exp=oed envs=irt_direct 
#     python run_experiment.py seed=$seed llms=qwen include_prior=false exp=oed envs=survival_direct 
#     python run_experiment.py seed=$seed llms=qwen include_prior=false exp=oed envs=peregrines_direct 
#     python run_experiment.py seed=$seed llms=qwen include_prior=false exp=oed envs=emotion_direct 
#     python run_experiment.py seed=$seed llms=qwen include_prior=false exp=oed envs=morals_direct 
#     python run_experiment.py seed=$seed llms=qwen include_prior=false exp=oed envs=lotka_volterra_direct 
#     python run_experiment.py seed=$seed llms=qwen include_prior=false exp=oed envs=death_process_direct
#     python run_experiment.py seed=$seed llms=qwen include_prior=false exp=oed envs=hyperbolic_direct
#     python run_experiment.py seed=$seed llms=qwen include_prior=false exp=oed envs=location_finding_direct
#     python run_experiment.py seed=$seed llms=qwen include_prior=false exp=oed envs=dugongs_direct
# done 

#!/bin/bash

# Configuration
SEEDS=(1 2 3 4 5)
MODEL="qwen"
EXPERIMENT="oed"

# Create a progress tracking directory (separate from experiment results)
PROGRESS_DIR=".progress_tracking"
PROGRESS_FILE="${PROGRESS_DIR}/oed_progress.txt"
LOG_FILE="${PROGRESS_DIR}/oed_run.log"

# Create progress directory if it doesn't exist
mkdir -p "$PROGRESS_DIR"

# Initialize or read progress file
if [ ! -f "$PROGRESS_FILE" ]; then
    touch "$PROGRESS_FILE"
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

# Function to run a single experiment with tracking
run_single_experiment() {
    local seed="$1"
    local include_prior="$2"
    local env="$3"
    
    # Create experiment key for tracking
    local experiment_key="${seed}_${MODEL}_${include_prior}_${env}"
    
    # If experiment already completed, skip
    if is_completed "$experiment_key"; then
        log "SKIPPING already completed: seed=$seed, prior=$include_prior, env=$env"
        return 0
    fi
    
    # Log start of experiment
    log "STARTING: seed=$seed, prior=$include_prior, env=$env"
    
    # Run the experiment
    python run_experiment.py seed=$seed llms=$MODEL include_prior=$include_prior exp=$EXPERIMENT envs=$env
    
    # Check if experiment was successful
    if [ $? -eq 0 ]; then
        mark_completed "$experiment_key"
        log "COMPLETED: seed=$seed, prior=$include_prior, env=$env"
    else
        log "FAILED: seed=$seed, prior=$include_prior, env=$env"
    fi
}

# Main script execution
log "=============================================="
log "Starting OED experiments with Qwen model"
log "Seeds: ${SEEDS[*]}"
log "=============================================="

# Calculate total experiments and already completed
total_experiments=$((${#SEEDS[@]} * 2 * 10))  # seeds × prior settings × environments
completed_experiments=$(wc -l < "$PROGRESS_FILE")

log "Total experiments: $total_experiments"
log "Already completed: $completed_experiments"
log "Remaining: $((total_experiments - completed_experiments))"
log "=============================================="

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

# Run all experiments
for seed in "${SEEDS[@]}"; do
    log "Starting seed $seed"
    
    # Run with prior=true
    log "Running seed $seed with prior=true"
    for env in "${ENVIRONMENTS[@]}"; do
        run_single_experiment "$seed" "true" "$env"
        
        # Update progress after each experiment
        completed=$(grep -c "" "$PROGRESS_FILE")
        log "Progress: $completed/$total_experiments ($(( completed * 100 / total_experiments ))%)"
    done
    
    # Run with prior=false
    log "Running seed $seed with prior=false"
    for env in "${ENVIRONMENTS[@]}"; do
        run_single_experiment "$seed" "false" "$env"
        
        # Update progress after each experiment
        completed=$(grep -c "" "$PROGRESS_FILE")
        log "Progress: $completed/$total_experiments ($(( completed * 100 / total_experiments ))%)"
    done
    
    # Log completion of seed
    log "Completed all experiments for seed $seed"
    log "=============================================="
done

log "All experiments completed successfully!"
