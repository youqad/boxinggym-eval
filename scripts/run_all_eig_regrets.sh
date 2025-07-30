#!/bin/bash

# Configuration
SEEDS=(1 2 3 4 5)
LANGUAGE_MODELS=("gpt-4o") # Add other models as needed
EXPERIMENT_TYPES=("oed" "discovery")
NUM_RANDOM=100
REDO=true

# Create a progress tracking directory
PROGRESS_DIR=".progress_tracking"
PROGRESS_FILE="${PROGRESS_DIR}/eig_regret_progress.txt"
LOG_FILE="${PROGRESS_DIR}/eig_regret_run.log"

# Create directories if they don't exist
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

# Define standard environments (work with both prior=true and prior=false)
STANDARD_ENVIRONMENTS=(
   "hyperbolic_direct"
   "hyperbolic_direct_discovery"
   "location_finding_direct"
   "location_finding_direct_discovery"
   "death_process_direct"
   "death_process_direct_discovery"
   "irt_direct"
   "irt_direct_discovery"
   "survival_direct"
   "survival_direct_discovery"
   "dugongs_direct"
   "dugongs_direct_discovery"
   "peregrines_direct"
   "peregrines_direct_discovery"
   "lotka_volterra_direct"
   "lotka_volterra_direct_discovery"
)

# Define environments that only work with prior=true
PRIOR_ONLY_ENVIRONMENTS=(
   "emotion_direct"
   "emotion_direct_discovery"
   "morals_direct"
   "morals_direct_discovery"
)

# Function to run a single EIG regret calculation
run_eig_regret() {
   local seed="$1"
   local model="$2"
   local exp_type="$3"
   local env="$4"
   local include_prior="$5"
   local use_boxloop="$6"
   
   # Create experiment key for tracking
   local experiment_key="eig_regret_${seed}_${model}_${exp_type}_${env}_${include_prior}_boxloop${use_boxloop}"
   
   # If experiment already completed, skip
   if is_completed "$experiment_key"; then
       log "SKIPPING already completed: seed=$seed, model=$model, exp=$exp_type, env=$env, prior=$include_prior, boxloop=$use_boxloop"
       return 0
   fi
   
   # Log start of experiment
   log "STARTING: seed=$seed, model=$model, exp=$exp_type, env=$env, prior=$include_prior, boxloop=$use_boxloop"
   
   # Run the experiment
   python run_eig_regret.py seed=$seed llms.model_name=$model exp=$exp_type envs=$env include_prior=$include_prior num_random=$NUM_RANDOM redo=$REDO box=$use_boxloop
   
   # Check if experiment was successful
   if [ $? -eq 0 ]; then
       mark_completed "$experiment_key"
       log "COMPLETED: seed=$seed, model=$model, exp=$exp_type, env=$env, prior=$include_prior, boxloop=$use_boxloop"
   else
       log "FAILED: seed=$seed, model=$model, exp=$exp_type, env=$env, prior=$include_prior, boxloop=$use_boxloop"
   fi
}

# Calculate total experiments
function count_total_experiments() {
    local model_count=${#LANGUAGE_MODELS[@]}
    local seed_count=${#SEEDS[@]}
    local exp_type_count=${#EXPERIMENT_TYPES[@]}
    
    # Each standard environment runs with both prior=true and prior=false
    # And with both boxloop=true and boxloop=false
    local standard_total=$((model_count * seed_count * exp_type_count * ${#STANDARD_ENVIRONMENTS[@]} * 2 * 2))
    
    # Prior-only environments only run with prior=true
    # But still with both boxloop=true and boxloop=false
    local prior_only_total=$((model_count * seed_count * exp_type_count * ${#PRIOR_ONLY_ENVIRONMENTS[@]} * 1 * 2))
    
    echo $((standard_total + prior_only_total))
}

# Main script execution
log "==========================================================="
log "Starting EIG regret calculations for all models and environments"
log "Seeds: ${SEEDS[*]}"
log "Models: ${LANGUAGE_MODELS[*]}"
log "Experiment types: ${EXPERIMENT_TYPES[*]}"
log "==========================================================="

# Calculate total and completed experiments
total_experiments=$(count_total_experiments)
completed_experiments=$(wc -l < "$PROGRESS_FILE")

log "Total experiments: $total_experiments"
log "Already completed: $completed_experiments"
log "Remaining: $((total_experiments - completed_experiments))"
log "==========================================================="

# Run all combinations
for seed in "${SEEDS[@]}"; do
    for model in "${LANGUAGE_MODELS[@]}"; do
        for exp_type in "${EXPERIMENT_TYPES[@]}"; do
            # Run standard environments with both prior=true and prior=false
            for env in "${STANDARD_ENVIRONMENTS[@]}"; do
                # Run with boxloop=false
                run_eig_regret "$seed" "$model" "$exp_type" "$env" "true" "false"
                run_eig_regret "$seed" "$model" "$exp_type" "$env" "false" "false"
                
                # Run with boxloop=true
                run_eig_regret "$seed" "$model" "$exp_type" "$env" "true" "true"
                run_eig_regret "$seed" "$model" "$exp_type" "$env" "false" "true"
                
                # Update progress
                completed=$(grep -c "" "$PROGRESS_FILE")
                log "Progress: $completed/$total_experiments ($(( completed * 100 / total_experiments ))%)"
            done
            
            # Run prior-only environments (only with prior=true)
            for env in "${PRIOR_ONLY_ENVIRONMENTS[@]}"; do
                # Run with boxloop=false
                run_eig_regret "$seed" "$model" "$exp_type" "$env" "true" "false"
                
                # Run with boxloop=true
                run_eig_regret "$seed" "$model" "$exp_type" "$env" "true" "true"
                
                # Update progress
                completed=$(grep -c "" "$PROGRESS_FILE")
                log "Progress: $completed/$total_experiments ($(( completed * 100 / total_experiments ))%)"
            done
        done
    done
    
    # Log completion of seed
    log "Completed all experiments for seed $seed"
    log "==========================================================="
done

log "All EIG regret calculations completed successfully!"