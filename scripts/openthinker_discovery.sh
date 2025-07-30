#!/bin/bash

# Configuration
SEEDS=(1 2 3 4 5)
MODEL="openthinker"
EXPERIMENT="discovery"

# Set environment variables to connect to the VLLM server on coco-hgx
export VLLM_API_BASE="http://cocoflops-hgx-1:8000/v1"
export VLLM_API_KEY="token-abc123"
export VLLM_MODEL_NAME="open-thoughts/OpenThinker-32B"  # Adjust based on the model loaded

# Create a progress tracking directory (separate from experiment results)
PROGRESS_DIR=".progress_tracking"
PROGRESS_FILE="${PROGRESS_DIR}/${MODEL}_discovery_progress.txt"
LOG_FILE="${PROGRESS_DIR}/${MODEL}_discovery_run.log"

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
log "Starting DISCOVERY experiments with OpenThinker model"
log "Seeds: ${SEEDS[*]}"
log "Using VLLM server at: $VLLM_API_BASE"
log "Model: $VLLM_MODEL_NAME"
log "=============================================="

# Check connection to VLLM server before starting
log "Testing connection to VLLM server..."
if curl -s "$VLLM_API_BASE/models" > /dev/null; then
   log "Successfully connected to VLLM server"
else
   log "ERROR: Failed to connect to VLLM server at $VLLM_API_BASE"
   log "Please check if the server is running and network connectivity"
   exit 1
fi

# Define environments that work with both prior=true and prior=false
STANDARD_ENVIRONMENTS=(
   "hyperbolic_direct_discovery"
   "location_finding_direct_discovery"
   "death_process_direct_discovery"
   "irt_direct_discovery"
   "survival_direct_discovery"
   "dugongs_direct_discovery"
   "peregrines_direct_discovery"
   "lotka_volterra_direct_discovery"
)

# Define environments that only work with prior=true
PRIOR_ONLY_ENVIRONMENTS=(
   "emotion_direct_discovery"
   "morals_direct_discovery"
)

# Calculate total experiments
standard_count=$((${#SEEDS[@]} * ${#STANDARD_ENVIRONMENTS[@]} * 2)) # Both prior=true and prior=false
prior_only_count=$((${#SEEDS[@]} * ${#PRIOR_ONLY_ENVIRONMENTS[@]})) # Only prior=true
total_experiments=$((standard_count + prior_only_count))

# Calculate already completed
completed_experiments=$(wc -l < "$PROGRESS_FILE")

log "Total experiments: $total_experiments"
log "Already completed: $completed_experiments"
log "Remaining: $((total_experiments - completed_experiments))"
log "=============================================="

# Run all experiments
for seed in "${SEEDS[@]}"; do
   log "Starting seed $seed"
   
   # Run standard environments with prior=true
   log "Running standard environments with prior=true for seed $seed"
   for env in "${STANDARD_ENVIRONMENTS[@]}"; do
       run_single_experiment "$seed" "true" "$env"
       
       # Update progress after each experiment
       completed=$(grep -c "" "$PROGRESS_FILE")
       log "Progress: $completed/$total_experiments ($(( completed * 100 / total_experiments ))%)"
   done
   
   # Run standard environments with prior=false
   log "Running standard environments with prior=false for seed $seed"
   for env in "${STANDARD_ENVIRONMENTS[@]}"; do
       run_single_experiment "$seed" "false" "$env"
       
       # Update progress after each experiment
       completed=$(grep -c "" "$PROGRESS_FILE")
       log "Progress: $completed/$total_experiments ($(( completed * 100 / total_experiments ))%)"
   done
   
   # Run prior-only environments (only with prior=true)
   log "Running prior-only environments (emotion & morals) with prior=true for seed $seed"
   for env in "${PRIOR_ONLY_ENVIRONMENTS[@]}"; do
       run_single_experiment "$seed" "true" "$env"
       
       # Update progress after each experiment
       completed=$(grep -c "" "$PROGRESS_FILE")
       log "Progress: $completed/$total_experiments ($(( completed * 100 / total_experiments ))%)"
   done
   
   # Log completion of seed
   log "Completed all experiments for seed $seed"
   log "=============================================="
done

log "All experiments completed successfully!"