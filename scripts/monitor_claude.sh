#!/bin/bash

# Monitoring script for Claude experiments
# Creates a dashboard of current job status

# Terminal colors for better readability
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Clear screen
clear

# Model name
MODEL="claude"

# Progress directory
PROGRESS_DIR=".progress_tracking"
PROGRESS_FILE="${PROGRESS_DIR}/${MODEL}_experiments_progress.txt"
LOG_FILE="${PROGRESS_DIR}/${MODEL}_experiments_run.log"

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

# Print header
echo -e "${BLUE}===================================================${NC}"
echo -e "${BLUE}         CLAUDE EXPERIMENTS MONITOR               ${NC}"
echo -e "${BLUE}===================================================${NC}"
echo -e "Time: $(date)"
echo ""

# Job status summary
echo -e "${BLUE}JOB STATUS SUMMARY${NC}"
echo -e "${BLUE}------------------${NC}"

# Count running jobs
RUNNING_JOBS=$(squeue -u $USER -n claude-exps-array -t RUNNING | grep -v JOBID | wc -l)
PENDING_JOBS=$(squeue -u $USER -n claude-exps-array -t PENDING | grep -v JOBID | wc -l)
TOTAL_JOBS=$(squeue -u $USER -n claude-exps-array | grep -v JOBID | wc -l)

echo -e "Running: ${GREEN}$RUNNING_JOBS${NC}"
echo -e "Pending: ${YELLOW}$PENDING_JOBS${NC}"
echo -e "Total:   $TOTAL_JOBS"

# Progress from tracking file
if [ -f "$PROGRESS_FILE" ]; then
    COMPLETED=$(wc -l < "$PROGRESS_FILE")
    standard_envs=8  # Number of envs that work with both prior=true and prior=false
    prior_only_envs=2  # Number of envs that only work with prior=true
    TOTAL=$((5 * 2 * (2 * standard_envs + prior_only_envs)))  # 5 seeds × 2 exp types × (2 prior settings × standard envs + prior-only envs)
    PERCENT=$((COMPLETED * 100 / TOTAL))
    
    echo -e "Overall progress: ${GREEN}$COMPLETED/$TOTAL${NC} (${GREEN}$PERCENT%${NC})"
    
    # Show progress by experiment type
    echo ""
    echo -e "${BLUE}PROGRESS BY EXPERIMENT TYPE${NC}"
    echo -e "${BLUE}-------------------------${NC}"
    
    for exp_type in "oed" "discovery"; do
        count=$(grep -c "_${exp_type}$" "$PROGRESS_FILE")
        exp_total=$((TOTAL / 2))
        exp_percent=$((count * 100 / exp_total))
        echo -e "${exp_type}: ${GREEN}$count/$exp_total${NC} (${GREEN}$exp_percent%${NC})"
    done
    
    # Show detailed progress for each environment
    echo ""
    echo -e "${BLUE}DETAILED PROGRESS BY ENVIRONMENT${NC}"
    echo -e "${BLUE}------------------------------${NC}"
    
    for env in "${ENVIRONMENTS[@]}"; do
        if [[ "$env" == "emotion_direct" || "$env" == "morals_direct" ]]; then
            # These only work with prior=true
            count=$(grep -c "_${env}_" "$PROGRESS_FILE")
            env_total=10  # 5 seeds × 2 exp types × 1 prior setting
        else
            count=$(grep -c "_${env}_" "$PROGRESS_FILE")
            env_total=20  # 5 seeds × 2 exp types × 2 prior settings
        fi
        env_percent=$((count * 100 / env_total))
        
        echo -e "${env}: ${GREEN}$count/$env_total${NC} (${GREEN}$env_percent%${NC})"
    done
else
    echo -e "${RED}No progress file found${NC}"
fi

# Check log directory
echo ""
echo -e "${BLUE}LOG DIRECTORY STATUS${NC}"
echo -e "${BLUE}-------------------${NC}"
NUM_LOGS=$(find logs -name "claude_exps_*.log" 2>/dev/null | wc -l)
if [ $NUM_LOGS -gt 0 ]; then
    echo -e "Log files found: ${GREEN}$NUM_LOGS${NC}"
    
    # Show most recent logs
    echo ""
    echo -e "${BLUE}RECENT LOG ACTIVITY${NC}"
    echo -e "${BLUE}------------------${NC}"
    find logs -name "claude_exps_*.log" -mmin -60 2>/dev/null | head -5 | while read log; do
        echo -e "${GREEN}$log${NC}: $(tail -1 $log)"
    done
else
    echo -e "${RED}No log files found in logs directory${NC}"
fi

# Show recent completions
echo ""
echo -e "${BLUE}RECENTLY COMPLETED JOBS${NC}"
echo -e "${BLUE}----------------------${NC}"
COMPLETED_JOBS=$(sacct -u $USER -n claude-exps-array -S $(date -d "1 day ago" +"%Y-%m-%d") -s COMPLETED -o JobID,JobName,State,Elapsed,End | grep -v "RUNNING\|PENDING" | head -5)
if [ -n "$COMPLETED_JOBS" ]; then
    echo "$COMPLETED_JOBS"
else
    echo "No recently completed jobs"
fi

# Errors and failures
echo ""
echo -e "${BLUE}RECENT ERRORS & FAILURES${NC}"
echo -e "${BLUE}-----------------------${NC}"
FAILED_JOBS=$(sacct -u $USER -n claude-exps-array -S $(date -d "1 day ago" +"%Y-%m-%d") -s FAILED,TIMEOUT,NODE_FAIL -o JobID,JobName,State,Elapsed,End | head -5)
if [ -n "$FAILED_JOBS" ]; then
    echo -e "${RED}$FAILED_JOBS${NC}"
    
    # Show error details from logs
    echo ""
    echo -e "${BLUE}ERROR DETAILS FROM LOGS${NC}"
    echo -e "${BLUE}---------------------${NC}"
    find logs -name "claude_exps_*.log" -mmin -60 2>/dev/null | xargs grep -l "Error\|error\|ERROR\|Exception\|Failed" | head -3 | while read log; do
        echo -e "${RED}$log${NC}:"
        grep -A 2 "Error\|error\|ERROR\|Exception\|Failed" $log | head -3
        echo ""
    done
else
    echo -e "${GREEN}No recent job failures${NC}"
fi

# Check result files 
echo ""
echo -e "${BLUE}RESULT FILES STATUS${NC}"
echo -e "${BLUE}------------------${NC}"

for env in "${ENVIRONMENTS[@]}"; do
    ENV_BASE=$(echo $env | cut -d '_' -f 1)
    result_dir="results/${ENV_BASE}"
    
    if [ -d "$result_dir" ]; then
        files=$(ls -la $result_dir/*claude*${env}* 2>/dev/null | wc -l)
        echo -e "${env}: ${GREEN}$files${NC} result files"
    else
        echo -e "${env}: ${RED}No result directory${NC}"
    fi
done

# Cluster status
echo ""
echo -e "${BLUE}CLUSTER STATUS${NC}"
echo -e "${BLUE}--------------${NC}"
echo "Showing summary of cocoflops resources..."
pestat -G -p cocoflops | head -10

# Final instructions
echo ""
echo -e "${BLUE}===================================================${NC}"
echo -e "${YELLOW}USEFUL COMMANDS:${NC}"
echo "  - View all jobs: ${GREEN}squeue -u \$USER${NC}"
echo "  - Cancel a job: ${GREEN}scancel JOB_ID${NC}"
echo "  - View detailed job info: ${GREEN}sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,Start,End,NodeList${NC}"
echo "  - Check a specific log: ${GREEN}cat logs/claude_exps_*.log${NC}"
echo -e "${BLUE}===================================================${NC}"