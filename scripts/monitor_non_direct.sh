#!/bin/bash

# Monitoring script for GPT-4o non-direct goal experiments
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
MODEL="gpt-4o"

# Progress directory
PROGRESS_DIR=".progress_tracking"
PROGRESS_FILE="${PROGRESS_DIR}/${MODEL}_non_direct_goals_progress.txt"
LOG_FILE="${PROGRESS_DIR}/${MODEL}_non_direct_goals_run.log"

# Define environment-goal pairs with non-direct goals
ENV_GOAL_PAIRS=(
    "hyperbolic_temporal_discount discount"
    "location_finding source"
    "death_process infection"
)

# Print header
echo -e "${BLUE}===================================================${NC}"
echo -e "${BLUE}      GPT-4o NON-DIRECT GOAL EXPERIMENTS MONITOR   ${NC}"
echo -e "${BLUE}===================================================${NC}"
echo -e "Time: $(date)"
echo ""

# Job status summary
echo -e "${BLUE}JOB STATUS SUMMARY${NC}"
echo -e "${BLUE}------------------${NC}"

# Count running jobs
RUNNING_JOBS=$(squeue -u $USER -n gpt4o-non-direct -t RUNNING | grep -v JOBID | wc -l)
PENDING_JOBS=$(squeue -u $USER -n gpt4o-non-direct -t PENDING | grep -v JOBID | wc -l)
TOTAL_JOBS=$(squeue -u $USER -n gpt4o-non-direct | grep -v JOBID | wc -l)

echo -e "Running: ${GREEN}$RUNNING_JOBS${NC}"
echo -e "Pending: ${YELLOW}$PENDING_JOBS${NC}"
echo -e "Total:   $TOTAL_JOBS"

# Progress from tracking file
if [ -f "$PROGRESS_FILE" ]; then
    COMPLETED=$(wc -l < "$PROGRESS_FILE")
    # Note: Now only with prior=true (5 seeds × 3 env-goal pairs × 2 experiment types)
    TOTAL=30  
    PERCENT=$((COMPLETED * 100 / TOTAL))
    
    echo -e "Overall progress: ${GREEN}$COMPLETED/$TOTAL${NC} (${GREEN}$PERCENT%${NC})"
    
    # Show detailed progress for each environment-goal pair
    echo ""
    echo -e "${BLUE}DETAILED PROGRESS BY ENVIRONMENT-GOAL PAIR${NC}"
    echo -e "${BLUE}---------------------------------------${NC}"
    
    for pair in "${ENV_GOAL_PAIRS[@]}"; do
        read -r env goal <<< "$pair"
        count=$(grep -c "_${env}_${goal}_" "$PROGRESS_FILE")
        # 5 seeds × 2 experiment types (always prior=true)
        pair_total=10
        pair_percent=$((count * 100 / pair_total))
        
        echo -e "${env} - ${goal}: ${GREEN}$count/$pair_total${NC} (${GREEN}$pair_percent%${NC})"
    done
else
    echo -e "${RED}No progress file found${NC}"
fi

# Check log directory
echo ""
echo -e "${BLUE}LOG DIRECTORY STATUS${NC}"
echo -e "${BLUE}-------------------${NC}"
NUM_LOGS=$(find logs -name "gpt4o_non_direct_*.log" 2>/dev/null | wc -l)
if [ $NUM_LOGS -gt 0 ]; then
    echo -e "Log files found: ${GREEN}$NUM_LOGS${NC}"
    
    # Show most recent logs
    echo ""
    echo -e "${BLUE}RECENT LOG ACTIVITY${NC}"
    echo -e "${BLUE}------------------${NC}"
    find logs -name "gpt4o_non_direct_*.log" -mmin -60 2>/dev/null | head -5 | while read log; do
        echo -e "${GREEN}$log${NC}: $(tail -1 $log)"
    done
else
    echo -e "${RED}No log files found in logs directory${NC}"
fi

# Show recent completions
echo ""
echo -e "${BLUE}RECENTLY COMPLETED JOBS${NC}"
echo -e "${BLUE}----------------------${NC}"
COMPLETED_JOBS=$(sacct -u $USER -n gpt4o-non-direct -S $(date -d "1 day ago" +"%Y-%m-%d") -s COMPLETED -o JobID,JobName,State,Elapsed,End | grep -v "RUNNING\|PENDING" | head -5)
if [ -n "$COMPLETED_JOBS" ]; then
    echo "$COMPLETED_JOBS"
else
    echo "No recently completed jobs"
fi

# Errors and failures
echo ""
echo -e "${BLUE}RECENT ERRORS & FAILURES${NC}"
echo -e "${BLUE}-----------------------${NC}"
FAILED_JOBS=$(sacct -u $USER -n gpt4o-non-direct -S $(date -d "1 day ago" +"%Y-%m-%d") -s FAILED,TIMEOUT,NODE_FAIL -o JobID,JobName,State,Elapsed,End | head -5)
if [ -n "$FAILED_JOBS" ]; then
    echo -e "${RED}$FAILED_JOBS${NC}"
    
    # Show error details from logs
    echo ""
    echo -e "${BLUE}ERROR DETAILS FROM LOGS${NC}"
    echo -e "${BLUE}---------------------${NC}"
    find logs -name "gpt4o_non_direct_*.log" -mmin -60 2>/dev/null | xargs grep -l "Error\|error\|ERROR\|Exception\|Failed" | head -3 | while read log; do
        echo -e "${RED}$log${NC}:"
        grep -A 2 "Error\|error\|ERROR\|Exception\|Failed" $log | head -3
        echo ""
    done
else
    echo -e "${GREEN}No recent job failures${NC}"
fi

# Check config files
echo ""
echo -e "${BLUE}AVAILABLE CONFIGURATION FILES${NC}"
echo -e "${BLUE}----------------------------${NC}"

CONFIG_FILES=(
    "hyperbolic_discount"
    "location_finding_source"
    "death_process_infection"
)

for config in "${CONFIG_FILES[@]}"; do
    if [ -f "conf/envs/${config}.yaml" ]; then
        echo -e "${GREEN}${config}.yaml${NC}: Found"
    else
        echo -e "${RED}${config}.yaml${NC}: Not found"
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
echo "  - Check a specific log: ${GREEN}cat logs/gpt4o_non_direct_*.log${NC}"
echo "  - Run this monitor again: ${GREEN}./monitor_gpt4o_non_direct_updated.sh${NC}"
echo -e "${BLUE}===================================================${NC}"