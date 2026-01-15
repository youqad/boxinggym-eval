#!/bin/bash
# Live progress monitor for MiniMax M2.1 benchmark suite

clear
echo "🥊 MiniMax M2.1 Benchmark - Live Progress Monitor"
echo "============================================================"
echo ""

ENVS="dugongs peregrines irt death_process hyperbolic_temporal_discount morals"
SEEDS="1 2 3"
FILE_GLOB="*MiniMax-M2.1*boxloop*oed*true*.json"

while true; do
    # Count results
    TOTAL=$(find results -name "$FILE_GLOB" -type f 2>/dev/null | wc -l | tr -d ' ')
    ENV_COUNT=$(echo "$ENVS" | wc -w | tr -d ' ')
    SEED_COUNT=$(echo "$SEEDS" | wc -w | tr -d ' ')
    TARGET=$((ENV_COUNT * SEED_COUNT))

    # Calculate percentage
    PCT=$((TOTAL * 100 / TARGET))

    # Progress bar
    FILLED=$((PCT / 2))
    EMPTY=$((50 - FILLED))
    BAR=$(printf '█%.0s' $(seq 1 $FILLED))$(printf '░%.0s' $(seq 1 $EMPTY))

    # Current time
    NOW=$(date "+%H:%M:%S")

    # Move cursor to top
    tput cup 0 0

    echo "🥊 MiniMax M2.1 Benchmark - Live Progress Monitor"
    echo "============================================================"
    echo ""
    echo "📊 Overall Progress: $TOTAL / $TARGET ($PCT%)"
    echo "[$BAR] $PCT%"
    echo ""

    # Count by environment
    echo "🌍 By Environment:"
    for env in $ENVS; do
        COUNT=$(find "results/$env" -name "$FILE_GLOB" -type f 2>/dev/null | wc -l | tr -d ' ')
        if [ $COUNT -gt 0 ]; then
            printf "   %-30s %2d/%d runs\n" "$env:" "$COUNT" "$SEED_COUNT"
        fi
    done
    echo ""

    # Latest activity
    echo "⏱️  Latest Activity (last 5 lines):"
    LOG_FILE="logs/minimax_complete.log"
    if [ ! -f "$LOG_FILE" ]; then
        LOG_FILE=$(ls -t logs/minimax/*.log 2>/dev/null | head -1)
    fi
    if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
        tail -5 "$LOG_FILE" 2>/dev/null | sed 's/^/   /'
    else
        echo "   (no minimax logs found yet)"
    fi
    echo ""

    # Current time and ETA
    echo "🕐 Current Time: $NOW"

    # Calculate ETA (rough estimate: 10 min per experiment)
    REMAINING=$((TARGET - TOTAL))
    ETA_MIN=$((REMAINING * 10))
    ETA_HOUR=$((ETA_MIN / 60))
    ETA_MIN_REMAINDER=$((ETA_MIN % 60))

    if [ $REMAINING -gt 0 ]; then
        echo "⏳ Estimated Time Remaining: ${ETA_HOUR}h ${ETA_MIN_REMAINDER}m"
        # Calculate completion time
        ETA_TIME=$(date -v+${ETA_MIN}M "+%H:%M")
        echo "🎯 Estimated Completion: $ETA_TIME"
    else
        echo "✅ ALL EXPERIMENTS COMPLETE!"
    fi

    echo ""
    echo "Press Ctrl+C to exit"
    echo "============================================================"

    sleep 10
done
