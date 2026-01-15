#!/usr/bin/env bash
# Benchmark progress monitor
# Usage: ./monitor_benchmark.sh [log_dir]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Use latest log dir if not specified
if [ $# -eq 0 ]; then
    LOG_DIR=$(find logs -name "benchmark_*" -type d 2>/dev/null | sort -r | head -1)
    if [ -z "$LOG_DIR" ]; then
        echo "No benchmark logs found. Start a benchmark first."
        exit 1
    fi
else
    LOG_DIR="$1"
fi

echo "📊 BoxingGym Benchmark Monitor"
echo "================================"
echo "Log directory: $LOG_DIR"
echo "Press Ctrl+C to exit"
echo ""

while true; do
    clear
    echo "=== BoxingGym Benchmark Progress ==="
    echo "🕐 $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Summary from JSON if available
    if [ -f "$LOG_DIR/summary.json" ]; then
        echo "📋 Summary:"
        uv run python - "$LOG_DIR/summary.json" <<'EOF'
import json
import sys
try:
    with open(sys.argv[1]) as f:
        s = json.load(f)
        print(f"  ✅ Completed: {s.get('completed', 0)}/{s.get('total_tasks', '?')}")
        print(f"  ❌ Failed: {s.get('failed', 0)}")
        print(f"  ⏱️  Timeout: {s.get('timeout', 0)}")
        if 'stats' in s:
            print(f"  ⏰ Mean duration: {s['stats']['mean_duration']:.1f}s")
except: pass
EOF
        echo ""
    fi

    # Running processes
    echo "🔄 Running Processes:"
    RUNNING=$(ps aux | grep "run_experiment.py" | grep -v grep | wc -l | tr -d ' ')
    echo "  Python processes: $RUNNING"
    echo ""

    # Recent results
    echo "📁 Recent Results (last 5):"
    find results -name "*.json" -mmin -60 -type f 2>/dev/null | \
        xargs ls -lh 2>/dev/null | \
        tail -5 | \
        awk '{printf "  %s %s %s\n", $6, $7, $9}' || echo "  No recent results"
    echo ""

    # Log file activity
    echo "📝 Active Logs:"
    find "$LOG_DIR" -name "*.log" -type f -mmin -5 2>/dev/null | \
        head -5 | \
        xargs -I {} basename {} | \
        sed 's/^/  /' || echo "  No active logs"
    echo ""

    # Disk usage
    echo "💾 Disk Usage:"
    echo "  Results: $(du -sh results 2>/dev/null | awk '{print $1}' || echo '0B')"
    echo "  Logs: $(du -sh logs 2>/dev/null | awk '{print $1}' || echo '0B')"
    echo ""

    # Process health
    TOTAL_PROCS=$(ps aux | wc -l)
    echo "🏥 System Health:"
    echo "  Total processes: $TOTAL_PROCS"
    if [ $TOTAL_PROCS -gt 400 ]; then
        echo "  ⚠️  WARNING: High process count!"
    elif [ $TOTAL_PROCS -gt 300 ]; then
        echo "  ⚠️  CAUTION: Elevated process count"
    else
        echo "  ✅ Normal"
    fi
    echo ""

    echo "Refreshing in 30 seconds..."
    sleep 30
done
