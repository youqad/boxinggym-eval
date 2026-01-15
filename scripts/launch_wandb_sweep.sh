#!/usr/bin/env bash
# Launch W&B Sweep with multiple background agents
# Usage: ./scripts/launch_wandb_sweep.sh [sweep_yaml] [num_agents]
#
# Examples:
#   ./scripts/launch_wandb_sweep.sh                                    # Default: trio_oed_budget_0_5.yaml, 3 agents
#   ./scripts/launch_wandb_sweep.sh sweeps/paper_replication_lite_oed.yaml 4
#   ./scripts/launch_wandb_sweep.sh sweeps/trio_oed_budget_0_5.yaml 6

set -euo pipefail

SWEEP_YAML="${1:-sweeps/trio_oed_budget_0_5.yaml}"
NUM_AGENTS="${2:-3}"

echo "=========================================="
echo "🚀 BoxingGym W&B Sweep Launcher"
echo "=========================================="
echo "📋 Sweep config: $SWEEP_YAML"
echo "👥 Number of agents: $NUM_AGENTS"
echo ""

# Check sweep file exists
if [[ ! -f "$SWEEP_YAML" ]]; then
    echo "❌ Error: Sweep file not found: $SWEEP_YAML"
    exit 1
fi

# Ensure W&B is logged in
if ! wandb status &>/dev/null; then
    echo "⚠️  W&B not logged in. Running 'wandb login'..."
    wandb login
fi

# Verify environment variables
echo "🔐 Checking API keys..."
required_keys=(
    "OPENAI_API_KEY"
    "DEEPSEEK_API_KEY"
    "MINIMAX_API_KEY"
    "ZHIPUAI_API_KEY"
)
missing_keys=()
for key in "${required_keys[@]}"; do
    if [[ -z "${!key:-}" ]]; then
        missing_keys+=("$key")
    else
        echo "  ✓ $key is set"
    fi
done

if [[ ${#missing_keys[@]} -gt 0 ]]; then
    echo ""
    echo "⚠️  Warning: Missing API keys: ${missing_keys[*]}"
    echo "   Some models may fail. Continue? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Create the sweep and capture the sweep ID
echo ""
echo "📊 Creating W&B sweep..."
SWEEP_OUTPUT=$(wandb sweep "$SWEEP_YAML" 2>&1)
echo "$SWEEP_OUTPUT"

# Extract sweep ID (format: entity/project/sweep_id)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oE '[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+/[a-zA-Z0-9]+' | tail -1)

if [[ -z "$SWEEP_ID" ]]; then
    echo "❌ Failed to extract sweep ID from output"
    exit 1
fi

echo ""
echo "✅ Sweep created: $SWEEP_ID"
echo ""

# Create log directory for agent outputs
LOG_DIR="logs/sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "📁 Agent logs: $LOG_DIR"

# Launch background agents
echo ""
echo "🚀 Launching $NUM_AGENTS background agents..."
AGENT_PIDS=()

for i in $(seq 1 "$NUM_AGENTS"); do
    LOG_FILE="$LOG_DIR/agent_${i}.log"
    echo "  Starting agent $i → $LOG_FILE"

    # Launch agent in background with nohup
    nohup uv run wandb agent "$SWEEP_ID" > "$LOG_FILE" 2>&1 &
    AGENT_PIDS+=($!)

    # Small delay between agent starts to avoid race conditions
    sleep 2
done

echo ""
echo "=========================================="
echo "✅ All agents launched!"
echo "=========================================="
echo ""
echo "📊 Sweep dashboard: https://wandb.ai/${SWEEP_ID}"
echo "📁 Agent logs: $LOG_DIR"
echo ""
echo "Agent PIDs: ${AGENT_PIDS[*]}"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_DIR/agent_*.log"
echo ""
echo "To stop all agents:"
echo "  kill ${AGENT_PIDS[*]}"
echo ""
echo "To check agent status:"
echo "  ps aux | grep 'wandb agent'"
echo ""

# Save PIDs for later reference
echo "${AGENT_PIDS[*]}" > "$LOG_DIR/agent_pids.txt"
echo "SWEEP_ID=$SWEEP_ID" > "$LOG_DIR/sweep_info.txt"
echo "Started: $(date)" >> "$LOG_DIR/sweep_info.txt"
echo "Agents: $NUM_AGENTS" >> "$LOG_DIR/sweep_info.txt"
