#!/usr/bin/env bash
###############################################################################
# monitor_eig.sh  –  live dashboard for llm-eig Slurm array
###############################################################################
set -euo pipefail

LLM_SCRIPT="llm_eig.sh"     # path to the submission script
REFRESH=30                  # seconds between refreshes

# --------------------------------------------------------------------------- #
# Colours
# --------------------------------------------------------------------------- #
B='\033[0;34m'; G='\033[0;32m'; R='\033[0;31m'
Y='\033[0;33m'; C='\033[0;36m'; M='\033[0;35m'; N='\033[0m'

# --------------------------------------------------------------------------- #
# 1.  Extract MODELS / ENVIRONMENTS / SEEDS arrays *only*
# --------------------------------------------------------------------------- #
extract_array() { # $1 = NAME
  awk -v pat="^[[:space:]]*($1[[:space:]]*=|declare[[:space:]]+-a[[:space:]]+$1[[:space:]]*=)" '
      $0 ~ pat { in_arr=1 }
      in_arr     { print }
      in_arr && /\)/ { in_arr=0 }
  ' "$LLM_SCRIPT"
}

arrays=$(extract_array MODELS)
arrays+="
$(extract_array ENVIRONMENTS)
$(extract_array SEEDS)
"

if [[ -z $arrays ]]; then
  echo "❌  Could not find MODELS / ENVIRONMENTS / SEEDS in $LLM_SCRIPT" >&2
  exit 1
fi

# shellcheck disable=SC1090
eval "$arrays"           # now MODELS[], ENVIRONMENTS[], SEEDS[] are defined

NUM_M=${#MODELS[@]} ; NUM_E=${#ENVIRONMENTS[@]} ; NUM_S=${#SEEDS[@]}
TOTAL_EXPECTED=$(( NUM_M * NUM_E * NUM_S ))

# --------------------------------------------------------------------------- #
# 2.  Helpers
# --------------------------------------------------------------------------- #
progress_bar () {  # $1 current, $2 total
  local W=50 filled=$(( $1 * W / $2 ))
  printf "["
  for ((i=0;i<W;i++)); do
    (( i < filled )) && printf "█" || printf "░"
  done
  printf "]"
}

show() {
  clear
  echo -e "${B}====================  LLM-EIG DASHBOARD  ====================${N}"
  echo -e "Updated: $(date)\n"

  # ---- queue summary ----
  RUN=$(squeue -u "$USER" -n llm-eig -h -t RUNNING | wc -l)
  PEND=$(squeue -u "$USER" -n llm-eig -h -t PENDING | wc -l)
  TOT=$(squeue -u "$USER" -n llm-eig -h           | wc -l)
  echo -e "${B}SLURM QUEUE${N}"
  echo -e "Running : ${G}$RUN${N}"
  echo -e "Pending : ${Y}$PEND${N}"
  echo -e "Total   : $TOT\n"

  # ---- completed / failed ----
  COMP=0; FAIL=0
  for M in "${MODELS[@]}"; do
    [[ -d logs/$M ]] || continue
    COMP=$(( COMP + $(grep -l "SUCCESS" logs/$M/status_*.txt 2>/dev/null | wc -l) ))
    FAIL=$(( FAIL + $(grep -l "FAILED"  logs/$M/status_*.txt 2>/dev/null | wc -l) ))
  done
  PERC=$(( COMP * 100 / TOTAL_EXPECTED ))
  echo -e "${B}OVERALL PROGRESS${N}"
  echo -e "Succeeded: ${G}$COMP${N} / $TOTAL_EXPECTED  |  Failed: ${R}$FAIL${N}"
  progress_bar "$COMP" "$TOTAL_EXPECTED"; echo -e "  ${C}$PERC %${N}\n"

  # ---- by seed ----
  echo -e "${B}BY SEED${N}"
  for S in "${SEEDS[@]}"; do
    DONE=0
    for M in "${MODELS[@]}"; do
      DONE=$(( DONE + $(grep -l "_seed${S}_" logs/$M/status_*.txt 2>/dev/null | grep -c SUCCESS || true) ))
    done
    TOTAL=$(( NUM_E * NUM_M ))
    printf "Seed %-2s : " "$S"; progress_bar "$DONE" "$TOTAL"
    echo -e " ${C}$DONE/$TOTAL${N}"
  done
  echo

  # ---- by environment ----
  echo -e "${B}BY ENVIRONMENT${N}"
  for env in "${ENVIRONMENTS[@]}"; do
    DONE=0
    for M in "${MODELS[@]}"; do
      DONE=$(( DONE + $(grep -l "${env}_seed.*SUCCESS" logs/$M/status_*.txt 2>/dev/null | wc -l) ))
    done
    TOTAL=$(( NUM_S * NUM_M ))
    printf "%-18s : " "$env"; progress_bar "$DONE" "$TOTAL"
    echo -e " ${C}$DONE/$TOTAL${N}"
  done
  echo

  # ---- by model ----
  echo -e "${B}BY MODEL (✓ / ✗)${N}"
  for M in "${MODELS[@]}"; do
    RUN_M=$(squeue -u "$USER" -n llm-eig -h -t RUNNING | grep -c "$M" || true)
    PEND_M=$(squeue -u "$USER" -n llm-eig -h -t PENDING | grep -c "$M" || true)
    OK=$(grep -l "SUCCESS" logs/$M/status_*.txt 2>/dev/null | wc -l || true)
    BAD=$(grep -l "FAILED"  logs/$M/status_*.txt 2>/dev/null | wc -l || true)
    [[ $RUN_M -gt 0 ]]  && STAT="${G}Running:${RUN_M}${N}" \
      || [[ $PEND_M -gt 0 ]] && STAT="${Y}Pending:${PEND_M}${N}" \
      || STAT="${C}Idle${N}"
    echo -e "${M}: $STAT | ✓ ${G}$OK${N} ✗ ${R}$BAD${N}"
  done
  echo
}

# --------------------------------------------------------------------------- #
# 3.  Main loop
# --------------------------------------------------------------------------- #
while true; do
  show
  sleep "$REFRESH"
done
