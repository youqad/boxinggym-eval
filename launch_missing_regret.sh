#!/usr/bin/env bash
###############################################################################
# launch_missing_regret.sh  (v3)
#
# Find missing regret JSONs named
#   results/<env_dir>/regret_direct_<model>_oed_True_<seed>.json
# and submit one Slurm array (via llm_eig.sh) for exactly those jobs.
#
# Usage:
#   ACCOUNT=myacct PARTITION=gpua ./launch_missing_regret.sh  /path/to/results
#   (or just ./launch_missing_regret.sh  to scan ./results with defaults)
###############################################################################
set -euo pipefail

RESULTS_DIR=${1:-results}
ACCOUNT=${ACCOUNT:-cocoflops}
PARTITION=${PARTITION:-cocoflops}
PRIOR_STR="True"                # file uses “True”, matches include_prior=true

# ---------------------------------------------------------------------------
# Grids  --- MUST match llm_eig.sh
# ---------------------------------------------------------------------------
MODELS=( "qwen2.5-7b-instruct" "OpenThinker-7B" "OpenThinker-32B"
         "claude-3-7-sonnet-20250219" )
SEEDS=(1 2 3 4 5)

ENV_DIRS=( "hyperbolic_temporal_discount" "location_finding" "death_process"
           "irt" "survival" "dugongs" "peregrines" )
ENV_NAMES=( "hyperbolic_direct" "location_finding_direct" "death_process_direct"
            "irt_direct" "survival_direct" "dugongs_direct" "peregrines_direct" )

NUM_M=${#MODELS[@]} ; NUM_E=${#ENV_NAMES[@]} ; NUM_S=${#SEEDS[@]}

pad(){ printf "%-38s" "$1"; }

echo "Scanning '${RESULTS_DIR}' for regret_direct_<model>_oed_${PRIOR_STR}_<seed>.json …"
echo "$(pad Environment/Model) Seed  Status   File"
echo "----------------------------------------------------------------------------"

missing_idx=()        # indices to launch

for ((e=0; e<NUM_E; e++)); do
  env_dir=${ENV_DIRS[$e]}
  for ((m=0; m<NUM_M; m++)); do
    model=${MODELS[$m]}
    for ((s=0; s<NUM_S; s++)); do
      seed=${SEEDS[$s]}
      file="${RESULTS_DIR}/${env_dir}/regret_direct_${model}_oed_${PRIOR_STR}_${seed}.json"

      if [[ -f "$file" ]]; then
        echo "$(pad ${env_dir}/${model}) $seed   FOUND ✔︎ $file"
      else
        echo "$(pad ${env_dir}/${model}) $seed   MISSING ✘ $file"
        idx=$(( s*NUM_M*NUM_E + e*NUM_M + m ))   # same formula as llm_eig.sh
        missing_idx+=( "$idx" )
      fi
    done
  done
done

echo
if [[ ${#missing_idx[@]} -eq 0 ]]; then
  echo "✅  All expected regret files exist. Nothing to submit."
  exit 0
fi

# ---------------------------------------------------------------------------
# Build a compact Slurm array spec: 0,2,5-7,12 …
# ---------------------------------------------------------------------------
IFS=$'\n' sorted=($(sort -n <<<"${missing_idx[*]}")); unset IFS
array_spec=""
start=${sorted[0]} ; prev=$start
for idx in "${sorted[@]:1}"; do
  if (( idx == prev+1 )); then
       prev=$idx
  else
       [[ -n $array_spec ]] && array_spec+=","
       [[ $start -eq $prev ]] && array_spec+="$start" || array_spec+="$start-$prev"
       start=$prev=$idx
  fi
done
[[ -n $array_spec ]] && array_spec+=","
[[ $start -eq $prev ]] && array_spec+="$start" || array_spec+="$start-$prev"

echo "🚀  Submitting 1 Slurm array for ${#missing_idx[@]} missing jobs:"
echo "    --array=${array_spec}  (account=${ACCOUNT}, partition=${PARTITION})"

sbatch --account="$ACCOUNT" --partition="$PARTITION" --array="$array_spec" llm_eig.sh
