#!/usr/bin/env bash
#SBATCH --job-name=eig_regret_array
#SBATCH --partition=cocoflops
#SBATCH --account=cocoflops
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --array=1-50                # 50 independent tasks
#SBATCH --output=logs/job_%A_%a.out # one log per array-element

# ---------------------------------------------------------------------
# 1. Load Conda and make sure shell aliases are available
# ---------------------------------------------------------------------
source /scr/$USER/miniconda3/bin/activate  # sets up the conda helpers

shopt -s expand_aliases                    # allow aliases in non-interactive Bash
source ~/.bashrc                           # or ~/.bash_profile, wherever you
                                           # defined  alias boxing-env='conda activate boxing-env'

# ---------------------------------------------------------------------
# 2. Activate the environment via your alias
# ---------------------------------------------------------------------
boxing-env                                  # <-- alias that runs `conda activate boxing-env`

# ---------------------------------------------------------------------
# 3. Compute job-specific parameters from SLURM_ARRAY_TASK_ID
# ---------------------------------------------------------------------
SEED=$((SLURM_ARRAY_TASK_ID % 5 + 1))
MODEL="qwen2.5-32b-instruct"

ID=$SLURM_ARRAY_TASK_ID                    # 1 … 50
ENV_INDEX=$((ID / 10))                     # changes every 10 jobs (0–4)
PRIOR=$((ID % 2))                          # toggles 0/1
EXP_TYPE=$((ID % 2))                       # also toggles 0/1

case $ENV_INDEX in
  0) ENV="hyperbolic_direct" ;;
  1) ENV="location_finding_direct" ;;
  2) ENV="death_process_direct" ;;        # add more as needed
  3) ENV="temporal_discount_direct" ;;
  4) ENV="some_other_env" ;;
esac

[[ $PRIOR      -eq 1 ]] && PRIOR_VALUE="true"      || PRIOR_VALUE="false"
[[ $EXP_TYPE   -eq 1 ]] && EXP_TYPE_VALUE="discovery" || EXP_TYPE_VALUE="oed"

# ---------------------------------------------------------------------
# 4. Launch your Python experiment
# ---------------------------------------------------------------------
python run_eig_regret.py \
    seed=$SEED \
    llms.model_name=$MODEL \
    exp=$EXP_TYPE_VALUE \
    envs=$ENV \
    include_prior=$PRIOR_VALUE \
    num_random=100 \
    redo=true \
    box=false
