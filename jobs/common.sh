# Shared setup for every job script. Usage (first line after the #SBATCH block):
#   source jobs/common.sh
# SLURM starts jobs in the directory you ran `sbatch` from, so always submit
# from the repo root:  cd ~/cv_lab/VidFM3D && sbatch jobs/<script>.sh

set -eo pipefail          # stop on the first error / failed pipe
                          # (no `set -u` yet: conda's activate scripts use unset variables)

module purge
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate vidfm3d
set -u                    # from here on, using an unset variable is an error

export DATA_DIR=$HOME/cv_lab/data            # real location of all data (symlinked into the repo)
export HF_HOME=$DATA_DIR/hf_cache            # model weights, pre-downloaded on the login node
export HF_HUB_OFFLINE=1                      # never download inside a job: fail fast if not cached
export WANDB_MODE=offline

echo "== job ${SLURM_JOB_ID:-local} '${SLURM_JOB_NAME:-}' on $(hostname) at $(date)"
echo "== code: $(git log -1 --oneline)"
echo "== free on /home: $(df -h ~ | awk 'NR==2{print $4}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "== no GPU"
