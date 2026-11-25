#!/bin/bash
# Create the `vidfm3d` conda env WITHOUT the full CUDA toolkit.
#
# Run this on the LOGIN node (bash jobs/00_setup_env_login.sh), NOT with sbatch:
#  - it only downloads and unpacks packages (no heavy compute),
#  - the package caches go to the login node's local /tmp (355 GB, not the full
#    shared /home), so they don't count against the little free space on /home.
# Only the finished env lands in ~/.conda/envs (compute nodes need to see it).
set -eo pipefail                   # no -u yet: module/conda activate scripts use unset variables
cd "$(dirname "$0")/.."            # repo root, so `pip install -e .` works

ENV=vidfm3d
CACHE=/tmp/$USER/install_cache
export CONDA_PKGS_DIRS=$CACHE/conda_pkgs
export PIP_CACHE_DIR=$CACHE/pip
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"
trap 'rm -rf "$CACHE"' EXIT        # delete the caches even if the script fails

module load Miniforge3
eval "$(conda shell.bash hook)"    # makes `conda activate` work inside a script

echo "free on /home before: $(df -h ~ | tail -1)"

# re-running updates an existing env instead of rebuilding it
if [ -d "$HOME/.conda/envs/$ENV" ]; then echo "== env $ENV exists, updating it"; else conda create -n "$ENV" python=3.11 -y; fi
conda activate "$ENV"
set -u

# pip wheels bundle the CUDA *runtime* libs torch needs; no nvcc/toolkit needed to run.
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-pretasks.txt      # = the repo's requirements.txt + our pins
pip install -e .

# opencv-python (from requirements.txt) is the desktop build and usually needs libGL, which Bender's login node
# doesn't have. Both builds share the cv2 folder, so remove both, then install the headless one.
pip uninstall -y opencv-python opencv-python-headless
pip install --no-deps opencv-python-headless

python -c "import torch, vidfm3d; print('torch', torch.__version__, '| built with CUDA', torch.version.cuda)"

# smoke test: which packages can be imported here?
echo "== needed for our pre-tasks:"
for m in cv2 safetensors.torch transformers lightning.pytorch hydra wandb roma trimesh typeguard pl_bolts.optimizers.lr_scheduler; do
    python -W ignore -c "import $m" > /dev/null 2> "$CACHE/err" && echo "   OK    $m" || echo "   FAIL  $m: $(tail -n 1 "$CACHE/err")"
done
echo "== only needed by other backbones/tools (a FAIL here doesn't block us):"
for m in diffusers deepspeed open3d gradio viser; do
    python -W ignore -c "import $m" > /dev/null 2> "$CACHE/err" && echo "   OK    $m" || echo "   FAIL  $m: $(tail -n 1 "$CACHE/err")"
done
du -sh "$CONDA_PREFIX"
echo "free on /home after:  $(df -h ~ | tail -1)"
