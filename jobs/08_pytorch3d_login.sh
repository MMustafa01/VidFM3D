#!/bin/bash
# Step 1 of 3 for pytorch3d: everything that needs internet. Run on the LOGIN node (~1-2 min):
#   bash jobs/08_pytorch3d_login.sh
set -eo pipefail
cd "$(dirname "$0")/.."

module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate vidfm3d
set -u
SRC=$(dirname "$(pwd -P)")/data/src/pytorch3d

# pytorch3d gets compiled with GCC 12, whose programs need a newer C++ library (libstdc++) than
# Bender's system has. Put one into the env. --freeze-installed: don't touch any other package.
conda install -n vidfm3d -c conda-forge --override-channels --freeze-installed -y libstdcxx

# runtime dependency of pytorch3d; the torch entries are a guard so pip can't replace torch
pip install iopath ninja "torch==2.5.1" "torchvision==0.20.1"

# V0.7.8 is the version the README's "@stable" points to
[ -d "$SRC" ] || git clone --depth 1 --branch V0.7.8 https://github.com/facebookresearch/pytorch3d.git "$SRC"

python -c "import torch; print('torch', torch.__version__)"
ls "$SRC/setup.py" && echo "== ready: now run  sbatch jobs/09_build_pytorch3d.sh"
