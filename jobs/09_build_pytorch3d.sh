#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --job-name=09_build_pytorch3d
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8                # compiles 8 files at a time
#SBATCH --mem=48G
#SBATCH --output=logs/slurm/%x-%j.out
# Step 2 of 3 for pytorch3d: compile it. Needs no GPU: the CUDA compiler (nvcc) runs on the CPU.
source jobs/common.sh

set +u                                   # module scripts use unset variables
module load GCC/12.3.0 CUDA/12.4.0       # torch 2.5 needs GCC >= 9 (system: 8.5); CUDA must match torch (12.4)
set -u
gcc --version | head -n 1
nvcc --version | tail -n 1

export CC=$(which gcc) CXX=$(which g++)
export CUB_HOME=$CUDA_HOME/include       # header library that ships with CUDA
export FORCE_CUDA=1                      # build the GPU code although this node gives us no GPU
export TORCH_CUDA_ARCH_LIST="8.0;8.6"    # 8.0 = A100, 8.6 = A40
export MAX_JOBS=$SLURM_CPUS_PER_TASK

# build on the node's own disk: thousands of small files are slow on the shared /home
SCRATCH=/tmp; [ -w /local/nvme ] && SCRATCH=/local/nvme
BUILD=$(mktemp -d -p "$SCRATCH")
trap 'rm -rf "$BUILD"' EXIT
export TMPDIR=$BUILD
cp -r "$DATA_DIR/src/pytorch3d" "$BUILD/"

# make a wheel (an installable file) and keep it, so rebuilding the env won't need another compile
mkdir -p "$DATA_DIR/wheels"
pip wheel --no-build-isolation --no-deps --no-cache-dir -w "$DATA_DIR/wheels" "$BUILD/pytorch3d"
pip install --no-deps --no-cache-dir --force-reinstall "$DATA_DIR"/wheels/pytorch3d-*.whl
ls -lh "$DATA_DIR/wheels"

python tools/check_pytorch3d.py          # CPU part only here
