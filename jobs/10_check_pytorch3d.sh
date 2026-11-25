#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --job-name=10_check_pytorch3d
#SBATCH --time=0:10:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/slurm/%x-%j.out
# Step 3 of 3 for pytorch3d: test it on a GPU and import the training code.
source jobs/common.sh

python tools/check_pytorch3d.py --require-gpu
