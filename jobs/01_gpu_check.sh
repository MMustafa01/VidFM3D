#!/bin/bash
#SBATCH --partition=A100devel
#SBATCH --job-name=01_gpu_check
#SBATCH --time=0:10:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/slurm/%x-%j.out
source jobs/common.sh

python tools/gpu_check.py
