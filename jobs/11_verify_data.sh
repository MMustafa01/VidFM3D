#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --job-name=11_verify_data
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/slurm/%x-%j.out
# Task 5: checks targets, features, split and training samples. No GPU needed.
source jobs/common.sh

python -u -W ignore tools/verify_data.py "$@"
