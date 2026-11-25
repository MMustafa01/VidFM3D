#!/bin/bash
#SBATCH --partition=A100devel
#SBATCH --job-name=03_extract_frames
#SBATCH --time=0:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6               # extract_frames.py uses a pool of 6 processes
#SBATCH --output=logs/slurm/%x-%j.out
# CPU-only work; --gpus=1 only because the partition may require it (check `sinfo -s` for a CPU partition).
source jobs/common.sh

python -m vidfm3d.data.processing.co3d.extract_frames \
    --raw_root vidfm3d/data/CO3D/CO3D-data \
    --out_root vidfm3d/data/CO3D/CO3D-raw \
    --stride 1 --num_frames 81 --trunc_thresh 0.25 --resize_to 960 540

echo "== accepted videos per category:"
for d in vidfm3d/data/CO3D/CO3D-raw/*/; do echo "   $(basename "$d"): $(find "$d" -mindepth 1 -maxdepth 1 -type d | wc -l)"; done
du -sh vidfm3d/data/CO3D/CO3D-raw
