#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --job-name=13_depth_only
#SBATCH --time=2:00:00                   # the 3-head run took ~44 min on an A100; 2 h leaves room for the test
#SBATCH --nodes=1
#SBATCH --ntasks=1                       # the repo's Lightning setup reads SLURM_NTASKS, so it must be set
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/%x-%j.out
# Depth-only probe on WAN2.1-1.3B features: the paper's probe with ONLY the depth head.
#
#   sbatch jobs/13_train_depth_only.sh            # seed 42, compare with the 3-head run "v1" (also seed 42)
#   sbatch jobs/13_train_depth_only.sh 43         # other seeds, compare with "s43" / "s44"
#
# Everything else is the same as the 3-head WAN runs (jobs/12, 20 epochs, each video 20 times per epoch,
# batch 10 on one A100, seed 42 by default), so the only difference is the heads:
#   - active_heads=[depth]: the point and camera heads are not built at all, so only the depth loss is trained.
#     The repo checks for each output before using it, so the point and camera losses, the aligned point
#     error, the camera AUCs and the .ply point clouds are simply not computed or logged.
#   - compare: val/loss_depth (same function as in the 3-head runs) and the depth images.
#     val/loss is now the depth loss alone, so it is NOT comparable with val/loss of the 3-head runs.
#
# Output: logs/co3d/runs/co3d_wan_depth_s<SEED>/csv (curves), logs/co3d-eval/runs/co3d-eval_wan_depth_s<SEED> (test)
# Disk: one last.ckpt (smaller than the 3-head ~4.3 GB, the point and camera heads are missing). Check the quota first.

SEED=${1:-42}

exec bash jobs/12_train_probe.sh wan depth_s$SEED 20 20 \
    seed=$SEED \
    "model.probe.active_heads=[depth]" \
    data.data_module.batch_size_per_device=10 \
    data.data_module.batch_size_per_device_val=10 \
    ++trainer.accumulate_grad_batches=1
