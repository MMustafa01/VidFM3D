#!/bin/bash
#SBATCH --partition=A100devel
#SBATCH --job-name=04_vggt_targets
#SBATCH --time=1:00:00                 # the A100devel maximum; finished videos are skipped, so just resubmit if it runs out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm/%x-%j.out
# Runs VGGT-1B on every accepted video -> CO3D-processed/<cat>/<seq>.sft (the probe's targets).
# Extra arguments are passed through, so test on 2 videos first:
#   sbatch jobs/04_vggt_targets.sh --start 0 --end 2
#   sbatch jobs/04_vggt_targets.sh                     # then all
source jobs/common.sh

python -m vidfm3d.data.processing.process_co3d --root vidfm3d/data/CO3D "$@"
# process_co3d hides per-video errors in its log instead of failing; surface them:
grep -c "ERROR" vidfm3d/data/CO3D/CO3D-processed/processing-all.log && echo "!! some videos failed, see the log" || echo "== no errors"
ls vidfm3d/data/CO3D/CO3D-processed/*/*.sft | wc -l
du -sh vidfm3d/data/CO3D/CO3D-processed
