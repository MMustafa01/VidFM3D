#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --job-name=12_train_probe
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1                       # the repo's Lightning setup reads SLURM_NTASKS, so it must be set
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/%x-%j.out
# Task 7: train the paper's probe on cached features, then test it.
#
#   sbatch jobs/12_train_probe.sh <dino|wan> <run name> <epochs> <repeat> [more Hydra overrides]
#
#   sbatch jobs/12_train_probe.sh dino smoke 2 2          # does it run at all? 2 epochs of 5 steps
#   sbatch jobs/12_train_probe.sh dino v1 20 20           # 20 epochs of 52 steps = 1040 steps
#
# Submitting the same command again resumes from the last checkpoint (the repo's autoresume),
# so a job that hits the time limit isn't lost. A changed setting needs a NEW run name.
source jobs/common.sh

MODEL=${1:?usage: sbatch jobs/12_train_probe.sh dino|wan RUN EPOCHS REPEAT}
RUN=${2:?run name missing}
EPOCHS=${3:?number of epochs missing}
REPEAT=${4:?repeat missing}
shift 4

# everything the two experiments have in common in the repo's configs
COMMON="root='vidfm3d/data/CO3D/CO3D-processed', root_vfm='vidfm3d/data/CO3D/FEAT', subset='all', feat_pixalign=True, num_views=4, min_view_interval=5, context_len=76, query_idx_divisor=4"
case $MODEL in
    dino) DATA="$COMMON, vfm_name='dino', feat_postfix='', fold_stride=1"; CHANNELS=384 ;;   # DINOv2-small (repo default: large, 1024)
    wan)  DATA="$COMMON, vfm_name='wan', feat_postfix='_t749_layer20'";    CHANNELS=1536 ;;  # WAN 1.3B
    *)    echo "unknown model '$MODEL' (use dino or wan)"; exit 1 ;;
esac

# The paper's config uses batch size 10, which needs more than the A40's 48 GB. Two batches of 5 whose gradients
# are added up before each update use much less memory and train on (almost) the same 10 samples per step.
# On an A100 (80 GB) the original can be tried: add  data.data_module.batch_size_per_device=10 ++trainer.accumulate_grad_batches=1
BATCH=5
ACCUMULATE=2

WARMUP=$(( (EPOCHS * 10 + 74) / 75 ))    # same share of warmup as the paper's config (10 of 75 epochs), at least 1

ARGS=(
    experiment=co3d/$MODEL
    job_name=${MODEL}_$RUN
    model.probe.video_channels=$CHANNELS
    trainer.devices=1
    trainer.max_epochs=$EPOCHS
    model.scheduler.warmup_epochs=$WARMUP
    data.data_module.batch_size_per_device=$BATCH
    data.data_module.batch_size_per_device_val=$BATCH
    ++trainer.accumulate_grad_batches=$ACCUMULATE
    # one checkpoint is ~4.6 GB (382 M parameters + optimizer), so keep only the last one, which is what we test
    callbacks.model_checkpoint.save_top_k=0
    data.data_module.num_workers=8
    data.data_module.num_workers_val=2
    # 26 videos with batch size 10 are only 2 steps per epoch. "N * dataset" (built into the repo) goes through
    # them N times per epoch; every pass picks 4 new random frames per video.
    "data.data_module.train_datasets=[\"$REPEAT * VideoProbeDataset($DATA, split='train')\"]"
    # seed=1 instead of 0: the code treats 0 as "no seed", which would change the test frames on every read
    "data.data_module.validation_datasets=[\"VideoProbeDataset($DATA, split='val', seed=1)\"]"
    # write metrics to a CSV file instead of wandb (no internet, no account needed)
    "~logger.wandb"
    +logger.csv._target_=lightning.pytorch.loggers.CSVLogger
    "+logger.csv.save_dir=\${paths.output_dir}"
    +logger.csv.name=csv
    "$@"
)

echo "== training $MODEL: $EPOCHS epochs (warmup $WARMUP), each video $REPEAT times per epoch"
python -W ignore vidfm3d/train.py "${ARGS[@]}"

# test the LAST checkpoint, not the "best" one: "best" is chosen on the test videos (finding F1)
CKPT=logs/co3d/runs/co3d_${MODEL}_$RUN/checkpoints/last.ckpt
ls -lh "$(dirname "$CKPT")"
echo "== testing $CKPT"
python -W ignore vidfm3d/train.py "${ARGS[@]}" task_name=co3d-eval train=false test=true ckpt_path=$CKPT

echo "== done. Curves: logs/co3d/runs/co3d_${MODEL}_$RUN/csv   Test images: logs/co3d-eval/runs/co3d-eval_${MODEL}_$RUN/viz"
