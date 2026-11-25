#!/bin/bash
#SBATCH --partition=A100devel
#SBATCH --job-name=05_extract_features
#SBATCH --time=0:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm/%x-%j.out
# Runs the frozen DINOv2-small backbone ONCE per video and caches its features, so that
# every probe training run can read them from disk instead of running the backbone again:
#   CO3D/FEAT/dino/<cat>/<seq>/feature.sft   key "feat": (81, 30, 52, 384), fp16, ~0.1 GB per video
# Needs the VGGT targets (job 04) first. Finished videos are skipped, so it's safe to resubmit.
#
# Careful: the cache path has no model name in it, and the skip check doesn't compare models.
# For another DINO size, pass a different folder, e.g.:
#   sbatch jobs/05_extract_features.sh --model-id facebook/dinov2-base --out-root vidfm3d/data/CO3D/FEAT_dinov2b
# (then the checks below still look at FEAT, and training needs the matching root_vfm)
#
#   sbatch jobs/05_extract_features.sh --start 0 --end 2     # test on 2 videos
#   sbatch jobs/05_extract_features.sh                       # all
source jobs/common.sh

FEAT=vidfm3d/data/CO3D/FEAT
python -m features.run_co3d --vfm dino --subset all --batch-size 64 --model-id facebook/dinov2-small "$@"

# run_co3d reports a failed video in its log but still exits normally; surface it:
if grep -E "FAILED|ERROR" "$FEAT/dino/processing-all.log"; then
    echo "!! some videos failed, see $FEAT/dino/processing-all.log"
else
    echo "== no errors"
fi

# check every cached file without loading the data (reads only the file headers)
python - "$FEAT/dino" <<'PY'
import sys
from collections import Counter
from pathlib import Path
from safetensors import safe_open

files = sorted(Path(sys.argv[1]).glob("*/*/feature.sft"))
kinds = Counter()
for f in files:
    with safe_open(f, framework="pt") as s:
        x = s.get_slice("feat")
        kinds[(tuple(x.get_shape()), x.get_dtype())] += 1
print(f"== {len(files)} cached videos")
for (shape, dtype), n in kinds.items():
    print(f"   {n} x {shape} {dtype}")
PY
du -sh "$FEAT"
