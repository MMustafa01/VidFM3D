#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --job-name=07_extract_wan
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G                        # the text encoder alone is 22.7 GB and passes through normal memory
#SBATCH --output=logs/slurm/%x-%j.out
# Runs frozen WAN2.1-1.3B ONCE per video and caches the features of one layer:
#   CO3D/FEAT/wan/<cat>/<seq>/feature_t749_layer20.sft   key "feat": (21, 30, 52, 1536), fp16, ~0.1 GB per video
# Layer 20, t=749 and the empty prompt are the repo's settings (features/run_co3d.py, configs/experiment/co3d/wan.yaml).
# Needs the VGGT targets (job 04) and the weights (06). Finished videos are skipped.
#
#   sbatch jobs/07_extract_wan.sh --start 0 --end 2      # test on 2 videos
#   sbatch jobs/07_extract_wan.sh                        # all
#   sbatch --partition=A100devel jobs/07_extract_wan.sh  # 80 GB GPU, if the A40 (48 GB) runs out of memory
source jobs/common.sh

FEAT=vidfm3d/data/CO3D/FEAT
python -m features.run_co3d --vfm wan --subset all \
    --model-id "$DATA_DIR/models/Wan2.1-T2V-1.3B-Diffusers" \
    --prompt "" --output-layers 20 --t 749 "$@"

# run_co3d reports a failed video in its log but still exits normally; surface it:
if grep -E "FAILED|ERROR" "$FEAT/wan/processing-all.log"; then
    echo "!! some videos failed, see $FEAT/wan/processing-all.log"
else
    echo "== no errors"
fi

# check every cached file without loading the data (reads only the file headers)
python - "$FEAT/wan" <<'PY'
import sys
from collections import Counter
from pathlib import Path
from safetensors import safe_open

files = sorted(Path(sys.argv[1]).glob("*/*/feature_t749_layer20.sft"))
kinds = Counter()
for f in files:
    with safe_open(f, framework="pt") as s:
        x = s.get_slice("feat")
        kinds[(tuple(x.get_shape()), x.get_dtype())] += 1
print(f"== {len(files)} cached videos")
for (shape, dtype), n in kinds.items():
    print(f"   {n} x {shape} {dtype}")
PY
du -sh "$FEAT/wan"
