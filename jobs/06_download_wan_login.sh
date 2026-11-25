#!/bin/bash
# Download WAN2.1-T2V-1.3B (~29 GB, Apache-2.0). Run on the LOGIN node, ideally inside tmux:
#   bash jobs/06_download_wan_login.sh
#
# Goes into a normal folder instead of the HuggingFace cache: diffusers 0.39 can't load sharded
# weights by model name without internet, but it can load them from a local path.
# Safe to re-run: finished files are skipped.
set -eo pipefail
cd "$(dirname "$0")/.."

module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate vidfm3d
set -u

DATA_DIR=$(dirname "$(pwd -P)")/data
export WAN_DIR=$DATA_DIR/models/Wan2.1-T2V-1.3B-Diffusers   # the name must contain "T2V": the extractor checks for it
mkdir -p "$WAN_DIR"

echo "== your quota (space used / limit):"; quota -s | tail -n 1
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", local_dir=os.environ["WAN_DIR"],
                  allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*",
                                  "text_encoder/*", "transformer/*", "vae/*"])
PY
du -sh "$WAN_DIR"/*/
echo "== total: $(du -sh "$WAN_DIR" | cut -f1) (expected: about 27G as shown by du, which is 29 GB)"
