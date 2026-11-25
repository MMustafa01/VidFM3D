#!/bin/bash
# Downloads for the pre-tasks. Run on the LOGIN node, after 00_setup_env_login.sh:
#   bash jobs/02_download_login.sh
#
# 1) The whole CO3Dv2 single-sequence subset (8.65 GB, 88 videos, 48 categories) with the
#    OFFICIAL downloader (it verifies SHA256 checksums). It goes to the login node's local /tmp.
# 2) Only images/ + masks/ of the categories in CATS are copied to /home,
#    because that's all extract_frames.py reads and /home is nearly full.
# 3) Model weights: VGGT-1B (only model.safetensors: the repo also has a second 5 GB
#    copy, model.pt) and DINOv2-small.
#
# Safe to re-run: finished steps are skipped.
set -eo pipefail                              # -u is enabled after conda activate
cd "$(dirname "$0")/.."                       # repo root

module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate vidfm3d                        # the downloader needs requests + tqdm
set -u

DATA_DIR=$HOME/cv_lab/data
RAW=$DATA_DIR/CO3D/CO3D-data
TMP_ROOT=${TMP_ROOT:-/tmp/$USER}
CO3D_REPO=$TMP_ROOT/co3d_repo                 # official code (download script + link list)
DL=$TMP_ROOT/co3d_dl                          # the full subset, unpacked

# Categories copied to /home: 8 categories x 3 videos = 24 videos, 15 pass extract_frames.py's filter.
# All 28 passing videos (of 88) would need ~25 GB on /home once targets + features are added.
CATS=(apple bench toytruck toytrain skateboard remote ball broccoli)

mkdir -p "$RAW" "$DL"
# the repo expects vidfm3d/data/CO3D; point it at the real data location
[ -L vidfm3d/data/CO3D ] || ln -s "$DATA_DIR/CO3D" vidfm3d/data/CO3D

# ---------- 1) official download to /tmp ----------
if [ -f "$DL/.done" ]; then
    echo "== CO3D subset already in $DL"
else
    echo "== free on $TMP_ROOT: $(df -h "$TMP_ROOT" | awk 'NR==2{print $4}') (needs up to ~20 GB)"
    [ -d "$CO3D_REPO" ] || git clone -q --depth 1 https://github.com/facebookresearch/co3d.git "$CO3D_REPO"
    echo "== downloading; follow it from another terminal with: tail -f $TMP_ROOT/co3d_download.log"
    # its progress output is very noisy, so it goes to a log file
    if ! python "$CO3D_REPO/co3d/download_dataset.py" \
            --download_folder "$DL" --single_sequence_subset \
            --n_download_workers 2 --n_extract_workers 2 \
            > "$TMP_ROOT/co3d_download.log" 2>&1; then
        echo "!! download failed, last lines of the log:"
        tail -n 15 "$TMP_ROOT/co3d_download.log"
        exit 1
    fi
    rm -f "$DL"/*.zip                         # everything is unpacked: drop the archives
    touch "$DL/.done"                         # marker: download + unpacking finished completely
fi
du -sh "$DL"

# ---------- 2) copy only images + masks of CATS to /home ----------
for c in "${CATS[@]}"; do
    rsync -am --include='*/' --include='/*/*/images/*' --include='/*/*/masks/*' --exclude='*' \
        "$DL/$c" "$RAW/"
    echo "   $c: $(find "$RAW/$c" -mindepth 1 -maxdepth 1 -type d | wc -l) videos"
done
du -sh "$RAW"

# ---------- 3) model weights ----------
echo "== model weights"
export HF_HOME=$DATA_DIR/hf_cache
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("facebook/VGGT-1B", allow_patterns=["config.json", "model.safetensors"])
snapshot_download("facebook/dinov2-small",
                  allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"])
PY
du -sh "$HF_HOME"
echo "== free on /home: $(df -h ~ | awk 'NR==2{print $4}')"
echo "== depths, point clouds and annotations are still in $DL; free it with: rm -rf $DL $CO3D_REPO"
