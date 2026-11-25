"""Show cached features as colours with PCA, next to the RGB frames.

    python tools/feature_pca.py --target <targets.sft> --out <folder> \
        --feat dino=<feature.sft> --feat wan=<feature_t749_layer20.sft>

Each patch has hundreds of numbers (384 for DINOv2-small, 1536 for WAN). PCA finds the 3 directions
in which the patches of this video differ most, and shows them as red, green and blue.
Patches with similar features get similar colours. --object looks only at the object, not the background.
Needs numpy, Pillow and matplotlib (no torch), so it also runs on the laptop.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sft_to_images import read_safetensors


def pca_colors(feat: np.ndarray, keep: np.ndarray | None) -> np.ndarray:
    """(T,H,W,C) features -> (T,H,W,3) colours. One PCA for the whole video, so frames are comparable."""
    t, h, w, c = feat.shape
    x = feat.reshape(-1, c).astype(np.float32)                 # every patch of every frame is one row
    fit = x if keep is None else x[keep.reshape(-1)]           # patches that define the directions
    mean = fit.mean(0)
    cov = np.cov((fit - mean).T)                               # (C,C): how the numbers vary together
    values, vectors = np.linalg.eigh(cov)                      # sorted from small to large
    proj = (x - mean) @ vectors[:, -3:][:, ::-1]               # keep the 3 largest directions

    share = values[::-1][:3] / values.sum()
    top_channel = cov.diagonal().max() / cov.diagonal().sum()
    print(f"   3 components hold {share.sum():.0%} of the variance ({', '.join(f'{s:.0%}' for s in share)}); "
          f"largest single channel: {top_channel:.0%}")

    ref = proj if keep is None else proj[keep.reshape(-1)]
    lo, hi = np.percentile(ref, [5, 95], axis=0)               # same 5-95% range as the repo's own PCA
    rgb = (np.clip((proj - lo) / (hi - lo + 1e-8), 0, 1) * 255).astype(np.uint8)
    if keep is not None:
        rgb[~keep.reshape(-1)] = 0                             # background black
    return rgb.reshape(t, h, w, 3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--feat", action="append", required=True, metavar="NAME=FILE")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--object", action="store_true", help="PCA on the object only (uses the mask)")
    args = p.parse_args()

    target = read_safetensors(args.target)
    frames = target["images"].transpose(0, 2, 3, 1)            # (81,H,W,3)
    masks = np.asarray(target["masks"][..., 0])                # (81,H,W)

    rows = {"rgb": None}
    for item in args.feat:
        name, path = item.split("=", 1)
        feat = read_safetensors(Path(path))["feat"]
        t, h, w, _ = feat.shape
        # which RGB frame does each feature map belong to?
        if t == len(frames):
            frame_of = np.arange(t)                            # DINO: one map per frame
        elif t == (len(frames) - 1) // 4 + 1:
            frame_of = np.arange(t) * 4                        # WAN: frame 1, then one map per 4 frames (show the last)
        else:
            raise SystemExit(f"{name}: don't know how {t} feature maps fit {len(frames)} frames")
        print(f"{name}: {feat.shape}, finite: {bool(np.isfinite(feat).all())}")
        keep = None
        if args.object:
            small = [np.asarray(Image.fromarray(masks[i]).resize((w, h), Image.NEAREST)) for i in frame_of]
            keep = np.stack(small).astype(bool)
        colors = pca_colors(feat, keep)
        folder = args.out / f"pca_{name}{'_object' if args.object else ''}"
        folder.mkdir(parents=True, exist_ok=True)
        for i, c in zip(frame_of, colors):
            Image.fromarray(c).resize((w * 10, h * 10), Image.NEAREST).save(folder / f"frame_{i + 1:05d}.png")
        rows[name] = (colors, {int(f): k for k, f in enumerate(frame_of)})

    # overview: frames that exist in every model (multiples of 4)
    picks = (np.linspace(0, (len(frames) - 1) // 4, 6).round() * 4).astype(int)
    cw, ch, left, top = 259, 147, 60, 20
    sheet = Image.new("RGB", (left + cw * len(picks), top + ch * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for c, i in enumerate(picks):
        draw.text((left + c * cw + 5, 4), f"frame {i + 1}", fill="black")
    for r, (name, data) in enumerate(rows.items()):
        draw.text((5, top + r * ch + ch // 2), name, fill="black")
        for c, i in enumerate(picks):
            img = np.asarray(frames[i]) if data is None else data[0][data[1][int(i)]]
            sheet.paste(Image.fromarray(img).resize((cw, ch), Image.NEAREST), (left + c * cw, top + r * ch))
    name = f"pca_overview{'_object' if args.object else ''}.png"
    sheet.save(args.out / name)
    print(f"wrote {args.out / name}")


if __name__ == "__main__":
    main()
