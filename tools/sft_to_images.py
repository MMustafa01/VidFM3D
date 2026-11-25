"""Turn a VGGT target file (.sft) into images you can look at.

    python tools/sft_to_images.py <file.sft> <output_folder>

Writes one PNG per frame into rgb/, mask/, depth/, confidence/, points/, plus overview.png.
Needs only numpy, Pillow and matplotlib (no torch), so it also runs on the laptop.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

DTYPES = {"F32": np.float32, "F16": np.float16, "U8": np.uint8, "BOOL": np.bool_, "I32": np.int32}


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    """File layout: 8 bytes = header length, then a JSON header, then the raw numbers."""
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    # memmap = the numbers are read from disk only when we use them
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + header_len)
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        tensors[name] = raw[start:end].view(DTYPES[info["dtype"]]).reshape(info["shape"])
    return tensors


def to_color(x: np.ndarray, lo: float, hi: float, cmap: str = "viridis") -> np.ndarray:
    """Map numbers in [lo, hi] to colours. One range for the whole video keeps frames comparable."""
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    return (matplotlib.colormaps[cmap](x)[..., :3] * 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sft", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()

    t = read_safetensors(args.sft)
    for name, x in t.items():
        print(f"{name:10s} {str(x.shape):22s} {str(x.dtype):8s} min {x.min():9.3f}  max {x.max():9.3f}")

    # every array becomes (frames, H, W, 3) uint8
    depth, conf, pts = t["depthmaps"][..., 0], t["confmaps"][..., 0], t["pointmaps"]
    # percentiles instead of min/max: a few extreme pixels would otherwise flatten all the contrast
    p_lo, p_hi = np.percentile(pts, [2, 98], axis=(0, 1, 2))
    layers = {
        "rgb": t["images"].transpose(0, 2, 3, 1),                                     # (S,3,H,W) -> (S,H,W,3)
        "mask": np.repeat(t["masks"], 3, axis=-1).astype(np.uint8) * 255,
        "depth": to_color(depth, *np.percentile(depth, [2, 98])),                     # dark = near, yellow = far
        "confidence": to_color(conf, *np.percentile(conf, [2, 98])),                  # dark = unsure, yellow = sure
        "points": (np.clip((pts - p_lo) / (p_hi - p_lo), 0, 1) * 255).astype(np.uint8),  # x,y,z -> R,G,B
    }

    for name, frames in layers.items():
        folder = args.out / name
        folder.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames, 1):
            Image.fromarray(np.asarray(frame)).save(folder / f"frame_{i:05d}.png")

    # overview: one row per layer, 6 frames spread over the video
    picks = np.linspace(0, len(depth) - 1, 6).round().astype(int)
    w, h, left, top = 259, 147, 90, 20
    sheet = Image.new("RGB", (left + w * len(picks), top + h * len(layers)), "white")
    draw = ImageDraw.Draw(sheet)
    for c, i in enumerate(picks):
        draw.text((left + c * w + 5, 4), f"frame {i + 1}", fill="black")
    for r, (name, frames) in enumerate(layers.items()):
        draw.text((5, top + r * h + h // 2), name, fill="black")
        for c, i in enumerate(picks):
            sheet.paste(Image.fromarray(np.asarray(frames[i])).resize((w, h)), (left + c * w, top + r * h))
    sheet.save(args.out / "overview.png")
    print(f"wrote {sum(len(f) for f in layers.values())} images + overview.png to {args.out}")


if __name__ == "__main__":
    main()
