"""Task 5: check the training data before trusting any result.

    python tools/verify_data.py              # all videos (use jobs/11_verify_data.sh)
    python tools/verify_data.py --limit 1    # quick look at one video

Run from the repo root. CPU only.
"""
import argparse
import json
import random
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path("vidfm3d/data/CO3D")
TARGETS, FEATS = ROOT / "CO3D-processed", ROOT / "FEAT"
H, W, S = 294, 518, 81
WAN_FILE = "feature_t749_layer20.sft"
results: list[tuple[str, bool]] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok)))
    print(f"   {'PASS' if ok else 'FAIL'}  {name:46s} {detail}")


# ----------------------------------------------------------------------------- one target file
def check_target(path: Path) -> dict[str, float]:
    d = load_file(path)
    depth, pts, conf = d["depthmaps"][..., 0], d["pointmaps"], d["confmaps"][..., 0]   # (S,H,W), (S,H,W,3), (S,H,W)
    K, E = d["intrinsic"], d["extrinsic"]                                              # (S,3,3), (S,3,4)
    R, t = E[:, :, :3], E[:, :, 3]
    eye = torch.eye(3)
    out = {}

    out["shapes_ok"] = float(d["images"].shape == (S, 3, H, W) and depth.shape == (S, H, W) and pts.shape == (S, H, W, 3)
                             and conf.shape == (S, H, W) and K.shape == (S, 3, 3) and E.shape == (S, 3, 4))
    out["finite"] = float(all(torch.isfinite(x).all() for x in (depth, pts, conf, K, E)))
    out["depth_min"] = depth.min().item()

    # cameras: R must be a pure rotation: R R^T = I and det(R) = +1
    out["rot_err"] = (R @ R.transpose(1, 2) - eye).abs().max().item()     # @ works on the whole batch at once
    out["det_err"] = (torch.linalg.det(R) - 1).abs().max().item()
    out["cam0_err"] = (E[0] - torch.cat([eye, torch.zeros(3, 1)], 1)).abs().max().item()   # world = first camera?
    out["cx_err"] = max((K[:, 0, 2] - W / 2).abs().max().item(), (K[:, 1, 2] - H / 2).abs().max().item())
    out["fx_fy"] = (K[:, 0, 0] / K[:, 1, 1]).median().item()

    # how far the camera travels around the object: angle between first and last viewing direction
    cos = ((torch.trace(R[0] @ R[-1].T) - 1) / 2).clamp(-1, 1)
    out["turn_deg"] = torch.rad2deg(torch.acos(cos)).item()

    # 1) depth + cameras must give back the stored 3D points (pixel -> camera -> world)
    v, u = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    f, c = K[:, [0, 1], [0, 1]], K[:, [0, 1], [2, 2]]                     # (S,2) focal lengths / principal points
    x = (u - c[:, 0, None, None]) * depth / f[:, 0, None, None]           # broadcasting: (H,W) with (S,1,1)
    y = (v - c[:, 1, None, None]) * depth / f[:, 1, None, None]
    cam = torch.stack([x, y, depth], -1)                                  # (S,H,W,3) in each camera's own frame
    # extrinsic maps world -> camera: X_cam = R X + t, so X = R^T (X_cam - t)
    world = torch.einsum("sji,shwj->shwi", R, cam - t[:, None, None, :])
    out["unproject_err"] = ((world - pts).norm(dim=-1) / depth).max().item()

    # 2) views must agree: take frame j's 3D points, look at them from camera i, compare with frame i's depth
    i = torch.arange(0, S - 8, 8)
    j = i + 8                                                             # pairs of frames 8 apart
    p = pts[j][:, ::4, ::4].reshape(len(j), -1, 3)                        # (P,N,3) every 4th pixel is enough
    sure = conf[j][:, ::4, ::4].reshape(len(j), -1)
    sure = sure > sure.median(dim=1, keepdim=True).values                 # only VGGT's confident half
    seen = torch.einsum("pab,pnb->pna", R[i], p) + t[i][:, None]          # world -> camera i
    z = seen[..., 2]
    uv = torch.einsum("pab,pnb->pna", K[i], seen)[..., :2] / z[..., None].clamp(min=1e-6)
    ui, vi = uv[..., 0].round().long(), uv[..., 1].round().long()
    inside = (z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & sure
    idx = vi.clamp(0, H - 1) * W + ui.clamp(0, W - 1)
    d_i = depth[i].reshape(len(i), -1).gather(1, idx)                     # frame i's own depth at that pixel
    out["in_front"] = (z[sure] > 0).float().mean().item()
    out["views_err"] = ((z - d_i).abs() / d_i)[inside].median().item()    # median: hidden surfaces are outliers
    return out


def check_targets(files: list[Path]) -> None:
    print(f"\n== VGGT targets ({len(files)} videos)")
    rows = {}
    for f in files:
        rows[f] = r = check_target(f)
        print(f"   {f.parent.name + '/' + f.stem:36s} views disagree {r['views_err']:6.2%}  unproject {r['unproject_err']:.1e}  "
              f"cam0 {r['cam0_err']:.3f}  fx/fy {r['fx_fy']:.3f}  camera turn {r['turn_deg']:5.0f} deg", flush=True)

    def worst(key: str) -> float:
        return max(r[key] for r in rows.values())

    def least(key: str) -> float:
        return min(r[key] for r in rows.values())

    report("shapes and types as expected", least("shapes_ok") == 1)
    report("no NaN or infinity", least("finite") == 1)
    report("all depths > 0", least("depth_min") > 0, f"smallest {least('depth_min'):.3f}")
    report("R is a rotation (R R^T = I, det = 1)", max(worst("rot_err"), worst("det_err")) < 1e-3, f"worst {max(worst('rot_err'), worst('det_err')):.1e}")
    report("first camera = world origin (roughly)", worst("cam0_err") < 0.05, f"worst {worst('cam0_err'):.3f}")
    report("principal point = image centre", worst("cx_err") < 0.5, f"worst {worst('cx_err'):.2f} px")
    report("fx and fy similar", 0.9 < least("fx_fy") and worst("fx_fy") < 1.1, f"{least('fx_fy'):.3f} to {worst('fx_fy'):.3f}")
    report("depth + cameras reproduce the 3D points", worst("unproject_err") < 1e-3, f"worst {worst('unproject_err'):.1e}")
    report("points lie in front of other cameras", least("in_front") > 0.99, f"least {least('in_front'):.2%}")
    report("views agree (median depth error < 5%)", worst("views_err") < 0.05, f"worst {worst('views_err'):.2%}")


# ----------------------------------------------------------------------------- features
def check_features(files: list[Path]) -> None:
    print("\n== cached features")
    for name, fname, shape in (("dino", "feature.sft", (81, 30, 52, 384)), ("wan", WAN_FILE, (21, 30, 52, 1536))):
        missing, bad = [], []
        for f in files:
            p = FEATS / name / f.parent.name / f.stem / fname
            if not p.exists():
                missing.append(p)
                continue
            with safe_open(p, framework="pt") as s:
                x = s.get_tensor("feat")
            if tuple(x.shape) != shape or not torch.isfinite(x).all():
                bad.append(p)
        report(f"{name}: every video, right shape, finite", not missing and not bad, f"{len(files) - len(missing) - len(bad)}/{len(files)}")


# ----------------------------------------------------------------------------- splits
def check_splits() -> None:
    print("\n== train/test split")
    from vidfm3d.data.processing.co3d.create_split import collect_scenes

    train, val = (json.load(open(TARGETS / f"{s}.json")) for s in ("train", "val"))
    tr, va = set(map(tuple, train)), set(map(tuple, val))
    on_disk = collect_scenes(TARGETS, "all")
    report("no video in both", not tr & va, f"{len(tr)} train / {len(va)} test")
    report("every video is used once", tr | va == set(on_disk) and len(train) + len(val) == len(on_disk))
    # repeat what create_split.py does with its defaults
    again = list(on_disk)
    random.seed(0)
    random.shuffle(again)
    n = max(1, int(len(again) * 0.1))
    report("re-creating it gives the same split", [tuple(x) for x in val] == again[:n] and [tuple(x) for x in train] == again[n:])
    print(f"   note: the code's 'test' split is val.json: {['/'.join(x) for x in val]}")


# ----------------------------------------------------------------------------- what the training code really gets
def check_dataset() -> None:
    print("\n== training samples (VideoProbeDataset)")
    from vidfm3d.data.components.video_probe_dataset import VideoProbeDataset

    for name, post, extra in (("dino", "", {"fold_stride": 1}), ("wan", "_t749_layer20", {})):
        def make(seed):
            return VideoProbeDataset(root=str(TARGETS), root_vfm=str(FEATS), subset="all", split="val", vfm_name=name,
                                     feat_postfix=post, feat_pixalign=True, num_views=4, min_view_interval=5,
                                     context_len=76, query_idx_divisor=4, seed=seed, **extra)
        ds = make(1)
        s = ds[0]
        path = Path(s["scene_path"])
        # which frames were picked? find them by comparing the images
        imgs = load_file(path)["images"]
        small = imgs[:, :, ::16, ::16].float()
        got = (s["image"][:, :, ::16, ::16] * 255).round()
        sel = [(small - g).abs().mean(dim=(1, 2, 3)).argmin().item() for g in got]

        eye = torch.cat([torch.eye(3), torch.zeros(3, 1)], 1)
        report(f"{name}: first view is the reference (identity)", (s["extrinsics"][0] - eye).abs().max() < 1e-4)
        dist = s["pmaps"][0].norm(dim=0).mean().item()
        report(f"{name}: mean distance in first view = 1", abs(dist - 1) < 1e-3, f"{dist:.4f}")
        zerr = ((s["pmaps"][0, 2] - s["dmaps"][0, 0]).abs() / s["dmaps"][0, 0]).max().item()
        report(f"{name}: depth = z of the points (first view)", zerr < 1e-3, f"worst {zerr:.1e}")

        # each picked frame must come with ITS OWN features
        feat = load_file(FEATS / name / path.parent.name / path.stem / f"feature{post}.sft")["feat"]
        want = sel if name == "dino" else [0 if f == 0 else (f - 1) // 4 + 1 for f in sel]
        report(f"{name}: features belong to the picked frames", torch.equal(s["vfm_feat"], feat[want]), f"frames {sel} -> maps {want}")

        same = torch.equal(ds[0]["image"], ds[0]["image"])
        zero = make(0)
        picks = [zero[0]["image"] for _ in range(4)]
        fixed0 = all(torch.equal(picks[0], x) for x in picks[1:])
        report(f"{name}: seed=1 fixes the test frames", same)
        print(f"   note  {name}: with the config's seed=0 the test frames are {'fixed' if fixed0 else 'NOT fixed (they change on every read)'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="only check the first N videos")
    args = p.parse_args()
    files = sorted(TARGETS.glob("*/*.sft"))[: args.limit]

    check_targets(files)
    check_features(files)
    check_splits()
    check_dataset()

    failed = [n for n, ok in results if not ok]
    print(f"\n== {len(results) - len(failed)} of {len(results)} checks passed" + (f", FAILED: {failed}" if failed else ""))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
