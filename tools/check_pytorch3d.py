"""Does our pytorch3d build give correct answers on CPU and GPU, and does the training code import?"""
import argparse
import math

import torch
from pytorch3d.ops.knn import knn_gather, knn_points          # compiled C++/CUDA code
from pytorch3d.structures.pointclouds import Pointclouds
from pytorch3d.transforms import so3_relative_angle


def check(device: str) -> None:
    # nearest neighbours with a known answer: shift a point set a little and shuffle it
    torch.manual_seed(0)
    a = torch.rand(1, 500, 3, device=device)
    order = torch.randperm(500, device=device)
    b = a[:, order] + 1e-5                                    # b[j] is (almost) a[order[j]]
    nn = knn_points(b, a, K=1)
    assert torch.equal(nn.idx[0, :, 0], order), "wrong neighbours"
    assert knn_gather(a, nn.idx).shape == (1, 500, 1, 3)
    assert Pointclouds(points=list(a)).num_points_per_cloud().item() == 500

    # angle between "no rotation" and a 90 degree turn around z
    r90 = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]], device=device)[None]
    angle = so3_relative_angle(torch.eye(3, device=device)[None], r90).item()
    assert math.isclose(angle, math.pi / 2, abs_tol=1e-3), f"angle {angle}"
    print(f"   OK  {device}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--require-gpu", action="store_true")
    args = p.parse_args()

    print(f"torch {torch.__version__} | GPU visible: {torch.cuda.is_available()}")
    check("cpu")
    if torch.cuda.is_available():
        check("cuda")
    elif args.require_gpu:
        raise SystemExit("no GPU visible")

    import vidfm3d.models.video_probe_module                  # noqa: F401  everything training needs

    print("   OK  training code imports")
    print("== all checks passed")


if __name__ == "__main__":
    main()
