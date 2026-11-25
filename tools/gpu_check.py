"""Task 1: sanity-check the GPU and learn how to time GPU code correctly."""
import time

import torch


def timed_matmul(device: str, n: int = 4096, reps: int = 10, dtype=torch.float32) -> float:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    a @ b                                   # warm-up: first call pays for CUDA init / kernel selection
    if device == "cuda":
        torch.cuda.synchronize()            # GPU calls are async: wait until the GPU is really done
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    if device == "cuda":
        torch.cuda.synchronize()            # without this you'd time only the *launch*, not the work
    dt = (time.perf_counter() - t0) / reps
    tflops = 2 * n**3 / dt / 1e12           # an n×n matmul is ~2n³ floating-point ops
    print(f"  {device:4s} {str(dtype):15s} n={n} {dt * 1e3:8.2f} ms/matmul  {tflops:6.1f} TFLOP/s")
    return dt


def main() -> None:
    print(f"torch {torch.__version__} | built with CUDA {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible to PyTorch: check --gpus in the job script.")

    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name} | {props.total_memory / 1e9:.0f} GB | "
          f"compute capability {props.major}.{props.minor} | bf16: {torch.cuda.is_bf16_supported()}")

    print("n×n matmul (compare TFLOP/s: the CPU uses a smaller n):")
    timed_matmul("cpu", n=2048, reps=2)     # smaller on CPU, it's slow
    timed_matmul("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True   # A100 tensor cores for fp32 matmuls (slightly less precise)
    timed_matmul("cuda")
    timed_matmul("cuda", dtype=torch.bfloat16)     # what `precision: bf16-mixed` uses in training


if __name__ == "__main__":
    main()
