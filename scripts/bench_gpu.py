# scripts/bench_gpu.py —— 8GB 显存预算与精度吞吐基准
import argparse
import json
import time
import pathlib
import torch


def bench(dtype: torch.dtype, n: int, iters: int, device: str) -> dict:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    return {
        "dtype": str(dtype),
        "n": n,
        "ms": round(dt * 1000, 2),
        "tflops": round(2 * n**3 / dt / 1e12, 2),
    }


def vram_budget() -> dict:
    free, total = torch.cuda.mem_get_info()
    return {
        "total_gb": round(total / 1024**3, 2),
        "free_gb": round(free / 1024**3, 2),
        "used_by_screen_and_others_gb": round((total - free) / 1024**3, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/w01_bench.json")
    args = ap.parse_args()
    assert torch.cuda.is_available() or args.device == "cpu", "CUDA 不可用，先查驱动"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    results = [
        bench(dt, 4096, 20, args.device)
        for dt in (torch.float32, torch.float16, torch.bfloat16)
    ]
    payload = {
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "budget": vram_budget(),
        "results": results,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
