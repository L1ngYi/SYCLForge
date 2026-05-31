from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from syclforge.tasks import GemmTask


REPO_ROOT = Path(__file__).resolve().parents[1]
CUDAFORGE_ROOT = REPO_ROOT / "CudaForge" if (REPO_ROOT / "CudaForge" / "prompts").is_dir() else REPO_ROOT
GPU_SPEC_FILE = CUDAFORGE_ROOT / "prompts" / "hardware" / "gpu_specs.py"


def _load_gpu_specs() -> dict[str, Any]:
    if not GPU_SPEC_FILE.is_file():
        return {}
    spec = importlib.util.spec_from_file_location("syclforge_gpu_specs", GPU_SPEC_FILE)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    sys.modules["syclforge_gpu_specs"] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return getattr(module, "GPU_SPEC_INFO", {})


def _gpu_block(gpu_name: str) -> str:
    specs = _load_gpu_specs()
    info = specs.get(gpu_name) or specs.get("A100") or {}
    if not info:
        return f"GPU Name: {gpu_name}\nArchitecture: NVIDIA A100-compatible CUDA backend"
    arch = info.get("GPU Architecture", "Unknown")
    items = "\n".join(f"- {key}: {value}" for key, value in info.items() if key != "GPU Architecture")
    return f"GPU Name: {gpu_name}\nArchitecture: {arch}\nDetails:\n{items}"


def _shape_block(task: GemmTask) -> str:
    return f"Shape: M={task.m}, K={task.k}, N={task.n}\nFLOPs: {int(task.flops)}"


def _result_block(result: dict[str, Any]) -> str:
    compact = {
        "compile_success": result.get("compile_success"),
        "runnable": result.get("runnable"),
        "correctness_pass": result.get("correctness_pass"),
        "time_ms": result.get("time_ms"),
        "gflops": result.get("gflops"),
        "peak_pct": result.get("peak_pct"),
        "error_type": result.get("error_type"),
        "message": str(result.get("message") or "")[-3000:],
    }
    return json.dumps(compact, ensure_ascii=False, indent=2)


def build_repair_prompt(
    *,
    task: GemmTask,
    gpu_name: str,
    current_code: str,
    bench_result: dict[str, Any],
) -> tuple[str, str]:
    system = """You are a senior SYCL compiler engineer repairing one GEMM kernel.
Return only corrected C++ SYCL source in one fenced cpp block. No prose."""
    prompt = f"""# Target
{_gpu_block(gpu_name)}

# GEMM task
{_shape_block(task)}

# Required API
The output must define exactly this callable API:
void gemm(float *A, float *B, float *C, int m, int k, int n, sycl::queue &q)

The function must enqueue one SYCL kernel and leave synchronization to the caller.
Do not include main(), tests, host allocation, oneMKL, CUDA code, or non-SYCL libraries.

# Failure report
{_result_block(bench_result)}

# Current source
```cpp
{current_code}
```

Repair the code while preserving the API and numerical behavior C = A @ B.
"""
    return system, prompt


def build_judge_prompt(
    *,
    task: GemmTask,
    gpu_name: str,
    current_code: str,
    bench_result: dict[str, Any],
    ncu_metrics_block: str,
) -> tuple[str, str]:
    system = """You are a senior NVIDIA A100 performance engineer for SYCL kernels.
Pick exactly one highest-impact optimization target. Return only JSON."""
    prompt = f"""# Target GPU
{_gpu_block(gpu_name)}

# GEMM task
{_shape_block(task)}

# Current benchmark result
{_result_block(bench_result)}

# Nsight Compute metrics
{ncu_metrics_block or "{}"}

# Current SYCL source
```cpp
{current_code}
```

Return exactly this JSON object:
```json
{{
  "bottleneck": "<max 30 words>",
  "optimisation method": "<max 35 words>",
  "modification plan": "<max 45 words>"
}}
```
"""
    return system, prompt


def build_optimization_prompt(
    *,
    task: GemmTask,
    gpu_name: str,
    current_code: str,
    bench_result: dict[str, Any],
    ncu_metrics_block: str,
    strategy: Any,
) -> tuple[str, str]:
    system = """You are a SYCL GEMM optimization specialist targeting NVIDIA A100 through the SYCL CUDA backend.
Return only improved C++ SYCL source in one fenced cpp block. No prose."""
    strategy_text = json.dumps(strategy, ensure_ascii=False, indent=2) if not isinstance(strategy, str) else strategy
    prompt = f"""# Target
{_gpu_block(gpu_name)}

# GEMM task
{_shape_block(task)}

# Required API and constraints
- Define exactly: void gemm(float *A, float *B, float *C, int m, int k, int n, sycl::queue &q)
- Input matrices are row-major: A[M,K], B[K,N], C[M,N].
- Preserve float32 output correctness with rtol/atol around 1e-3.
- Enqueue one kernel from q using nd_range or parallel_for.
- Standard SYCL only: no CUDA syntax, no oneMKL, no main(), no host malloc/free.
- Shape-specialized constants are allowed for this M/K/N case.
- Prefer robust A100 optimizations: 2D tiling, local_accessor caching, coalesced global reads, register micro-tiles, K unrolling, local-memory padding, reqd_sub_group_size(32).

# Current benchmark result
{_result_block(bench_result)}

# Nsight Compute metrics
{ncu_metrics_block or "{}"}

# Strategy to apply
{strategy_text}

# Current SYCL source
```cpp
{current_code}
```

Produce one improved source file now.
"""
    return system, prompt
