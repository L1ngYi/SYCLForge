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


def _tensor_core_block(
    enabled: bool,
    report: dict[str, Any] | None = None,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> str:
    if not enabled:
        return """Tensor-core mode: disabled.
Use portable SIMT SYCL optimizations only. Do not use joint_matrix or other experimental matrix extensions."""

    report = report or {}
    flavor = report.get("flavor") or "tf32_joint_matrix"
    return f"""Tensor-core mode: enabled.
SYCLForge compiled and ran a {flavor} canary using <sycl/ext/oneapi/matrix/matrix.hpp>.
You may use sycl::ext::oneapi::experimental::matrix joint_matrix APIs:
- joint_matrix, joint_matrix_load, joint_matrix_store, joint_matrix_fill, joint_matrix_mad
- Prefer the modern void API form: mx::joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
- sycl::sub_group with [[sycl::reqd_sub_group_size(32)]]
- TF32 input fragments via sycl::ext::oneapi::experimental::matrix::precision::tf32 with float accumulators

For the NVIDIA CUDA backend on A100, follow these hard rules:
- Use namespace alias `namespace mx = sycl::ext::oneapi::experimental::matrix;`.
- A and B fragments must use `mx::precision::tf32`, not `float`.
- A and B fragments must specify a concrete layout, usually `mx::layout::row_major`; do not leave them as layout::dynamic.
- Accumulator fragments use `float` and `mx::use::accumulator`.
- Prefer supported TF32 shape `M=16, N=16, K=8` for one subgroup tile.
- Launch one subgroup per 16x16 C tile. Use a work-group containing exactly one 32-lane subgroup, for example `sycl::nd_range<2>(sycl::range<2>(num_tiles, 32), sycl::range<2>(1, 32))`; do not use SIMT-style local ranges such as `(16,16)` for a joint_matrix kernel.
- joint_matrix_load/store require a SYCL multi_ptr or accessor pointer, not a raw `float*`.
- For USM pointers inside the kernel, first create:
  `auto pA = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(A);`
  and similarly for B and C, then pass `pA + offset`, `pB + offset`, `pC + offset` to joint_matrix_load/store.
- Never write `joint_matrix<sub_group, float, mx::use::a, ...>` or `joint_matrix<sub_group, float, mx::use::b, ...>` on CUDA; that instantiates an unsupported `joint_matrix_cuda<float,...>`.

Minimal legal TF32 pattern:
```cpp
namespace mx = sycl::ext::oneapi::experimental::matrix;
using tf32 = mx::precision::tf32;
auto pA = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(A);
auto pB = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(B);
auto pC = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(C);
mx::joint_matrix<sycl::sub_group, tf32, mx::use::a, 16, 8, mx::layout::row_major> sub_a;
mx::joint_matrix<sycl::sub_group, tf32, mx::use::b, 8, 16, mx::layout::row_major> sub_b;
mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 16, 16> sub_c;
mx::joint_matrix_fill(sg, sub_c, 0.0f);
mx::joint_matrix_load(sg, sub_a, pA + a_offset, k);
mx::joint_matrix_load(sg, sub_b, pB + b_offset, n);
mx::joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
mx::joint_matrix_store(sg, sub_c, pC + c_offset, n, mx::layout::row_major);
```

Keep the public API as float* A, float* B, float* C. Do not use CUDA syntax or oneMKL.
For A100 FP32 GEMM, prefer a TF32 Tensor Core plan when correctness can still pass rtol={rtol:g}, atol={atol:g}.
Use Nsight metric sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active as the main signal that Tensor Cores are actually used."""


def build_repair_prompt(
    *,
    task: GemmTask,
    gpu_name: str,
    current_code: str,
    bench_result: dict[str, Any],
    tensor_core_enabled: bool = False,
    tensor_core_report: dict[str, Any] | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-3,
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

# Tensor Core policy
{_tensor_core_block(tensor_core_enabled, tensor_core_report, rtol=rtol, atol=atol)}

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
    tensor_core_enabled: bool = False,
    tensor_core_report: dict[str, Any] | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> tuple[str, str]:
    system = """You are a senior NVIDIA A100 performance engineer for SYCL kernels.
Pick exactly one highest-impact optimization target. Return only JSON."""
    prompt = f"""# Target GPU
{_gpu_block(gpu_name)}

# GEMM task
{_shape_block(task)}

# Tensor Core policy
{_tensor_core_block(tensor_core_enabled, tensor_core_report, rtol=rtol, atol=atol)}

# Current benchmark result
{_result_block(bench_result)}

# Nsight Compute metrics
{ncu_metrics_block or "{}"}

# Current SYCL source
```cpp
{current_code}
```

When tensor-core mode is enabled and tensor pipe utilization is near zero, strongly prefer a joint_matrix/TF32 Tensor Core optimization target unless the shape or correctness constraints make it unreasonable.

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
    tensor_core_enabled: bool = False,
    tensor_core_report: dict[str, Any] | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-3,
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
- Preserve float32 output correctness with rtol={rtol:g}, atol={atol:g}.
- Enqueue one kernel from q using nd_range or parallel_for.
- No CUDA syntax, no oneMKL, no main(), no host malloc/free.
- Shape-specialized constants are allowed for this M/K/N case.
- Prefer robust A100 optimizations: 2D tiling, local_accessor caching, coalesced global reads, register micro-tiles, K unrolling, local-memory padding, reqd_sub_group_size(32).

# Tensor Core policy
{_tensor_core_block(tensor_core_enabled, tensor_core_report, rtol=rtol, atol=atol)}

If tensor-core mode is enabled, try a complete joint_matrix TF32 Tensor Core implementation before making another small SIMT-only tweak. The generated source should include any required SYCL matrix-extension header itself.

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
