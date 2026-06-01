from __future__ import annotations

import csv
import ctypes
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from syclforge.tasks import GemmTask


FAILURE_MS = 1_000_000.0

NCU_METRICS = [
    "sm__cycles_active.avg",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__registers_per_thread",
    "sm__inst_executed.sum",
    "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum.per_second",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lts__t_sector_hit_rate.pct",
    "lts__throughput.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
]


@dataclass
class CompileResult:
    success: bool
    output: str = ""
    command: list[str] = field(default_factory=list)


@dataclass
class BenchResult:
    compile_success: bool = False
    runnable: bool = False
    correctness_pass: bool = False
    time_ms: float | None = None
    gflops: float | None = None
    peak_pct: float | None = None
    score: float = float("-inf")
    time_samples_ms: list[float] = field(default_factory=list)
    compile_output: str = ""
    runtime_output: str = ""
    error_type: str = ""
    message: str = ""
    harness_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("harness_source", None)
        for key, value in list(data.items()):
            if isinstance(value, float) and not math.isfinite(value):
                data[key] = None
        return data


@dataclass
class ProfileResult:
    success: bool
    metrics_block: str = "{}"
    rows: list[dict[str, Any]] = field(default_factory=list)
    csv_path: str = ""
    command: list[str] = field(default_factory=list)
    output: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TensorCoreProbeResult:
    requested: bool
    enabled: bool
    flavor: str = ""
    source_path: str = ""
    binary_path: str = ""
    compile_success: bool = False
    compile_output: str = ""
    command: list[str] = field(default_factory=list)
    run_success: bool = False
    run_output: str = ""
    run_returncode: int | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_param_decl(param_decl: str) -> dict[str, Any]:
    normalized = param_decl.replace("*", " * ").replace("&", " & ").strip()
    tokens = normalized.split()
    if not tokens:
        raise ValueError("Encountered an empty SYCL parameter declaration.")

    name = tokens[-1]
    type_tokens = tokens[:-1]
    is_pointer = "*" in type_tokens
    dtype = " ".join(token for token in type_tokens if token not in {"*", "&"}).strip()
    storage_dtype = re.sub(r"\bconst\b", "", dtype)
    storage_dtype = re.sub(r"\s+", " ", storage_dtype).strip()
    return {
        "name": name,
        "dtype": re.sub(r"\s+", " ", dtype).strip(),
        "storage_dtype": storage_dtype,
        "full": param_decl.strip(),
        "is_pointer": is_pointer,
    }


def parse_sycl_function_metadata(source: str) -> dict[str, Any]:
    match = re.search(r"\bvoid\s+(\w+)\s*\(([^)]*)\)", source, re.S)
    if match is None:
        raise ValueError("Could not find `void <name>(...)` SYCL function signature.")

    raw_params = [param.strip() for param in match.group(2).split(",") if param.strip()]
    parsed = [_parse_param_decl(param) for param in raw_params]
    data_params = [param for param in parsed if "queue" not in param["dtype"]]
    pointer_params = [param for param in data_params if param["is_pointer"]]
    scalar_params = [param for param in data_params if not param["is_pointer"]]
    return {
        "kernel_name": match.group(1),
        "raw_params": parsed,
        "data_params": data_params,
        "pointer_params": pointer_params,
        "scalar_params": scalar_params,
    }


def get_matmul_size_exprs(metadata: dict[str, Any], task: GemmTask) -> list[str]:
    pointer_params = metadata["pointer_params"]
    scalar_params = metadata["scalar_params"]
    if len(pointer_params) != 3:
        raise NotImplementedError("SYCL GEMM harness expects exactly 3 pointer parameters.")

    if len(scalar_params) == 3:
        m_name, k_name, n_name = [param["name"] for param in scalar_params]
        return [f"{m_name} * {k_name}", f"{k_name} * {n_name}", f"{m_name} * {n_name}"]
    if scalar_params:
        raise NotImplementedError("SYCL GEMM harness supports either 0 or exactly 3 scalar dims.")
    return [str(task.m * task.k), str(task.k * task.n), str(task.m * task.n)]


def get_invocation_args(
    metadata: dict[str, Any],
    *,
    pointer_name_map: dict[str, str] | None = None,
    queue_name: str = "q",
) -> list[str]:
    pointer_name_map = pointer_name_map or {}
    call_args: list[str] = []
    for param in metadata["raw_params"]:
        if "queue" in param["dtype"]:
            call_args.append(queue_name)
        elif param["is_pointer"]:
            call_args.append(pointer_name_map.get(param["name"], param["name"]))
        else:
            call_args.append(param["name"])
    return call_args


def add_explicit_kernel_names(source: str, base_name: str) -> str:
    counter = 0

    def replace_parallel_for(match: re.Match[str]) -> str:
        nonlocal counter
        replacement = f"parallel_for<class __{base_name}_sycl_kernel_{counter}>("
        counter += 1
        return replacement

    def replace_single_task(match: re.Match[str]) -> str:
        nonlocal counter
        replacement = f"single_task<class __{base_name}_sycl_kernel_{counter}>("
        counter += 1
        return replacement

    source = re.sub(r"\bparallel_for\s*(?!<)\(", replace_parallel_for, source)
    source = re.sub(r"\bsingle_task\s*(?!<)\(", replace_single_task, source)
    return source


def rewrite_kernel_to_event(source: str, *, kernel_tag: str) -> tuple[str, dict[str, Any]]:
    metadata = parse_sycl_function_metadata(source)
    kernel_name = metadata["kernel_name"]
    rewritten = add_explicit_kernel_names(source, kernel_tag)

    rewritten, sig_subs = re.subn(
        rf"\bvoid\s+{re.escape(kernel_name)}\s*\(",
        f"sycl::event {kernel_name}(",
        rewritten,
        count=1,
    )
    if sig_subs != 1:
        raise ValueError("Could not rewrite SYCL function signature to return sycl::event.")

    queue_params = [param for param in metadata["raw_params"] if "queue" in param["dtype"]]
    if len(queue_params) != 1:
        raise ValueError("Expected exactly one sycl::queue parameter.")
    queue_name = queue_params[0]["name"]

    replacements = (
        (rf"\b{re.escape(queue_name)}\s*\.\s*submit\s*\(", f"return {queue_name}.submit("),
        (rf"\b{re.escape(queue_name)}\s*\.\s*parallel_for\s*\(", f"return {queue_name}.parallel_for("),
        (rf"\b{re.escape(queue_name)}\s*\.\s*single_task\s*\(", f"return {queue_name}.single_task("),
    )
    for pattern, replacement in replacements:
        rewritten, count = re.subn(pattern, replacement, rewritten, count=1)
        if count == 1:
            return rewritten, metadata

    raise ValueError("Could not find a SYCL queue submission in the candidate source.")


def _resolve_sycl_env_script() -> str | None:
    explicit = os.environ.get("SYCL_ENV_SCRIPT")
    candidates = []
    if explicit:
        candidates.append(explicit)
    repo_root = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            Path.cwd() / "env_sycl.sh",
            repo_root / "env_sycl.sh",
            Path.home() / "env_sycl.sh",
        ]
    )

    for base in (Path.cwd(), repo_root):
        candidates.extend(parent / "env_sycl.sh" for parent in base.resolve().parents)

    for candidate in candidates:
        path = Path(candidate).expanduser().resolve()
        if path.is_file():
            return str(path)
    return None


@lru_cache(maxsize=None)
def _load_env_from_script(script_path: str) -> dict[str, str]:
    command = f"source {shlex.quote(script_path)} >/dev/null 2>&1 && env -0"
    proc = subprocess.run(
        ["bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    loaded: dict[str, str] = {}
    for item in proc.stdout.decode("utf-8", errors="ignore").split("\0"):
        if "=" in item:
            key, value = item.split("=", 1)
            loaded[key] = value
    return loaded


@lru_cache(maxsize=1)
def _detect_cuda_arch_from_nvidia_smi() -> str | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        proc = subprocess.run(
            [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            check=True,
            timeout=5,
        )
    except Exception:
        return None
    first_line = proc.stdout.strip().splitlines()[0].strip()
    normalized = first_line.replace(".", "")
    return f"sm_{normalized}" if normalized.isdigit() else None


def get_sycl_environment(*, backend: str = "nvidia", cuda_arch: str = "sm_80") -> dict[str, str]:
    env = os.environ.copy()
    script_path = _resolve_sycl_env_script()
    if script_path:
        env.update(_load_env_from_script(script_path))

    if env.get("SYCL_DEVICE_SELECTOR"):
        env["ONEAPI_DEVICE_SELECTOR"] = env["SYCL_DEVICE_SELECTOR"]

    if backend == "nvidia":
        env.setdefault("ONEAPI_DEVICE_SELECTOR", "cuda:gpu")
        env.setdefault("SYCL_TARGETS", "nvptx64-nvidia-cuda")
        env.setdefault("SYCL_CUDA_ARCH", cuda_arch)
    return env


def _get_sycl_compile_flags(env: dict[str, str], *, backend: str) -> list[str]:
    flags = shlex.split(env.get("SYCL_EXTRA_FLAGS", ""))
    if "-fsycl-unnamed-lambda" not in flags:
        flags.append("-fsycl-unnamed-lambda")

    requested_targets = env.get("SYCL_TARGETS", "").strip()
    has_targets = any(flag.startswith("-fsycl-targets=") for flag in flags)
    if requested_targets and not has_targets:
        flags.append(f"-fsycl-targets={requested_targets}")
        has_targets = True

    using_cuda = backend == "nvidia" or requested_targets == "nvptx64-nvidia-cuda"
    if using_cuda and not has_targets:
        flags.append("-fsycl-targets=nvptx64-nvidia-cuda")

    has_arch = any("cuda-gpu-arch" in flag or "offload-arch" in flag for flag in flags)
    if using_cuda and not has_arch:
        cuda_arch = env.get("SYCL_CUDA_ARCH") or env.get("CUDAARCHS") or _detect_cuda_arch_from_nvidia_smi() or "sm_80"
        flags.extend(["-Xsycl-target-backend=nvptx64-nvidia-cuda", f"--cuda-gpu-arch={cuda_arch}"])
    return flags


def _sycl_compiler_command(env: dict[str, str]) -> list[str]:
    configured = shlex.split(env.get("SYCL_COMPILER", ""))
    if configured:
        return configured
    for name in ("icpx", "clang++"):
        detected = shutil.which(name, path=env.get("PATH"))
        if detected:
            return [detected]
    raise FileNotFoundError("No SYCL compiler found. Set SYCL_COMPILER or load env_sycl.sh.")


def compile_sycl_source(
    source_path: Path,
    output_path: Path,
    *,
    shared: bool,
    backend: str = "nvidia",
    cuda_arch: str = "sm_80",
    timeout: int = 180,
) -> CompileResult:
    try:
        env = get_sycl_environment(backend=backend, cuda_arch=cuda_arch)
        cmd = _sycl_compiler_command(env) + ["-fsycl", "-O3", "-std=c++17"]
        if shared:
            cmd.extend(["-fPIC", "-shared"])
        cmd.extend(_get_sycl_compile_flags(env, backend=backend))
        cmd.extend([str(source_path), "-o", str(output_path)])
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            text=True,
            check=True,
            timeout=timeout,
            env=env,
        )
        return CompileResult(success=True, output=proc.stdout, command=cmd)
    except subprocess.CalledProcessError as exc:
        return CompileResult(success=False, output=exc.output or str(exc), command=cmd if "cmd" in locals() else [])
    except subprocess.TimeoutExpired:
        return CompileResult(success=False, output="Compilation timed out", command=cmd if "cmd" in locals() else [])
    except Exception as exc:
        return CompileResult(success=False, output=str(exc), command=[])


TENSOR_CORE_TF32_CANARY_SOURCE = r"""
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

int main() {
  namespace mx = sycl::ext::oneapi::experimental::matrix;
  using tf32 = mx::precision::tf32;
  constexpr int TM = 16;
  constexpr int TN = 16;
  constexpr int TK = 8;
  constexpr int SG = 32;

  float A[TM * TK] = {};
  float B[TK * TN] = {};
  float C[TM * TN] = {};
  sycl::buffer<float, 2> bufA(A, sycl::range<2>(TM, TK));
  sycl::buffer<float, 2> bufB(B, sycl::range<2>(TK, TN));
  sycl::buffer<float, 2> bufC(C, sycl::range<2>(TM, TN));
  sycl::queue q{sycl::gpu_selector_v};

  q.submit([&](sycl::handler &h) {
    sycl::accessor accA(bufA, h, sycl::read_only);
    sycl::accessor accB(bufB, h, sycl::read_only);
    sycl::accessor accC(bufC, h, sycl::write_only, sycl::no_init);
    h.parallel_for(
        sycl::nd_range<2>(sycl::range<2>(1, SG), sycl::range<2>(1, SG)),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG)]] {
          sycl::sub_group sg = item.get_sub_group();
          mx::joint_matrix<sycl::sub_group, tf32, mx::use::a, TM, TK, mx::layout::row_major> sub_a;
          mx::joint_matrix<sycl::sub_group, tf32, mx::use::b, TK, TN, mx::layout::row_major> sub_b;
          mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> sub_c;
          mx::joint_matrix_fill(sg, sub_c, 0.0f);
          mx::joint_matrix_load(sg, sub_a, accA.get_pointer(), TK);
          mx::joint_matrix_load(sg, sub_b, accB.get_pointer(), TN);
          mx::joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
          mx::joint_matrix_store(sg, sub_c, accC.get_pointer(), TN, mx::layout::row_major);
        });
  });
  q.wait();
  return 0;
}
"""


TENSOR_CORE_TF32_RETURN_CANARY_SOURCE = TENSOR_CORE_TF32_CANARY_SOURCE.replace(
    "mx::joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);",
    "sub_c = mx::joint_matrix_mad(sg, sub_a, sub_b, sub_c);",
)


TENSOR_CORE_CANARIES = [
    ("tf32_joint_matrix", TENSOR_CORE_TF32_CANARY_SOURCE),
    ("tf32_joint_matrix_return_api", TENSOR_CORE_TF32_RETURN_CANARY_SOURCE),
]


def probe_tensor_core_support(
    work_dir: Path,
    *,
    backend: str = "nvidia",
    cuda_arch: str = "sm_80",
) -> TensorCoreProbeResult:
    """Compile small TF32 joint_matrix kernels before enabling tensor-core prompts."""
    probe_dir = work_dir / "tensor_core_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []

    for flavor, source in TENSOR_CORE_CANARIES:
        source_path = probe_dir / f"{flavor}_canary.cpp"
        binary_path = probe_dir / f"{flavor}_canary"
        source_path.write_text(source, encoding="utf-8")
        result = compile_sycl_source(
            source_path,
            binary_path,
            shared=False,
            backend=backend,
            cuda_arch=cuda_arch,
            timeout=180,
        )
        attempt = {
            "flavor": flavor,
            "source_path": str(source_path),
            "binary_path": str(binary_path),
            "compile_success": result.success,
            "compile_output": result.output,
            "command": result.command,
        }
        attempts.append(attempt)
        run_success = False
        run_output = ""
        run_returncode: int | None = None
        if result.success:
            try:
                run_proc = subprocess.run(
                    [str(binary_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    encoding="utf-8",
                    text=True,
                    timeout=30,
                    env=get_sycl_environment(backend=backend, cuda_arch=cuda_arch),
                )
                run_success = run_proc.returncode == 0
                run_output = run_proc.stdout
                run_returncode = run_proc.returncode
            except subprocess.TimeoutExpired as exc:
                run_output = (exc.output or "") + "\nTensor Core probe timed out."
                run_returncode = None
            except Exception as exc:
                run_output = str(exc)
                run_returncode = None
            attempt.update(
                {
                    "run_success": run_success,
                    "run_output": run_output,
                    "run_returncode": run_returncode,
                }
            )

        if result.success and run_success:
            return TensorCoreProbeResult(
                requested=True,
                enabled=True,
                flavor=flavor,
                source_path=str(source_path),
                binary_path=str(binary_path),
                compile_success=True,
                compile_output=result.output,
                command=result.command,
                run_success=True,
                run_output=run_output,
                run_returncode=run_returncode,
                attempts=attempts,
            )

    last = attempts[-1] if attempts else {}
    return TensorCoreProbeResult(
        requested=True,
        enabled=False,
        flavor="",
        source_path=str(last.get("source_path") or ""),
        binary_path=str(last.get("binary_path") or ""),
        compile_success=any(bool(attempt.get("compile_success")) for attempt in attempts),
        compile_output="\n\n".join(
            f"===== {attempt['flavor']} =====\n{attempt.get('compile_output') or ''}" for attempt in attempts
        ),
        command=list(last.get("command") or []),
        run_success=False,
        run_output="\n\n".join(
            f"===== {attempt['flavor']} =====\n{attempt.get('run_output') or ''}" for attempt in attempts
        ),
        run_returncode=last.get("run_returncode"),
        attempts=attempts,
    )


@lru_cache(maxsize=None)
def preload_sycl_runtime(backend: str = "nvidia", cuda_arch: str = "sm_80") -> str | None:
    env = get_sycl_environment(backend=backend, cuda_arch=cuda_arch)
    rtld_flag = getattr(os, "RTLD_GLOBAL", getattr(ctypes, "RTLD_GLOBAL", 0))
    for directory in env.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if not directory:
            continue
        for library_name in ("libsycl.so", "libsycl.so.8", "libsycl-preview.so"):
            path = Path(directory) / library_name
            if path.is_file():
                ctypes.CDLL(str(path), mode=rtld_flag)
                return str(path)
    return None


def normalize_sycl_dtype(dtype: str) -> str:
    normalized = dtype.replace("sycl::", "").replace("const", "")
    return " ".join(normalized.split()).strip()


def get_numpy_dtype(dtype: str) -> Any:
    import numpy as np

    mapping = {
        "half": np.float16,
        "float": np.float32,
        "double": np.float64,
        "int": np.int32,
        "int32_t": np.int32,
        "size_t": np.int64,
    }
    normalized = normalize_sycl_dtype(dtype)
    if normalized not in mapping:
        raise NotImplementedError(f"Unsupported SYCL dtype: {dtype}")
    return mapping[normalized]


def get_ctype(dtype: str) -> Any:
    mapping = {
        "half": ctypes.c_uint16,
        "float": ctypes.c_float,
        "double": ctypes.c_double,
        "int": ctypes.c_int,
        "int32_t": ctypes.c_int32,
        "size_t": ctypes.c_size_t,
    }
    normalized = normalize_sycl_dtype(dtype)
    if normalized not in mapping:
        raise NotImplementedError(f"Unsupported SYCL dtype: {dtype}")
    return mapping[normalized]


def build_benchmark_source(
    source: str,
    *,
    task: GemmTask,
    kernel_tag: str,
    warmup: int,
    iters: int,
) -> str:
    rewritten, metadata = rewrite_kernel_to_event(source, kernel_tag=kernel_tag)
    pointer_params = metadata["pointer_params"]
    size_exprs = get_matmul_size_exprs(metadata, task)
    pointer_name_map = {param["name"]: f"{param['name']}_sycl" for param in pointer_params}

    alloc_lines = []
    h2d_lines = []
    free_lines = []
    for index, param in enumerate(pointer_params):
        storage_dtype = param["storage_dtype"]
        host_name = param["name"]
        device_name = pointer_name_map[host_name]
        size_expr = f"static_cast<size_t>(({size_exprs[index]}))"
        alloc_lines.append(f"{storage_dtype} *{device_name} = sycl::malloc_device<{storage_dtype}>({size_expr}, q);")
        free_lines.append(f"sycl::free({device_name}, q);")
        if index < len(pointer_params) - 1:
            h2d_lines.append(f"q.memcpy({device_name}, {host_name}, {size_expr} * sizeof({storage_dtype}));")

    output_param = pointer_params[-1]
    output_device = pointer_name_map[output_param["name"]]
    output_size = f"static_cast<size_t>(({size_exprs[-1]}))"
    invocation_args = ", ".join(get_invocation_args(metadata, pointer_name_map=pointer_name_map, queue_name="q"))
    extern_c_params = ", ".join(param["full"] for param in metadata["data_params"])

    return f"""
#include <cstdint>
#include <exception>
#include <iostream>
#include <sycl/sycl.hpp>

{rewritten}

extern "C" float timed_gemm_entry({extern_c_params}) {{
  try {{
    sycl::queue q{{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{{}}}};
    {" ".join(alloc_lines)}
    {" ".join(h2d_lines)}
    q.wait();

    for (int i = 0; i < {warmup}; ++i) {{
      sycl::event event = {metadata["kernel_name"]}({invocation_args});
      event.wait();
    }}

    std::uint64_t total_ns = 0;
    for (int i = 0; i < {iters}; ++i) {{
      sycl::event event = {metadata["kernel_name"]}({invocation_args});
      event.wait();
      total_ns += static_cast<std::uint64_t>(
          event.get_profiling_info<sycl::info::event_profiling::command_end>() -
          event.get_profiling_info<sycl::info::event_profiling::command_start>());
    }}

    q.memcpy({output_param["name"]}, {output_device}, {output_size} * sizeof({output_param["storage_dtype"]})).wait();
    {" ".join(free_lines)}
    return static_cast<float>(total_ns / 1.0e6 / {iters});
  }} catch (const sycl::exception &e) {{
    std::cerr << "[SYCL Harness Error] " << e.what() << std::endl;
    return {FAILURE_MS:.1f}f;
  }} catch (const std::exception &e) {{
    std::cerr << "[SYCL Harness Error] " << e.what() << std::endl;
    return {FAILURE_MS:.1f}f;
  }} catch (...) {{
    std::cerr << "[SYCL Harness Error] unknown exception" << std::endl;
    return {FAILURE_MS:.1f}f;
  }}
}}
"""


def build_standalone_source(benchmark_source: str, task: GemmTask) -> str:
    return (
        benchmark_source
        + f"""
#include <vector>

int main() {{
  std::vector<float> A(static_cast<size_t>({task.m}) * {task.k}, 1.0f);
  std::vector<float> B(static_cast<size_t>({task.k}) * {task.n}, 1.0f);
  std::vector<float> C(static_cast<size_t>({task.m}) * {task.n}, 0.0f);
  float ms = timed_gemm_entry(A.data(), B.data(), C.data(), {task.m}, {task.k}, {task.n});
  std::cerr << "[SYCLForge] timed_gemm_entry_ms=" << ms << std::endl;
  return (ms >= {FAILURE_MS:.1f}) ? 1 : 0;
}}
"""
    )


def _make_inputs(task: GemmTask) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np

    rng = np.random.default_rng(task.seed)
    a_mat = rng.standard_normal((task.m, task.k)).astype(np.float32)
    b_mat = rng.standard_normal((task.k, task.n)).astype(np.float32)
    c_mat = np.zeros((task.m, task.n), dtype=np.float32)
    reference = a_mat @ b_mat
    return a_mat, b_mat, c_mat, reference.astype(np.float32)


def benchmark_candidate(
    source: str,
    *,
    task: GemmTask,
    round_idx: int,
    work_dir: Path,
    warmup: int,
    iters: int,
    repeats: int,
    rtol: float,
    atol: float,
    peak_gflops: float,
    backend: str = "nvidia",
    cuda_arch: str = "sm_80",
    dump_harness: bool = True,
) -> BenchResult:
    kernel_tag = f"{task.stem}_r{round_idx}"
    try:
        harness_source = build_benchmark_source(source, task=task, kernel_tag=kernel_tag, warmup=warmup, iters=iters)
    except Exception as exc:
        return BenchResult(error_type="HarnessError", message=str(exc))

    result = BenchResult(harness_source=harness_source)
    with tempfile.TemporaryDirectory(prefix="syclforge_") as tmp:
        tmp_path = Path(tmp)
        src_path = tmp_path / "candidate_harness.cpp"
        so_path = tmp_path / "candidate_harness.so"
        src_path.write_text(harness_source, encoding="utf-8")
        if dump_harness:
            (work_dir / "harness").mkdir(parents=True, exist_ok=True)
            (work_dir / "harness" / f"round_{round_idx:03d}.cpp").write_text(harness_source, encoding="utf-8")

        compile_result = compile_sycl_source(src_path, so_path, shared=True, backend=backend, cuda_arch=cuda_arch)
        result.compile_success = compile_result.success
        result.compile_output = compile_result.output
        if not compile_result.success:
            result.error_type = "CompilationError"
            result.message = compile_result.output[-8000:]
            return result

        try:
            import numpy as np

            preload_sycl_runtime(backend=backend, cuda_arch=cuda_arch)
            rtld_flag = getattr(os, "RTLD_GLOBAL", getattr(ctypes, "RTLD_GLOBAL", 0))
            lib = ctypes.CDLL(str(so_path), mode=rtld_flag)
            entry = lib.timed_gemm_entry
            metadata = parse_sycl_function_metadata(source)
            pointer_ctypes = [ctypes.POINTER(get_ctype(param["dtype"])) for param in metadata["pointer_params"]]
            scalar_ctypes = [get_ctype(param["dtype"]) for param in metadata["scalar_params"]]
            entry.argtypes = pointer_ctypes + scalar_ctypes
            entry.restype = ctypes.c_float

            a_mat, b_mat, c_mat, reference = _make_inputs(task)
            scalar_values = [task.m, task.k, task.n][: len(metadata["scalar_params"])]
            samples: list[float] = []
            for _ in range(repeats):
                c_mat.fill(0.0)
                elapsed = float(
                    entry(
                        a_mat.ctypes.data_as(pointer_ctypes[0]),
                        b_mat.ctypes.data_as(pointer_ctypes[1]),
                        c_mat.ctypes.data_as(pointer_ctypes[2]),
                        *scalar_values,
                    )
                )
                if elapsed >= FAILURE_MS or not math.isfinite(elapsed) or elapsed <= 0.0:
                    raise RuntimeError(f"timed_gemm_entry returned invalid time: {elapsed}")
                samples.append(elapsed)

            np.testing.assert_allclose(c_mat.astype(np.float32), reference, rtol=rtol, atol=atol)
            time_ms = statistics.median(samples)
            gflops = task.flops / (time_ms * 1e6)
            result.runnable = True
            result.correctness_pass = True
            result.time_samples_ms = samples
            result.time_ms = time_ms
            result.gflops = gflops
            result.peak_pct = gflops / peak_gflops * 100.0 if peak_gflops > 0 else None
            result.score = gflops
            return result
        except Exception as exc:
            result.runnable = False
            result.correctness_pass = False
            result.error_type = exc.__class__.__name__
            result.message = str(exc)
            return result


def _benchmark_worker_entry(
    source: str,
    task: GemmTask,
    round_idx: int,
    work_dir: str,
    warmup: int,
    iters: int,
    repeats: int,
    rtol: float,
    atol: float,
    peak_gflops: float,
    backend: str,
    cuda_arch: str,
    dump_harness: bool,
    conn: Any,
) -> None:
    try:
        result = benchmark_candidate(
            source,
            task=task,
            round_idx=round_idx,
            work_dir=Path(work_dir),
            warmup=warmup,
            iters=iters,
            repeats=repeats,
            rtol=rtol,
            atol=atol,
            peak_gflops=peak_gflops,
            backend=backend,
            cuda_arch=cuda_arch,
            dump_harness=dump_harness,
        )
        conn.send(("ok", result.to_dict()))
    except Exception as exc:
        conn.send(
            (
                "err",
                {
                    "compile_success": False,
                    "runnable": False,
                    "correctness_pass": False,
                    "score": None,
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                },
            )
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass


def benchmark_candidate_isolated(
    source: str,
    *,
    task: GemmTask,
    round_idx: int,
    work_dir: Path,
    warmup: int,
    iters: int,
    repeats: int,
    rtol: float,
    atol: float,
    peak_gflops: float,
    backend: str = "nvidia",
    cuda_arch: str = "sm_80",
    dump_harness: bool = True,
) -> BenchResult:
    """Run the benchmark in a subprocess so the parent never owns a CUDA context.

    This matters on A100 systems configured as Exclusive_Process: if the parent
    process keeps a SYCL/CUDA context alive after ctypes benchmarking, the later
    Nsight Compute subprocess cannot acquire the device.
    """
    from multiprocessing import get_context

    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_benchmark_worker_entry,
        args=(
            source,
            task,
            round_idx,
            str(work_dir),
            warmup,
            iters,
            repeats,
            rtol,
            atol,
            peak_gflops,
            backend,
            cuda_arch,
            dump_harness,
            child_conn,
        ),
    )
    process.start()
    try:
        child_conn.close()
    except Exception:
        pass
    process.join()

    payload = parent_conn.recv() if parent_conn.poll() else None
    try:
        parent_conn.close()
    except Exception:
        pass

    if isinstance(payload, tuple) and len(payload) == 2:
        data = payload[1]
        if isinstance(data, dict):
            allowed = set(BenchResult.__dataclass_fields__.keys())
            return BenchResult(**{key: value for key, value in data.items() if key in allowed})

    return BenchResult(
        compile_success=False,
        runnable=False,
        correctness_pass=False,
        error_type="BenchmarkSubprocessCrashed",
        message=f"benchmark subprocess exited without a result; exitcode={process.exitcode}",
    )


def _numeric(value: str) -> float | None:
    if value is None:
        return None
    cleaned = str(value).replace(",", "").replace("%", "").strip()
    if cleaned == "" or cleaned.lower() in {"nan", "n/a"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _kernel_name(row: dict[str, str]) -> str:
    for key in ("Kernel Name", "KernelName", "Name"):
        if key in row:
            return str(row.get(key) or "")
    return ""


def load_ncu_metrics(csv_path: Path, *, kernel_tag: str) -> list[dict[str, Any]]:
    raw_lines = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = [line for line in raw_lines if line and not line.startswith("=")]
    if not lines:
        return []

    reader = csv.DictReader(lines)
    rows: list[dict[str, str]] = [row for row in reader if row]
    if rows:
        first_values = " ".join(str(value).lower() for value in rows[0].values())
        if any(token in first_values for token in ("cycle", "register/thread", "byte", "inst", "%")):
            rows = rows[1:]

    matched = [row for row in rows if kernel_tag in _kernel_name(row)]
    if not matched:
        matched = [
            row
            for row in rows
            if _kernel_name(row)
            and "memcpy" not in _kernel_name(row).lower()
            and "memset" not in _kernel_name(row).lower()
            and "fill" not in _kernel_name(row).lower()
            and "init" not in _kernel_name(row).lower()
        ]

    out: list[dict[str, Any]] = []
    for row in matched:
        record: dict[str, Any] = {"Kernel Name": _kernel_name(row)}
        for metric in NCU_METRICS:
            if metric in row:
                record[metric] = _numeric(row[metric])
        out.append(record)
    return out


def metrics_to_prompt(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "{}"
    keyed: dict[str, Any] = {}
    for row in rows:
        name = str(row.get("Kernel Name", "kernel"))
        value = {key: val for key, val in row.items() if key != "Kernel Name"}
        if name in keyed:
            if not isinstance(keyed[name], list):
                keyed[name] = [keyed[name]]
            keyed[name].append(value)
        else:
            keyed[name] = value
    return json.dumps(keyed, ensure_ascii=False, indent=2)


def profile_candidate_with_ncu(
    source: str,
    *,
    task: GemmTask,
    round_idx: int,
    work_dir: Path,
    warmup: int,
    iters: int,
    backend: str = "nvidia",
    cuda_arch: str = "sm_80",
    timeout: int = 900,
) -> ProfileResult:
    ncu_bin = shutil.which("ncu")
    if not ncu_bin:
        return ProfileResult(success=False, error="ncu not found in PATH")

    kernel_tag = f"{task.stem}_r{round_idx}_profile"
    try:
        benchmark_source = build_benchmark_source(source, task=task, kernel_tag=kernel_tag, warmup=warmup, iters=iters)
        standalone = build_standalone_source(benchmark_source, task)
    except Exception as exc:
        return ProfileResult(success=False, error=f"HarnessError: {exc}")

    profile_dir = work_dir / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    src_path = profile_dir / f"round_{round_idx:03d}_profile.cpp"
    exe_path = profile_dir / f"round_{round_idx:03d}_profile"
    csv_path = profile_dir / f"round_{round_idx:03d}_ncu.csv"
    src_path.write_text(standalone, encoding="utf-8")

    compile_result = compile_sycl_source(src_path, exe_path, shared=False, backend=backend, cuda_arch=cuda_arch, timeout=240)
    if not compile_result.success:
        return ProfileResult(success=False, error=compile_result.output[-8000:], command=compile_result.command)

    def make_cmd(replay_mode: str, csv_file: Path) -> list[str]:
        return [
            ncu_bin,
            "--csv",
            "--page=raw",
            "--kernel-name-base=demangled",
            "--target-processes=all",
            f"--replay-mode={replay_mode}",
            "--profile-from-start=on",
            f"--log-file={csv_file}",
            f"--metrics={','.join(NCU_METRICS)}",
            "--launch-skip=0",
            "--launch-count=30",
            str(exe_path),
        ]

    # SYCL CUDA backend can fail to create a CUDA context under NCU kernel replay
    # on some systems, and that failed attempt can poison the immediately
    # following profiling run on exclusive-process GPUs. Try application replay
    # first because it is the mode that works reliably for this runtime.
    replay_modes = ("application", "kernel")
    attempts: list[str] = []
    try:
        env = get_sycl_environment(backend=backend, cuda_arch=cuda_arch)
        last_cmd: list[str] = []
        last_csv_path = csv_path
        for replay_mode in replay_modes:
            mode_csv_path = profile_dir / f"round_{round_idx:03d}_ncu_{replay_mode}.csv"
            cmd = make_cmd(replay_mode, mode_csv_path)
            last_cmd = cmd
            last_csv_path = mode_csv_path
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                text=True,
                timeout=timeout,
                env=env,
            )
            output = proc.stdout
            attempts.append(f"===== ncu replay-mode={replay_mode} returncode={proc.returncode} =====\n{output}")
            if proc.returncode != 0:
                continue
            rows = load_ncu_metrics(mode_csv_path, kernel_tag=kernel_tag)
            if rows:
                if mode_csv_path != csv_path:
                    try:
                        shutil.copy2(mode_csv_path, csv_path)
                    except Exception:
                        pass
                return ProfileResult(
                    success=True,
                    rows=rows,
                    metrics_block=metrics_to_prompt(rows),
                    csv_path=str(mode_csv_path),
                    command=cmd,
                    output="\n".join(attempts),
                    error="",
                )
            attempts.append(f"ncu replay-mode={replay_mode} completed but no matching kernel rows were extracted.")

        combined_output = "\n".join(attempts)
        return ProfileResult(
            success=False,
            error=combined_output[-12000:],
            command=last_cmd,
            output=combined_output,
            csv_path=str(last_csv_path),
        )
    except subprocess.TimeoutExpired:
        return ProfileResult(success=False, error="ncu profiling timed out", command=last_cmd if "last_cmd" in locals() else [], csv_path=str(last_csv_path if "last_csv_path" in locals() else csv_path))
    except Exception as exc:
        return ProfileResult(success=False, error=str(exc), command=last_cmd if "last_cmd" in locals() else [], csv_path=str(last_csv_path if "last_csv_path" in locals() else csv_path))
