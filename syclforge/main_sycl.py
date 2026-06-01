from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from syclforge.code_io import normalize_sycl_source, save_text
from syclforge.prompts import build_judge_prompt, build_optimization_prompt, build_repair_prompt
from syclforge.sycl_tools import (
    benchmark_candidate,
    benchmark_candidate_isolated,
    probe_tensor_core_support,
    profile_candidate_with_ncu,
)
from syclforge.tasks import GemmTask, discover_tasks


REPO_ROOT = Path(__file__).resolve().parents[1]
CUDAFORGE_ROOT = REPO_ROOT / "CudaForge" if (REPO_ROOT / "CudaForge" / "agents").is_dir() else REPO_ROOT


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("SYCLForge: CudaForge-style optimization for SYCL GEMM kernels")
    parser.add_argument("sycl_path", nargs="?", default="cpu_sycl", type=Path, help="A gemm_*.cpp file or a directory containing them.")
    parser.add_argument("--case-stem", default="", help="Only run one case, e.g. gemm_1024_1024_1024.")
    parser.add_argument("--first-n", type=int, default=0, help="Run the first N sorted GEMM cases.")
    parser.add_argument("--round", "-G", type=int, default=6, help="Total rounds including the seed evaluation.")
    parser.add_argument("--work-dir", type=Path, default=Path("syclforge_runs"), help="Output root directory.")
    parser.add_argument("--write-back-dir", type=Path, default=None, help="Optional directory to receive best gemm_*.cpp files.")
    parser.add_argument("--gpu", default="A100", help="GPU name used in prompts.")
    parser.add_argument("--backend", default="nvidia", choices=["nvidia", "generic"], help="SYCL backend preset.")
    parser.add_argument("--cuda-arch", default="sm_80", help="CUDA backend architecture for A100.")
    parser.add_argument("--peak-gflops", type=float, default=19.5 * 1000.0, help="A100 FP32 peak throughput in GFLOPS.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup kernel launches inside the harness.")
    parser.add_argument("--iters", type=int, default=20, help="Timed kernel launches per harness call.")
    parser.add_argument("--repeats", type=int, default=5, help="Independent harness calls; median is reported.")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--no-ncu", action="store_true", help="Disable Nsight Compute feedback and use latency-only feedback.")
    parser.add_argument(
        "--tensor-core",
        action="store_true",
        help="Enable Tensor Core oriented prompts after a TF32 joint_matrix compile probe succeeds.",
    )
    parser.add_argument(
        "--require-tensor-core",
        action="store_true",
        help="Abort if --tensor-core is requested but the TF32 joint_matrix compile probe fails.",
    )
    parser.add_argument(
        "--isolated-benchmark",
        action="store_true",
        help="Run benchmarking in a subprocess. Useful for GPUs in Exclusive_Process compute mode.",
    )
    parser.add_argument("--no-llm", action="store_true", help="Only evaluate the existing seed kernels.")
    parser.add_argument("--server_type", default="local", help="CudaForge query_server provider.")
    parser.add_argument("--server_address", default="localhost")
    parser.add_argument("--server_port", type=int, default=8000)
    parser.add_argument("--model_name", default="deepseek-ai/deepseek-coder-6.7b-instruct")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=1.0)
    return parser


def _make_batch_dir(args: argparse.Namespace) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.case_stem or ("single" if args.sycl_path.is_file() else "batch")
    model_tag = str(args.model_name).replace("/", "_").replace(":", "_")
    batch_dir = (args.work_dir / f"{stamp}_{tag}_{model_tag}").resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir


def _make_llm_caller(args: argparse.Namespace, usage_log: Path):
    if str(CUDAFORGE_ROOT) not in sys.path:
        sys.path.insert(0, str(CUDAFORGE_ROOT))
    from agents.query_server import query_server

    def call(prompt: str, system_prompt: str, *, call_type: str, round_idx: int) -> str:
        result = query_server(
            prompt=prompt,
            system_prompt=system_prompt,
            server_type=args.server_type,
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            server_address=args.server_address,
            server_port=args.server_port,
            log_path=str(usage_log),
            call_type=call_type,
            round_idx=round_idx,
        )
        if isinstance(result, list):
            return str(result[0]) if result else ""
        return str(result)

    return call


def _safe_json_from_reply(raw: str) -> Any:
    try:
        if str(CUDAFORGE_ROOT) not in sys.path:
            sys.path.insert(0, str(CUDAFORGE_ROOT))
        from utils.kernel_io import extract_json

        return extract_json(raw)
    except Exception:
        return {"raw_strategy": raw.strip()[:2000]}


def _round_code_path(task_dir: Path, round_idx: int, phase: str) -> Path:
    return task_dir / "code" / f"round_{round_idx:03d}_{phase}.cpp"


def _save_round_artifacts(
    task_dir: Path,
    *,
    round_idx: int,
    phase: str,
    code: str,
    bench: dict[str, Any],
    profile: dict[str, Any] | None,
) -> Path:
    code_path = save_text(_round_code_path(task_dir, round_idx, phase), code)
    eval_dir = task_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    payload = {"round": round_idx, "phase": phase, "code_path": str(code_path), "bench": bench, "profile": profile}
    save_text(eval_dir / f"round_{round_idx:03d}_{phase}.json", json.dumps(payload, indent=2, ensure_ascii=False))
    return code_path


def _normalize_generated_reply(raw: str, *, previous_code: str, error_path: Path) -> str:
    try:
        return normalize_sycl_source(raw)
    except Exception as exc:
        save_text(
            error_path,
            json.dumps(
                {
                    "error": str(exc),
                    "note": "Keeping the previous candidate because the LLM reply was not valid SYCL source.",
                    "raw_reply_prefix": raw[:4000],
                },
                indent=2,
                ensure_ascii=False,
            ),
        )
        return previous_code


def _profile_if_enabled(
    code: str,
    *,
    task: GemmTask,
    round_idx: int,
    task_dir: Path,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any] | None]:
    if args.no_ncu:
        return "{}", {"success": False, "error": "disabled by --no-ncu"}
    profile = profile_candidate_with_ncu(
        code,
        task=task,
        round_idx=round_idx,
        work_dir=task_dir,
        warmup=args.warmup,
        iters=min(args.iters, 20),
        backend=args.backend,
        cuda_arch=args.cuda_arch,
    )
    return profile.metrics_block if profile.success else json.dumps({"profile_error": profile.error}, indent=2), profile.to_dict()


def run_task(task: GemmTask, args: argparse.Namespace, batch_dir: Path) -> dict[str, Any]:
    if not hasattr(args, "tensor_core_enabled"):
        args.tensor_core_enabled = False
    if not hasattr(args, "tensor_core_report"):
        args.tensor_core_report = {"requested": False, "enabled": False}

    task_dir = batch_dir / task.stem
    task_dir.mkdir(parents=True, exist_ok=True)
    usage_log = task_dir / "usage.csv"
    call_llm = None if args.no_llm else _make_llm_caller(args, usage_log)

    current_code = normalize_sycl_source(task.path.read_text(encoding="utf-8", errors="ignore"))
    best_code = ""
    best_score = float("-inf")
    best_round = -1
    seed_gflops: float | None = None
    rounds: list[dict[str, Any]] = []
    use_isolated_benchmark = bool(args.isolated_benchmark or args.tensor_core_enabled)
    if args.tensor_core_enabled and not args.isolated_benchmark:
        print(f"[{task.stem}] tensor-core mode: using isolated benchmark for native crash containment", flush=True)
    benchmark_fn = benchmark_candidate_isolated if use_isolated_benchmark else benchmark_candidate

    for round_idx in range(max(1, args.round)):
        phase = "seed" if round_idx == 0 else "candidate"
        print(f"[{task.stem}] round {round_idx} {phase}", flush=True)
        bench = benchmark_fn(
            current_code,
            task=task,
            round_idx=round_idx,
            work_dir=task_dir,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
            rtol=args.rtol,
            atol=args.atol,
            peak_gflops=args.peak_gflops,
            backend=args.backend,
            cuda_arch=args.cuda_arch,
        )
        bench_dict = bench.to_dict()
        if round_idx == 0 and bench.gflops is not None:
            seed_gflops = bench.gflops
        if bench.correctness_pass and bench.score > best_score:
            best_score = bench.score
            best_code = current_code
            best_round = round_idx

        profile_block = "{}"
        profile_dict = None
        if bench.correctness_pass:
            profile_block, profile_dict = _profile_if_enabled(
                current_code,
                task=task,
                round_idx=round_idx,
                task_dir=task_dir,
                args=args,
            )

        code_path = _save_round_artifacts(
            task_dir,
            round_idx=round_idx,
            phase=phase,
            code=current_code,
            bench=bench_dict,
            profile=profile_dict,
        )
        rounds.append({"round": round_idx, "phase": phase, "code_path": str(code_path), "bench": bench_dict, "profile": profile_dict})

        if args.no_llm or round_idx == args.round - 1:
            break
        assert call_llm is not None

        io_dir = task_dir / "llm_io"
        io_dir.mkdir(parents=True, exist_ok=True)
        if not bench.correctness_pass:
            system, prompt = build_repair_prompt(
                task=task,
                gpu_name=args.gpu,
                current_code=current_code,
                bench_result=bench_dict,
                tensor_core_enabled=args.tensor_core_enabled,
                tensor_core_report=args.tensor_core_report,
                rtol=args.rtol,
                atol=args.atol,
            )
            save_text(io_dir / f"round_{round_idx:03d}_repair_prompt.txt", prompt)
            raw = call_llm(prompt, system, call_type="repair", round_idx=round_idx)
            save_text(io_dir / f"round_{round_idx:03d}_repair_reply.txt", raw)
            current_code = _normalize_generated_reply(
                raw,
                previous_code=current_code,
                error_path=io_dir / f"round_{round_idx:03d}_repair_parse_error.json",
            )
            continue

        judge_system, judge_prompt = build_judge_prompt(
            task=task,
            gpu_name=args.gpu,
            current_code=current_code,
            bench_result=bench_dict,
            ncu_metrics_block=profile_block,
            tensor_core_enabled=args.tensor_core_enabled,
            tensor_core_report=args.tensor_core_report,
            rtol=args.rtol,
            atol=args.atol,
        )
        save_text(io_dir / f"round_{round_idx:03d}_judge_prompt.txt", judge_prompt)
        judge_raw = call_llm(judge_prompt, judge_system, call_type="judge_optimization", round_idx=round_idx)
        save_text(io_dir / f"round_{round_idx:03d}_judge_reply.txt", judge_raw)
        strategy = _safe_json_from_reply(judge_raw)

        opt_system, opt_prompt = build_optimization_prompt(
            task=task,
            gpu_name=args.gpu,
            current_code=current_code,
            bench_result=bench_dict,
            ncu_metrics_block=profile_block,
            strategy=strategy,
            tensor_core_enabled=args.tensor_core_enabled,
            tensor_core_report=args.tensor_core_report,
            rtol=args.rtol,
            atol=args.atol,
        )
        save_text(io_dir / f"round_{round_idx:03d}_opt_prompt.txt", opt_prompt)
        opt_raw = call_llm(opt_prompt, opt_system, call_type="optimization", round_idx=round_idx)
        save_text(io_dir / f"round_{round_idx:03d}_opt_reply.txt", opt_raw)
        current_code = _normalize_generated_reply(
            opt_raw,
            previous_code=current_code,
            error_path=io_dir / f"round_{round_idx:03d}_opt_parse_error.json",
        )

    if best_code:
        best_path = save_text(task_dir / "best.cpp", best_code)
    else:
        best_path = Path("")

    if args.write_back_dir is not None and best_code:
        write_back_path = args.write_back_dir / f"{task.stem}.cpp"
        save_text(write_back_path, best_code)
    else:
        write_back_path = None

    speedup = best_score / seed_gflops if seed_gflops and seed_gflops > 0 and best_score > 0 else None
    summary = {
        "task": str(task.path),
        "stem": task.stem,
        "shape": {"m": task.m, "k": task.k, "n": task.n},
        "best_round": best_round,
        "best_gflops": best_score if best_score != float("-inf") else None,
        "seed_gflops": seed_gflops,
        "speedup_vs_seed": speedup,
        "best_path": str(best_path) if best_code else "",
        "write_back_path": str(write_back_path) if write_back_path else "",
        "rounds": rounds,
    }
    save_text(task_dir / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def save_batch_summary(batch_dir: Path, summaries: list[dict[str, Any]]) -> None:
    save_text(batch_dir / "summary.json", json.dumps({"tasks": summaries}, indent=2, ensure_ascii=False))
    csv_path = batch_dir / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stem", "best_round", "seed_gflops", "best_gflops", "speedup_vs_seed", "best_path"])
        for item in summaries:
            writer.writerow(
                [
                    item["stem"],
                    item["best_round"],
                    item.get("seed_gflops") or "",
                    item.get("best_gflops") or "",
                    item.get("speedup_vs_seed") or "",
                    item.get("best_path") or "",
                ]
            )


def configure_tensor_core_mode(args: argparse.Namespace, batch_dir: Path) -> None:
    args.tensor_core_enabled = False
    args.tensor_core_report = {"requested": bool(args.tensor_core or args.require_tensor_core), "enabled": False}
    if args.require_tensor_core:
        args.tensor_core = True
    if not args.tensor_core:
        return

    probe = probe_tensor_core_support(batch_dir, backend=args.backend, cuda_arch=args.cuda_arch)
    args.tensor_core_enabled = probe.enabled
    args.tensor_core_report = probe.to_dict()
    save_text(batch_dir / "tensor_core_probe.json", json.dumps(args.tensor_core_report, indent=2, ensure_ascii=False))

    if probe.enabled:
        print(f"[SYCLForge] tensor-core mode enabled via {probe.flavor}", flush=True)
        return

    print("[SYCLForge] tensor-core probe failed; falling back to SIMT prompts.", flush=True)
    print(f"[SYCLForge] tensor-core probe log: {probe.source_path}", flush=True)
    if args.require_tensor_core:
        raise SystemExit("Tensor Core mode was required, but the TF32 joint_matrix compile probe failed.")


def main() -> None:
    args = build_arg_parser().parse_args()
    started = time.time()
    tasks = discover_tasks(args.sycl_path.resolve(), case_stem=args.case_stem, first_n=args.first_n)
    batch_dir = _make_batch_dir(args)
    print(f"[SYCLForge] output: {batch_dir}", flush=True)
    print(f"[SYCLForge] tasks: {len(tasks)}", flush=True)
    configure_tensor_core_mode(args, batch_dir)

    summaries = []
    for index, task in enumerate(tasks, 1):
        print(f"\n===== [{index}/{len(tasks)}] {task.stem} =====", flush=True)
        summaries.append(run_task(task, args, batch_dir))

    save_batch_summary(batch_dir, summaries)
    elapsed = time.time() - started
    print(f"[SYCLForge] done in {elapsed:.1f}s", flush=True)
    print(f"[SYCLForge] summary: {batch_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
