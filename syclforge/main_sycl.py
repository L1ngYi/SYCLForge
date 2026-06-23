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
    parser.add_argument(
        "--resume-run",
        type=Path,
        default=None,
        help="Continue an existing run directory and skip cases that already have a per-case summary.json.",
    )
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
        help="Enable Tensor Core lanes. Use --tensor-core-mode to choose auto routing or force.",
    )
    parser.add_argument(
        "--tensor-core-mode",
        default="auto",
        choices=["auto", "force", "off"],
        help="Tensor Core routing mode when --tensor-core is set.",
    )
    parser.add_argument(
        "--require-tensor-core",
        action="store_true",
        help="Abort if --tensor-core is requested but the TF32 joint_matrix compile probe fails.",
    )
    parser.add_argument(
        "--tensor-core-skeleton",
        type=Path,
        default=None,
        help="Optional known-good joint_matrix source used for a Tensor Core skeleton mutation lane.",
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
    metadata: dict[str, Any] | None = None,
) -> Path:
    code_path = save_text(_round_code_path(task_dir, round_idx, phase), code)
    eval_dir = task_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    payload = {"round": round_idx, "phase": phase, "code_path": str(code_path), "bench": bench, "profile": profile}
    if metadata:
        payload.update(metadata)
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


def _classify_tensor_core_route(task: GemmTask, args: argparse.Namespace) -> dict[str, Any]:
    requested = bool(args.tensor_core or args.require_tensor_core)
    mode = "force" if args.require_tensor_core else args.tensor_core_mode
    aligned = task.m % 16 == 0 and task.n % 16 == 0 and task.k % 8 == 0
    reason_parts = []
    if not requested or mode == "off":
        return {
            "requested": requested,
            "mode": mode,
            "decision": "off",
            "enabled": False,
            "reason": "Tensor Core routing disabled by CLI.",
        }

    if not aligned:
        reason_parts.append("shape is not aligned to M%16==0, N%16==0, K%8==0")
    if task.k < 32:
        reason_parts.append("K is too small for useful TF32 joint_matrix reuse")
    if task.m < 256 or task.n < 256:
        reason_parts.append("M or N is too small to keep enough Tensor Core tiles in flight")

    if mode == "force":
        decision = "force"
        enabled = True
        reason = "forced by --tensor-core-mode force or --require-tensor-core"
    elif not aligned or task.k < 32 or task.m < 256 or task.n < 256:
        decision = "avoid"
        enabled = False
        reason = "; ".join(reason_parts)
    elif task.m >= 1024 and task.n >= 1024 and task.k >= 128:
        decision = "strong"
        enabled = True
        reason = "large aligned GEMM with high K; Tensor Core lane is strongly recommended"
    elif task.m >= 512 and task.n >= 512 and task.k >= 32:
        decision = "try"
        enabled = True
        reason = "aligned GEMM with enough tiles; Tensor Core lane is worth trying"
    else:
        decision = "avoid"
        enabled = False
        reason = "shape is aligned but likely too small to benefit from Tensor Core"

    return {
        "requested": requested,
        "mode": mode,
        "decision": decision,
        "enabled": enabled,
        "reason": reason,
        "aligned_m16_n16_k8": aligned,
        "shape": {"m": task.m, "k": task.k, "n": task.n},
    }


def _load_tensor_core_skeleton(args: argparse.Namespace) -> str:
    path = args.tensor_core_skeleton
    if path is None:
        return ""
    return path.expanduser().resolve().read_text(encoding="utf-8", errors="ignore")


def _candidate_lanes(
    *,
    task_tensor_core_enabled: bool,
    tensor_core_skeleton: str,
) -> list[dict[str, Any]]:
    lanes: list[dict[str, Any]] = [
        {
            "lane": "simt",
            "phase": "candidate_simt",
            "tensor_core_enabled": False,
            "skeleton_code": "",
        }
    ]
    if task_tensor_core_enabled:
        lanes.append(
            {
                "lane": "tensor_free",
                "phase": "candidate_tensor_free",
                "tensor_core_enabled": True,
                "skeleton_code": "",
            }
        )
        if tensor_core_skeleton:
            lanes.append(
                {
                    "lane": "tensor_skeleton",
                    "phase": "candidate_tensor_skeleton",
                    "tensor_core_enabled": True,
                    "skeleton_code": tensor_core_skeleton,
                }
            )
    return lanes


def _make_candidate_item(code: str, *, phase: str, lane: str, tensor_core_enabled: bool) -> dict[str, Any]:
    return {"code": code, "phase": phase, "lane": lane, "tensor_core_enabled": tensor_core_enabled}


def run_task(task: GemmTask, args: argparse.Namespace, batch_dir: Path) -> dict[str, Any]:
    task_dir = batch_dir / task.stem
    task_dir.mkdir(parents=True, exist_ok=True)
    usage_log = task_dir / "usage.csv"
    call_llm = None if args.no_llm else _make_llm_caller(args, usage_log)

    route = getattr(args, "tensor_core_routes", {}).get(task.stem, {"enabled": False, "decision": "off"})
    task_tensor_core_enabled = bool(getattr(args, "tensor_core_enabled", False) and route.get("enabled"))
    route = {**route, "effective_enabled": task_tensor_core_enabled}
    tensor_core_report = getattr(args, "tensor_core_report", {"requested": False, "enabled": False})
    tensor_core_skeleton = getattr(args, "tensor_core_skeleton_code", "")

    seed_code = normalize_sycl_source(task.path.read_text(encoding="utf-8", errors="ignore"))
    best_code = ""
    best_score = float("-inf")
    best_round = -1
    best_simt_code = seed_code
    best_simt_score = float("-inf")
    best_tensor_code = ""
    best_tensor_score = float("-inf")
    seed_gflops: float | None = None
    rounds: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = [
        _make_candidate_item(seed_code, phase="seed", lane="seed", tensor_core_enabled=False)
    ]
    max_rounds = max(1, args.round)

    route_path = save_text(task_dir / "tensor_core_route.json", json.dumps(route, indent=2, ensure_ascii=False))
    print(
        f"[{task.stem}] tensor-core route: {route.get('decision')} "
        f"effective={route.get('effective_enabled')} ({route.get('reason')})",
        flush=True,
    )

    use_isolated_benchmark = bool(args.isolated_benchmark or task_tensor_core_enabled)
    if task_tensor_core_enabled and not args.isolated_benchmark:
        print(f"[{task.stem}] tensor-core mode: using isolated benchmark for native crash containment", flush=True)
    benchmark_fn = benchmark_candidate_isolated if use_isolated_benchmark else benchmark_candidate

    round_idx = 0
    while round_idx < max_rounds and pending:
        item = pending.pop(0)
        current_code = item["code"]
        phase = item["phase"]
        lane = item.get("lane", phase)
        item_tensor_core_enabled = bool(item.get("tensor_core_enabled", False) and task_tensor_core_enabled)
        print(f"[{task.stem}] round {round_idx} {phase} lane={lane}", flush=True)
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
        if bench.correctness_pass and item_tensor_core_enabled and bench.score > best_tensor_score:
            best_tensor_score = bench.score
            best_tensor_code = current_code
        if bench.correctness_pass and not item_tensor_core_enabled and bench.score > best_simt_score:
            best_simt_score = bench.score
            best_simt_code = current_code

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
            metadata={
                "lane": lane,
                "base_lane": item.get("base_lane", ""),
                "candidate_tensor_core_enabled": item_tensor_core_enabled,
                "tensor_core_route": route,
            },
        )
        rounds.append(
            {
                "round": round_idx,
                "phase": phase,
                "lane": lane,
                "code_path": str(code_path),
                "bench": bench_dict,
                "profile": profile_dict,
                "base_lane": item.get("base_lane", ""),
            }
        )

        if args.no_llm or round_idx >= max_rounds - 1:
            round_idx += 1
            break
        assert call_llm is not None

        io_dir = task_dir / "llm_io"
        io_dir.mkdir(parents=True, exist_ok=True)
        remaining_slots = max_rounds - round_idx - 1

        if not bench.correctness_pass:
            if pending:
                round_idx += 1
                continue
            system, prompt = build_repair_prompt(
                task=task,
                gpu_name=args.gpu,
                current_code=current_code,
                bench_result=bench_dict,
                tensor_core_enabled=item_tensor_core_enabled,
                tensor_core_report=tensor_core_report,
                tensor_core_route=route,
                lane=lane,
                rtol=args.rtol,
                atol=args.atol,
            )
            save_text(io_dir / f"round_{round_idx:03d}_repair_prompt.txt", prompt)
            raw = call_llm(prompt, system, call_type="repair", round_idx=round_idx)
            save_text(io_dir / f"round_{round_idx:03d}_repair_reply.txt", raw)
            repair_code = _normalize_generated_reply(
                raw,
                previous_code=current_code,
                error_path=io_dir / f"round_{round_idx:03d}_repair_parse_error.json",
            )
            pending.append(
                _make_candidate_item(
                    repair_code,
                    phase=f"repair_{lane}",
                    lane=f"{lane}_repair",
                    tensor_core_enabled=item_tensor_core_enabled,
                )
            )
            round_idx += 1
            continue

        if pending:
            round_idx += 1
            continue

        next_lanes = _candidate_lanes(
            task_tensor_core_enabled=task_tensor_core_enabled,
            tensor_core_skeleton=tensor_core_skeleton,
        )
        if str(lane).startswith("tensor"):
            next_lanes = [item for item in next_lanes if item["tensor_core_enabled"]] + [
                item for item in next_lanes if not item["tensor_core_enabled"]
            ]

        for lane_cfg in next_lanes[:remaining_slots]:
            lane_name = lane_cfg["lane"]
            lane_tensor_enabled = bool(lane_cfg["tensor_core_enabled"])
            lane_base_code = current_code
            base_lane = lane
            if lane_name == "simt":
                lane_base_code = best_simt_code or seed_code
                base_lane = "best_simt"
            elif lane_tensor_enabled:
                lane_base_code = best_tensor_code or best_code or current_code
                base_lane = "best_tensor" if best_tensor_code else "best_overall"
            judge_system, judge_prompt = build_judge_prompt(
                task=task,
                gpu_name=args.gpu,
                current_code=lane_base_code,
                bench_result=bench_dict,
                ncu_metrics_block=profile_block,
                tensor_core_enabled=lane_tensor_enabled,
                tensor_core_report=tensor_core_report,
                tensor_core_route={**route, "lane": lane_name},
                lane=lane_name,
                rtol=args.rtol,
                atol=args.atol,
            )
            save_text(io_dir / f"round_{round_idx:03d}_{lane_name}_judge_prompt.txt", judge_prompt)
            judge_raw = call_llm(judge_prompt, judge_system, call_type=f"judge_{lane_name}", round_idx=round_idx)
            save_text(io_dir / f"round_{round_idx:03d}_{lane_name}_judge_reply.txt", judge_raw)
            strategy = _safe_json_from_reply(judge_raw)

            opt_system, opt_prompt = build_optimization_prompt(
                task=task,
                gpu_name=args.gpu,
                current_code=lane_base_code,
                bench_result=bench_dict,
                ncu_metrics_block=profile_block,
                strategy=strategy,
                tensor_core_enabled=lane_tensor_enabled,
                tensor_core_report=tensor_core_report,
                tensor_core_route={**route, "lane": lane_name},
                lane=lane_name,
                skeleton_code=lane_cfg.get("skeleton_code") or "",
                rtol=args.rtol,
                atol=args.atol,
            )
            save_text(io_dir / f"round_{round_idx:03d}_{lane_name}_opt_prompt.txt", opt_prompt)
            opt_raw = call_llm(opt_prompt, opt_system, call_type=f"optimization_{lane_name}", round_idx=round_idx)
            save_text(io_dir / f"round_{round_idx:03d}_{lane_name}_opt_reply.txt", opt_raw)
            next_code = _normalize_generated_reply(
                opt_raw,
                previous_code=lane_base_code,
                error_path=io_dir / f"round_{round_idx:03d}_{lane_name}_opt_parse_error.json",
            )
            pending.append(
                {
                    **_make_candidate_item(
                        next_code,
                        phase=lane_cfg["phase"],
                        lane=lane_name,
                        tensor_core_enabled=lane_tensor_enabled,
                    ),
                    "base_lane": base_lane,
                }
            )
        round_idx += 1

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
        "tensor_core_route": route,
        "tensor_core_route_path": str(route_path),
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
        writer.writerow(["stem", "best_round", "seed_gflops", "best_gflops", "speedup_vs_seed", "tensor_core_decision", "best_path"])
        for item in summaries:
            writer.writerow(
                [
                    item["stem"],
                    item["best_round"],
                    item.get("seed_gflops") or "",
                    item.get("best_gflops") or "",
                    item.get("speedup_vs_seed") or "",
                    (item.get("tensor_core_route") or {}).get("decision", ""),
                    item.get("best_path") or "",
                ]
            )


def _load_completed_summary(batch_dir: Path, task: GemmTask) -> dict[str, Any] | None:
    summary_path = batch_dir / task.stem / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("stem") != task.stem:
        return None
    return payload


def configure_tensor_core_mode(args: argparse.Namespace, batch_dir: Path, tasks: list[GemmTask]) -> None:
    args.tensor_core_enabled = False
    args.tensor_core_report = {"requested": bool(args.tensor_core or args.require_tensor_core), "enabled": False}
    if args.require_tensor_core:
        args.tensor_core = True
        args.tensor_core_mode = "force"
    args.tensor_core_routes = {task.stem: _classify_tensor_core_route(task, args) for task in tasks}
    args.tensor_core_skeleton_code = _load_tensor_core_skeleton(args)
    save_text(batch_dir / "tensor_core_routes.json", json.dumps(args.tensor_core_routes, indent=2, ensure_ascii=False))

    if not args.tensor_core or args.tensor_core_mode == "off":
        return

    if not any(route.get("enabled") for route in args.tensor_core_routes.values()):
        print("[SYCLForge] tensor-core requested, but shape router selected SIMT-only lanes for all tasks.", flush=True)
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
    if args.resume_run is not None:
        batch_dir = args.resume_run.resolve()
        batch_dir.mkdir(parents=True, exist_ok=True)
    else:
        batch_dir = _make_batch_dir(args)
    print(f"[SYCLForge] output: {batch_dir}", flush=True)
    print(f"[SYCLForge] tasks: {len(tasks)}", flush=True)
    if args.resume_run is not None:
        print("[SYCLForge] resume mode: completed cases with summary.json will be skipped.", flush=True)
    configure_tensor_core_mode(args, batch_dir, tasks)

    summaries = []
    for index, task in enumerate(tasks, 1):
        print(f"\n===== [{index}/{len(tasks)}] {task.stem} =====", flush=True)
        completed = _load_completed_summary(batch_dir, task) if args.resume_run is not None else None
        if completed is not None:
            print(f"[{task.stem}] resume: found existing summary.json; skipping.", flush=True)
            summaries.append(completed)
            save_batch_summary(batch_dir, summaries)
            continue
        summaries.append(run_task(task, args, batch_dir))
        save_batch_summary(batch_dir, summaries)

    save_batch_summary(batch_dir, summaries)
    elapsed = time.time() - started
    print(f"[SYCLForge] done in {elapsed:.1f}s", flush=True)
    print(f"[SYCLForge] summary: {batch_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
