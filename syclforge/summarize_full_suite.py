from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_baseline_detail() -> Path:
    candidates = [
        REPO_ROOT / "benchmark" / "baselines" / "gemm_detail_newprompt_ds.csv",
        REPO_ROOT / "APPT实验存档" / "第二版实验" / "NewPrompt" / "DS" / "results" / "gemm_detail.csv",
        REPO_ROOT / "APPT实验存档" / "第二版实验" / "0405" / "benchmark" / "results" / "gemm_detail.csv",
        REPO_ROOT / "APPT实验存档" / "FixCode0405" / "benchmark" / "results" / "gemm_detail.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _pct(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator * 100.0


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _stem(m: str | int, k: str | int, n: str | int) -> str:
    return f"gemm_{int(m)}_{int(k)}_{int(n)}"


def _load_baseline_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stem = _stem(row["m"], row["k"], row["n"])
            rows[stem] = row
    return rows


def _load_syclforge_summary(run_dir: Path) -> dict[str, dict[str, Any]]:
    summary_json = run_dir / "summary.json"
    if summary_json.is_file():
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        return {item["stem"]: item for item in payload.get("tasks", [])}

    summary_csv = run_dir / "summary.csv"
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Cannot find summary.json or summary.csv under {run_dir}")

    rows: dict[str, dict[str, Any]] = {}
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stem = row["stem"]
            rows[stem] = {
                "stem": stem,
                "best_round": row.get("best_round", ""),
                "seed_gflops": _float_or_none(row.get("seed_gflops")),
                "best_gflops": _float_or_none(row.get("best_gflops")),
                "speedup_vs_seed": _float_or_none(row.get("speedup_vs_seed")),
                "best_path": row.get("best_path", ""),
                "tensor_core_route": {"decision": row.get("tensor_core_decision", "")},
            }
    return rows


def _case_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    group_order = {"S": 0, "K": 1, "N": 2, "M": 3, "A": 4}
    case_id = str(row.get("case_id", ""))
    suffix = "".join(ch for ch in case_id if ch.isdigit())
    return group_order.get(str(row.get("group", "")), 99), int(suffix or 0), case_id


def _build_comparison_rows(
    *,
    syclforge_rows: dict[str, dict[str, Any]],
    baseline_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for stem, baseline in baseline_rows.items():
        tuned = syclforge_rows.get(stem, {})
        group = baseline.get("group", "")
        best = _float_or_none(tuned.get("best_gflops"))
        seed = _float_or_none(tuned.get("seed_gflops"))
        cuda = _float_or_none(baseline.get("cuda_gflops"))
        old_sycl = _float_or_none(baseline.get("our_sycl_gflops"))
        dpcpp = _float_or_none(baseline.get("dpcpp_gflops"))
        tensor_route = tuned.get("tensor_core_route") or {}
        primary_baseline = "OR-CUDA" if group == "A" else "DPCT"
        primary_baseline_gflops = cuda if group == "A" else dpcpp
        output.append(
            {
                "case_id": baseline.get("case_id", ""),
                "group": group,
                "category": baseline.get("category", ""),
                "stem": stem,
                "m": int(baseline["m"]),
                "k": int(baseline["k"]),
                "n": int(baseline["n"]),
                "cuda_gflops": cuda,
                "old_our_sycl_gflops": old_sycl,
                "dpcpp_gflops": dpcpp if group != "A" else None,
                "syclforge_seed_gflops": seed,
                "syclforge_best_gflops": best,
                "speedup_vs_seed": _ratio(best, seed),
                "speedup_vs_old_our_sycl": _ratio(best, old_sycl),
                "retention_vs_cuda_pct": _pct(best, cuda),
                "old_our_sycl_retention_pct": _pct(old_sycl, cuda),
                "dpcpp_retention_pct": _pct(dpcpp, cuda) if group != "A" else None,
                "speedup_vs_dpcpp": _ratio(best, dpcpp) if group != "A" else None,
                "primary_baseline": primary_baseline,
                "primary_baseline_gflops": primary_baseline_gflops,
                "speedup_vs_primary_baseline": _ratio(best, primary_baseline_gflops),
                "retention_vs_primary_baseline_pct": _pct(best, primary_baseline_gflops),
                "tensor_core_decision": tensor_route.get("decision", ""),
                "best_round": tuned.get("best_round", ""),
                "best_path": tuned.get("best_path", ""),
                "has_syclforge_result": best is not None,
                "cuda_status": baseline.get("cuda_status", ""),
                "dpcpp_status": baseline.get("dpcpp_status", "") if group != "A" else "not_applicable",
            }
        )
    return sorted(output, key=_case_sort_key)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "case_id",
        "group",
        "category",
        "stem",
        "m",
        "k",
        "n",
        "cuda_gflops",
        "old_our_sycl_gflops",
        "dpcpp_gflops",
        "syclforge_seed_gflops",
        "syclforge_best_gflops",
        "speedup_vs_seed",
        "speedup_vs_old_our_sycl",
        "retention_vs_cuda_pct",
        "old_our_sycl_retention_pct",
        "dpcpp_retention_pct",
        "speedup_vs_dpcpp",
        "primary_baseline",
        "primary_baseline_gflops",
        "speedup_vs_primary_baseline",
        "retention_vs_primary_baseline_pct",
        "tensor_core_decision",
        "best_round",
        "best_path",
        "has_syclforge_result",
        "cuda_status",
        "dpcpp_status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _median(values: list[float | None]) -> float | None:
    real_values = [value for value in values if value is not None]
    if not real_values:
        return None
    return float(median(real_values))


def _group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = ["S", "K", "N", "M", "A"]
    output = []
    for group in groups:
        group_rows = [row for row in rows if row["group"] == group]
        if not group_rows:
            continue
        output.append(
            {
                "group": group,
                "cases": len(group_rows),
                "syclforge_results": sum(1 for row in group_rows if row["has_syclforge_result"]),
                "median_cuda_gflops": _median([row["cuda_gflops"] for row in group_rows]),
                "median_old_our_sycl_gflops": _median([row["old_our_sycl_gflops"] for row in group_rows]),
                "median_dpcpp_gflops": _median([row["dpcpp_gflops"] for row in group_rows if row["group"] != "A"]),
                "median_syclforge_best_gflops": _median([row["syclforge_best_gflops"] for row in group_rows]),
                "median_speedup_vs_seed": _median([row["speedup_vs_seed"] for row in group_rows]),
                "median_speedup_vs_old_our_sycl": _median([row["speedup_vs_old_our_sycl"] for row in group_rows]),
                "median_retention_vs_cuda_pct": _median([row["retention_vs_cuda_pct"] for row in group_rows]),
                "median_speedup_vs_dpcpp": _median([row["speedup_vs_dpcpp"] for row in group_rows if row["group"] != "A"]),
            }
        )
    return output


def _primary_group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = [
        ("Standard", "DPCT", [row for row in rows if row["group"] != "A"]),
        ("Advanced", "OR-CUDA", [row for row in rows if row["group"] == "A"]),
    ]
    output = []
    for group_name, baseline_name, group_rows in groups:
        if not group_rows:
            continue
        output.append(
            {
                "suite_group": group_name,
                "primary_baseline": baseline_name,
                "cases": len(group_rows),
                "syclforge_results": sum(1 for row in group_rows if row["has_syclforge_result"]),
                "median_primary_baseline_gflops": _median([row["primary_baseline_gflops"] for row in group_rows]),
                "median_syclforge_best_gflops": _median([row["syclforge_best_gflops"] for row in group_rows]),
                "median_speedup_vs_primary_baseline": _median(
                    [row["speedup_vs_primary_baseline"] for row in group_rows]
                ),
                "median_retention_vs_primary_baseline_pct": _median(
                    [row["retention_vs_primary_baseline_pct"] for row in group_rows]
                ),
                "median_speedup_vs_seed": _median([row["speedup_vs_seed"] for row in group_rows]),
            }
        )
    return output


def _write_group_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "group",
        "cases",
        "syclforge_results",
        "median_cuda_gflops",
        "median_old_our_sycl_gflops",
        "median_dpcpp_gflops",
        "median_syclforge_best_gflops",
        "median_speedup_vs_seed",
        "median_speedup_vs_old_our_sycl",
        "median_retention_vs_cuda_pct",
        "median_speedup_vs_dpcpp",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_primary_group_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "suite_group",
        "primary_baseline",
        "cases",
        "syclforge_results",
        "median_primary_baseline_gflops",
        "median_syclforge_best_gflops",
        "median_speedup_vs_primary_baseline",
        "median_retention_vs_primary_baseline_pct",
        "median_speedup_vs_seed",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_primary_case_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "case_id",
        "suite_group",
        "shape",
        "stem",
        "primary_baseline",
        "primary_baseline_gflops",
        "syclforge_seed_gflops",
        "syclforge_best_gflops",
        "speedup_vs_seed",
        "speedup_vs_primary_baseline",
        "retention_vs_primary_baseline_pct",
        "tensor_core_decision",
        "best_round",
        "best_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "suite_group": "Advanced" if row["group"] == "A" else "Standard",
                    "shape": f"{row['m']}x{row['k']}x{row['n']}",
                    "stem": row["stem"],
                    "primary_baseline": row["primary_baseline"],
                    "primary_baseline_gflops": row["primary_baseline_gflops"],
                    "syclforge_seed_gflops": row["syclforge_seed_gflops"],
                    "syclforge_best_gflops": row["syclforge_best_gflops"],
                    "speedup_vs_seed": row["speedup_vs_seed"],
                    "speedup_vs_primary_baseline": row["speedup_vs_primary_baseline"],
                    "retention_vs_primary_baseline_pct": row["retention_vs_primary_baseline_pct"],
                    "tensor_core_decision": row["tensor_core_decision"],
                    "best_round": row["best_round"],
                    "best_path": row["best_path"],
                }
            )


def _write_markdown(path: Path, rows: list[dict[str, Any]], group_rows: list[dict[str, Any]], *, baseline_path: Path, run_dir: Path) -> None:
    lines = [
        "# SYCLForge Full GEMM Suite Summary",
        "",
        f"- SYCLForge run: `{run_dir}`",
        f"- Baseline detail: `{baseline_path}`",
        "",
        "## Group Summary",
        "",
        "| Group | Cases | SYCLForge Results | Median CUDA GFLOPS | Median Old Our-SYCL GFLOPS | Median DPCT GFLOPS | Median SYCLForge GFLOPS | Median Retention vs CUDA | Median Speedup vs Old Our-SYCL | Median Speedup vs DPCT |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in group_rows:
        lines.append(
            "| {group} | {cases} | {syclforge_results} | {cuda} | {old} | {dpcpp} | {best} | {ret} | {old_speed} | {dpct_speed} |".format(
                group=row["group"],
                cases=row["cases"],
                syclforge_results=row["syclforge_results"],
                cuda=_fmt(row["median_cuda_gflops"]),
                old=_fmt(row["median_old_our_sycl_gflops"]),
                dpcpp=_fmt(row["median_dpcpp_gflops"]),
                best=_fmt(row["median_syclforge_best_gflops"]),
                ret=_fmt(row["median_retention_vs_cuda_pct"], 2),
                old_speed=_fmt(row["median_speedup_vs_old_our_sycl"], 4),
                dpct_speed=_fmt(row["median_speedup_vs_dpcpp"], 4),
            )
        )
    lines.extend(
        [
            "",
            "## Case Detail",
            "",
            "| Case | Group | Shape | CUDA GFLOPS | Old Our-SYCL | DPCT | SYCLForge Seed | SYCLForge Best | Retention vs CUDA | Speedup vs Seed | Speedup vs Old Our-SYCL | Speedup vs DPCT | Tensor Route |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {case} | {group} | {shape} | {cuda} | {old} | {dpcpp} | {seed} | {best} | {ret} | {seed_speed} | {old_speed} | {dpct_speed} | {route} |".format(
                case=row["case_id"],
                group=row["group"],
                shape=f"{row['m']}x{row['k']}x{row['n']}",
                cuda=_fmt(row["cuda_gflops"]),
                old=_fmt(row["old_our_sycl_gflops"]),
                dpcpp=_fmt(row["dpcpp_gflops"]),
                seed=_fmt(row["syclforge_seed_gflops"]),
                best=_fmt(row["syclforge_best_gflops"]),
                ret=_fmt(row["retention_vs_cuda_pct"], 2),
                seed_speed=_fmt(row["speedup_vs_seed"], 4),
                old_speed=_fmt(row["speedup_vs_old_our_sycl"], 4),
                dpct_speed=_fmt(row["speedup_vs_dpcpp"], 4),
                route=row["tensor_core_decision"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_primary_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    primary_group_rows: list[dict[str, Any]],
    *,
    baseline_path: Path,
    run_dir: Path,
) -> None:
    lines = [
        "# SYCLForge Primary Baseline Comparison",
        "",
        f"- SYCLForge run: `{run_dir}`",
        f"- Baseline detail: `{baseline_path}`",
        "- Primary comparison: Standard GEMM vs DPCT; Advanced GEMM vs OR-CUDA.",
        "",
        "## Primary Group Summary",
        "",
        "| Suite Group | Primary Baseline | Cases | SYCLForge Results | Median Baseline GFLOPS | Median SYCLForge GFLOPS | Median Speedup vs Baseline | Median Retention vs Baseline |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in primary_group_rows:
        lines.append(
            "| {suite_group} | {baseline} | {cases} | {results} | {baseline_gflops} | {best} | {speedup} | {retention} |".format(
                suite_group=row["suite_group"],
                baseline=row["primary_baseline"],
                cases=row["cases"],
                results=row["syclforge_results"],
                baseline_gflops=_fmt(row["median_primary_baseline_gflops"]),
                best=_fmt(row["median_syclforge_best_gflops"]),
                speedup=_fmt(row["median_speedup_vs_primary_baseline"], 4),
                retention=_fmt(row["median_retention_vs_primary_baseline_pct"], 2),
            )
        )
    lines.extend(
        [
            "",
            "## Primary Case Detail",
            "",
            "| Case | Suite Group | Shape | Primary Baseline | Baseline GFLOPS | SYCLForge Best | Speedup vs Baseline | Retention vs Baseline | Tensor Route |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {case} | {suite_group} | {shape} | {baseline} | {baseline_gflops} | {best} | {speedup} | {retention} | {route} |".format(
                case=row["case_id"],
                suite_group="Advanced" if row["group"] == "A" else "Standard",
                shape=f"{row['m']}x{row['k']}x{row['n']}",
                baseline=row["primary_baseline"],
                baseline_gflops=_fmt(row["primary_baseline_gflops"]),
                best=_fmt(row["syclforge_best_gflops"]),
                speedup=_fmt(row["speedup_vs_primary_baseline"], 4),
                retention=_fmt(row["retention_vs_primary_baseline_pct"], 2),
                route=row["tensor_core_decision"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Summarize a full SYCLForge GEMM suite against old OR-CUDA/DPCT baselines.")
    parser.add_argument("run_dir", type=Path, help="A syclforge run directory containing summary.json or summary.csv.")
    parser.add_argument(
        "--baseline-detail",
        type=Path,
        default=_default_baseline_detail(),
        help="Old experiment gemm_detail.csv containing OR-CUDA, Our-SYCL, and DPCT measurements.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run_dir>/suite_comparison.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    baseline_detail = args.baseline_detail.resolve()
    out_dir = (args.out_dir or (run_dir / "suite_comparison")).resolve()

    baseline_rows = _load_baseline_rows(baseline_detail)
    syclforge_rows = _load_syclforge_summary(run_dir)
    rows = _build_comparison_rows(syclforge_rows=syclforge_rows, baseline_rows=baseline_rows)
    group_rows = _group_summary(rows)
    primary_group_rows = _primary_group_summary(rows)

    _write_csv(out_dir / "suite_case_comparison.csv", rows)
    _write_group_csv(out_dir / "suite_group_summary.csv", group_rows)
    _write_markdown(out_dir / "suite_comparison.md", rows, group_rows, baseline_path=baseline_detail, run_dir=run_dir)
    _write_primary_case_csv(out_dir / "suite_primary_case_comparison.csv", rows)
    _write_primary_group_csv(out_dir / "suite_primary_group_summary.csv", primary_group_rows)
    _write_primary_markdown(
        out_dir / "suite_primary_comparison.md",
        rows,
        primary_group_rows,
        baseline_path=baseline_detail,
        run_dir=run_dir,
    )

    print(f"[SYCLForge] case comparison: {out_dir / 'suite_case_comparison.csv'}")
    print(f"[SYCLForge] group summary: {out_dir / 'suite_group_summary.csv'}")
    print(f"[SYCLForge] markdown: {out_dir / 'suite_comparison.md'}")
    print(f"[SYCLForge] primary case comparison: {out_dir / 'suite_primary_case_comparison.csv'}")
    print(f"[SYCLForge] primary group summary: {out_dir / 'suite_primary_group_summary.csv'}")
    print(f"[SYCLForge] primary markdown: {out_dir / 'suite_primary_comparison.md'}")


if __name__ == "__main__":
    main()
