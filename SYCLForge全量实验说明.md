# SYCLForge 全量 GEMM 调优实验说明

目标：对旧实验里的 20 个 Standard GEMM 和 3 个 Advanced/WMMA GEMM 全部运行 SYCLForge 后优化，并和旧实验的 OR-CUDA baseline 对比；Standard GEMM 额外和 DPCT baseline 对比。

## 一键启动

在远端 A100 机器上进入仓库根目录后执行：

```bash
source /home/l1ngyi/env_sycl.sh
ROUNDS=5 ./run_syclforge_full_gemm_suite.sh
```

脚本默认参数：

```bash
SYCL_PATH=cpu_sycl
WORK_DIR=syclforge_full_runs
WRITE_BACK_DIR=syclforge_best_full
ROUNDS=5
SERVER_TYPE=deepseek
MODEL_NAME=deepseek-v4-pro
MAX_TOKENS=16000
RTOL=2e-2
ATOL=1e-1
BASELINE_DETAIL=benchmark/baselines/gemm_detail_newprompt_ds.csv
```

可以临时覆盖，例如：

```bash
ROUNDS=8 MAX_TOKENS=20000 ./run_syclforge_full_gemm_suite.sh
```

如果只想先验证流程，不接 LLM，可以直接用：

```bash
python -m syclforge.main_sycl cpu_sycl \
  --round 1 \
  --no-llm \
  --tensor-core \
  --rtol 2e-2 \
  --atol 1e-1 \
  --work-dir syclforge_full_runs_smoke
```

## 输出位置

一次完整运行会产生：

| 输出 | 位置 |
| --- | --- |
| SYCLForge 原始运行目录 | `syclforge_full_runs/<timestamp>_batch_<model>/` |
| 每个 case 的最佳代码 | `syclforge_full_runs/<timestamp>_batch_<model>/<case>/best.cpp` |
| 统一写回的最佳代码 | `syclforge_best_full/gemm_*.cpp` |
| SYCLForge 原始汇总 | `syclforge_full_runs/<timestamp>_batch_<model>/summary.csv` |
| 对比 OR-CUDA/DPCT 的 case 级汇总 | `syclforge_full_runs/<timestamp>_batch_<model>/suite_comparison/suite_case_comparison.csv` |
| 对比 OR-CUDA/DPCT 的 group 级汇总 | `syclforge_full_runs/<timestamp>_batch_<model>/suite_comparison/suite_group_summary.csv` |
| Markdown 版结果表 | `syclforge_full_runs/<timestamp>_batch_<model>/suite_comparison/suite_comparison.md` |

## 对比口径

默认内置旧实验结果表来自：

```text
benchmark/baselines/gemm_detail_newprompt_ds.csv
```

汇总脚本会按 `gemm_M_K_N` 对齐：

| 组别 | 比较对象 |
| --- | --- |
| Standard S/K/N/M 共 20 个 | OR-CUDA、旧 Our-SYCL、DPCT、SYCLForge tuned |
| Advanced A1/A2/A3 共 3 个 | OR-CUDA WMMA/Tensor Core baseline、旧 Our-SYCL、SYCLForge tuned |

Advanced 组没有 DPCT 可用结果，所以 `speedup_vs_dpcpp` 留空。

核心指标：

| 指标 | 含义 |
| --- | --- |
| `speedup_vs_seed` | SYCLForge tuned 相对本次 seed 的加速 |
| `speedup_vs_old_our_sycl` | SYCLForge tuned 相对旧论文 Our-SYCL 结果的加速 |
| `retention_vs_cuda_pct` | SYCLForge tuned / OR-CUDA baseline |
| `speedup_vs_dpcpp` | SYCLForge tuned / DPCT，仅 Standard 组 |

## 单独重新汇总

如果全量运行已经完成，只想重新生成对比表：

```bash
RUN=syclforge_full_runs/<timestamp>_batch_deepseek-v4-pro
python -m syclforge.summarize_full_suite "$RUN"
```

如果想换另一份旧实验 baseline：

```bash
python -m syclforge.summarize_full_suite "$RUN" \
  --baseline-detail APPT实验存档/FixCode0405/benchmark/results/gemm_detail.csv
```
