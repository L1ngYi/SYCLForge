# 实验 baseline 与转译文件整理

这份整理对应旧的 SYCL adaptation paper 实验。核心实验脚本是：

- `APPT实验存档/FixCode0405/run_sycl_paper_gemm_suite.sh`
- `APPT实验存档/FixCode0405/benchmark/tools/run_sycl_paper_gemm_suite.py`

脚本默认比较三路代码：

| 结果名 | 含义 | 默认目录 | 数量 | 备注 |
| --- | --- | --- | ---: | --- |
| OR-CUDA | 原始 CUDA baseline | `APPT实验存档/FixCode0405/benchmark/data/cuda_baseline_gemm` | 23 | S/K/N/M 是 FP32 shared-memory tiled CUDA；A 组是 WMMA/Tensor Core CUDA |
| Our-SYCL | 我们转译后的 SYCL | `APPT实验存档/FixCode0405/cpu_sycl` | 23 | Falcon CPU -> SYCL 的输出，也是论文里的 Our-SYCL |
| DPCT | Intel DPCT / DPC++ 迁移 baseline | `APPT实验存档/FixCode0405/benchmark/data/dpcpp_baseline_gemm` | 20 | 只有 S/K/N/M；A 组三个 WMMA case 迁移失败/缺失 |

辅助目录：

| 目录 | 作用 |
| --- | --- |
| `APPT实验存档/FixCode0405/benchmark/data/cpp_baseline_gemm` | CPU baseline / Falcon CPU->SYCL 的源输入，不是主要性能对比线 |
| `APPT实验存档/FixCode0405/benchmark/data/sycl_baseline_gemm` | 存在的历史 SYCL baseline 目录，但默认实验脚本没有把它作为结果表里的单独方法 |

## 实验流程

当时的流程在 `APPT实验存档/FixCode0405/artifact.md` 里也写过：

1. `bash cpu_to_sycl_falcon.sh`
   生成 `cpu_sycl/*.cpp`，即 Our-SYCL。
2. `bash cuda_to_dpcpp_batch.sh`
   生成 `benchmark/data/dpcpp_baseline_gemm/*.cpp`，即 DPCT baseline。
3. `bash run_sycl_paper_gemm_suite.sh`
   统一评测 OR-CUDA / Our-SYCL / DPCT。

## Standard GEMM: S/K/N/M 组

这 20 个 standard GEMM 都有三路文件：OR-CUDA、Our-SYCL、DPCT。

| Case | Group | Shape MxKxN | OR-CUDA | Our-SYCL | DPCT |
| --- | --- | --- | --- | --- | --- |
| S1 | Square | 128x128x128 | `benchmark/data/cuda_baseline_gemm/gemm_128_128_128.cu` | `cpu_sycl/gemm_128_128_128.cpp` | `benchmark/data/dpcpp_baseline_gemm/gemm_128_128_128.cpp` |
| S2 | Square | 256x256x256 | `benchmark/data/cuda_baseline_gemm/gemm_256_256_256.cu` | `cpu_sycl/gemm_256_256_256.cpp` | `benchmark/data/dpcpp_baseline_gemm/gemm_256_256_256.cpp` |
| S3 | Square | 512x512x512 | `benchmark/data/cuda_baseline_gemm/gemm_512_512_512.cu` | `cpu_sycl/gemm_512_512_512.cpp` | `benchmark/data/dpcpp_baseline_gemm/gemm_512_512_512.cpp` |
| S4 | Square | 1024x1024x1024 | `benchmark/data/cuda_baseline_gemm/gemm_1024_1024_1024.cu` | `cpu_sycl/gemm_1024_1024_1024.cpp` | `benchmark/data/dpcpp_baseline_gemm/gemm_1024_1024_1024.cpp` |
| S5 | Square | 2048x2048x2048 | `benchmark/data/cuda_baseline_gemm/gemm_2048_2048_2048.cu` | `cpu_sycl/gemm_2048_2048_2048.cpp` | `benchmark/data/dpcpp_baseline_gemm/gemm_2048_2048_2048.cpp` |
| K1 | Small-K | 1024x16x1024 | `benchmark/data/cuda_baseline_gemm/gemm_1024_16_1024.cu` | `cpu_sycl/gemm_1024_16_1024.cpp` | `benchmark/data/dpcpp_baseline_gemm/gemm_1024_16_1024.cpp` |
| K2 | Small-K | 1024x32x1024 | `benchmark/data/cuda_baseline_gemm/gemm_1024_32_1024.cu` | `cpu_sycl/gemm_1024_32_1024.cpp` | `benchmark/data/dpcpp_baseline_gemm/gemm_1024_32_1024.cpp` |
| K3 | Small-K | 1024x64x1024 | `benchmark/data/cuda_baseline_gemm/gemm_1024_64_1024.cu` | `cpu_sycl/gemm_1024_64_1024.cpp` | `benchmark/data/dpcpp_baseline_gemm/gemm_1024_64_1024.cpp` |
| K4 | Small-K | 2048x32x2048 | `benchmark/data/cuda_baseline_gemm/gemm_2048_32_2048.cu` | `cpu_sycl/gemm_2048_32_2048.cpp` | `benchmark/data/dpcpp_baseline_gemm/gemm_2048_32_2048.cpp` |
| K5 | Small-K | 2048x64x2048 | `benchmark/data/cuda_baseline_gemm/gemm_2048_64_2048.cu` | `cpu_sycl/gemm_2048_64_2048.cpp` | `benchmark/data/dpcpp_baseline_gemm/gemm_2048_64_2048.cpp` |
| N1 | Wide-N | 256x256x2048 | `benchmark/data/cuda_baseline_gemm/gemm_256_256_2048.cu` | `cpu_sycl/gemm_256_256_2048.cpp` | `benchmark/data/dpcpp_baseline_gemm/gemm_256_256_2048.cpp` |
| N2 | Wide-N | 512x256x4096 | `benchmark/data/cuda_baseline_gemm/gemm_512_256_4096.cu` | `cpu_sycl/gemm_512_256_4096.cpp` | `benchmark/data/dpcpp_baseline_gemm/gemm_512_256_4096.cpp` |
| N3 | Wide-N | 1024x256x4096 | `benchmark/data/cuda_baseline_gemm/gemm_1024_256_4096.cu` | `cpu_sycl/gemm_1024_256_4096.cpp` | `benchmark/data/dpcpp_baseline_gemm/gemm_1024_256_4096.cpp` |
| N4 | Wide-N | 2048x256x4096 | `benchmark/data/cuda_baseline_gemm/gemm_2048_256_4096.cu` | `cpu_sycl/gemm_2048_256_4096.cpp` | `benchmark/data/dpcpp_baseline_gemm/gemm_2048_256_4096.cpp` |
| N5 | Wide-N | 4096x128x4096 | `benchmark/data/cuda_baseline_gemm/gemm_4096_128_4096.cu` | `cpu_sycl/gemm_4096_128_4096.cpp` | `benchmark/data/dpcpp_baseline_gemm/gemm_4096_128_4096.cpp` |
| M1 | Tall-M | 2048x256x256 | `benchmark/data/cuda_baseline_gemm/gemm_2048_256_256.cu` | `cpu_sycl/gemm_2048_256_256.cpp` | `benchmark/data/dpcpp_baseline_gemm/gemm_2048_256_256.cpp` |
| M2 | Tall-M | 4096x256x256 | `benchmark/data/cuda_baseline_gemm/gemm_4096_256_256.cu` | `cpu_sycl/gemm_4096_256_256.cpp` | `benchmark/data/dpcpp_baseline_gemm/gemm_4096_256_256.cpp` |
| M3 | Tall-M | 4096x512x256 | `benchmark/data/cuda_baseline_gemm/gemm_4096_512_256.cu` | `cpu_sycl/gemm_4096_512_256.cpp` | `benchmark/data/dpcpp_baseline_gemm/gemm_4096_512_256.cpp` |
| M4 | Tall-M | 4096x1024x256 | `benchmark/data/cuda_baseline_gemm/gemm_4096_1024_256.cu` | `cpu_sycl/gemm_4096_1024_256.cpp` | `benchmark/data/dpcpp_baseline_gemm/gemm_4096_1024_256.cpp` |
| M5 | Tall-M | 4096x1024x512 | `benchmark/data/cuda_baseline_gemm/gemm_4096_1024_512.cu` | `cpu_sycl/gemm_4096_1024_512.cpp` | `benchmark/data/dpcpp_baseline_gemm/gemm_4096_1024_512.cpp` |

## Advanced / WMMA 组

这 3 个是当时用来覆盖 CUDA WMMA / Tensor Core 风格代码的 advanced case。OR-CUDA 是 WMMA CUDA baseline，Our-SYCL 是我们转译出的 SYCL。DPCT 在这组三个 case 上没有可用输出，结果表里记为 `ERROR`。

| Case | Shape MxKxN | OR-CUDA WMMA baseline | Our-SYCL 转译结果 | DPCT |
| --- | --- | --- | --- | --- |
| A1 | 4096x4096x4096 | `benchmark/data/cuda_baseline_gemm/gemm_4096_4096_4096.cu` | `cpu_sycl/gemm_4096_4096_4096.cpp` | missing / ERROR |
| A2 | 2048x16384x2048 | `benchmark/data/cuda_baseline_gemm/gemm_2048_16384_2048.cu` | `cpu_sycl/gemm_2048_16384_2048.cpp` | missing / ERROR |
| A3 | 8192x1024x8192 | `benchmark/data/cuda_baseline_gemm/gemm_8192_1024_8192.cu` | `cpu_sycl/gemm_8192_1024_8192.cpp` | missing / ERROR |

## 结果文件

规范存档版结果：

- `APPT实验存档/FixCode0405/benchmark/results/gemm_detail.csv`
- `APPT实验存档/FixCode0405/benchmark/results/performance_summary.csv`
- `APPT实验存档/FixCode0405/benchmark/results/translation_summary.csv`
- `APPT实验存档/FixCode0405/benchmark/results/experiment_tables.md`

第二版/较新结果：

- `APPT实验存档/第二版实验/0405/benchmark/results`
- `APPT实验存档/第二版实验/NewPrompt/DS/results`

其中 `NewPrompt/DS` 只有 `cpu_sycl` 和结果表，没有完整 baseline 数据目录；baseline 文件可按同名文件从 `APPT实验存档/第二版实验/0405/benchmark/data` 或 `APPT实验存档/FixCode0405/benchmark/data` 对应。

## 给当前 SYCLForge 的接入建议

如果要把旧实验 case 接进当前 SYCLForge：

1. seed 用 `cpu_sycl/gemm_M_K_N.cpp`，也就是 Our-SYCL。
2. 对照 baseline 用 `benchmark/data/cuda_baseline_gemm/gemm_M_K_N.cu` 的 OR-CUDA 数值。
3. 只对 S/K/N/M 组保留 DPCT 对比；A 组三个 WMMA case 不应期望 DPCT baseline。
4. A 组三个 case 适合作为 Tensor Core 后优化重点。
