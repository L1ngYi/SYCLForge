# SYCLForge

CudaForge-style iterative optimization for the fixed-shape SYCL GEMM kernels in
`cpu_sycl/`. The loop is:

1. Use the existing SYCL file as the seed candidate.
2. Compile it for the SYCL CUDA backend.
3. Validate `C = A @ B` against NumPy.
4. Measure device-side kernel time with SYCL event profiling.
5. Optionally run Nsight Compute and feed the metrics to an LLM judge.
6. Ask the LLM to repair or optimize the kernel and repeat.

## Environment

For NVIDIA A100, load a SYCL compiler that supports the CUDA backend, then set
the usual variables if auto-detection is not enough:

```bash
export ONEAPI_DEVICE_SELECTOR=cuda:gpu
export SYCL_TARGETS=nvptx64-nvidia-cuda
export SYCL_CUDA_ARCH=sm_80
# Optional:
# export SYCL_COMPILER=icpx
# export SYCL_EXTRA_FLAGS="..."
```

`ncu` is optional but recommended. Use `--no-ncu` for latency-only feedback.
Keep the GPU compute mode at `Default` for normal profiling runs. If the system
must run in `Exclusive_Process` and automatic NCU profiling cannot acquire the
device, add `--isolated-benchmark`.

Add `--tensor-core` to enable Tensor Core oriented prompts. SYCLForge first
compiles a TF32 `joint_matrix` canary; only a successful probe allows the LLM to
use `sycl::ext::oneapi::experimental::matrix` APIs.

中文快速上手见 [`使用说明.md`](使用说明.md)。

## Examples

Evaluate one seed kernel without LLM calls:

```bash
python -m syclforge.main_sycl cpu_sycl --case-stem gemm_1024_1024_1024 --round 1 --no-llm --no-ncu
```

Run iterative optimization for one case:

```bash
python -m syclforge.main_sycl cpu_sycl \
  --case-stem gemm_1024_1024_1024 \
  --gpu A100 \
  --server_type openai \
  --model_name o3 \
  --round 6
```

Run a Tensor Core oriented experiment:

```bash
python -m syclforge.main_sycl cpu_sycl \
  --case-stem gemm_128_128_128 \
  --server_type deepseek \
  --model_name deepseek-v4-pro \
  --round 2 \
  --max_tokens 16000 \
  --tensor-core
```

Write best kernels to a separate directory:

```bash
python -m syclforge.main_sycl cpu_sycl --first-n 20 --write-back-dir syclforge_best
```
