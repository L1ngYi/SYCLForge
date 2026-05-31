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

Write best kernels to a separate directory:

```bash
python -m syclforge.main_sycl cpu_sycl --first-n 20 --write-back-dir syclforge_best
```
