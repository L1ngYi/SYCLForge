#include <sycl/sycl.hpp>
using namespace sycl;

void gemm(float *A, float *B, float *C, int m, int k, int n, queue &q) {
  constexpr int TILE_M = 16;
  constexpr int TILE_N = 16;
  constexpr int TILE_K = 16;
  range<2> global_size(
      ((m) + TILE_M - 1) / TILE_M * TILE_M,
      ((n) + TILE_N - 1) / TILE_N * TILE_N);
  range<2> local_size(TILE_M, TILE_N);

  q.submit([&](handler &h) {
    local_accessor<float, 2> A_tile(
        range<2>(TILE_M, TILE_K), h);
    local_accessor<float, 2> B_tile(
        range<2>(TILE_K, TILE_N), h);
    h.parallel_for(
        nd_range<2>(global_size, local_size),
        [=](nd_item<2> item) {
          int row = item.get_global_id(0);
          int col = item.get_global_id(1);
          int local_row = item.get_local_id(0);
          int local_col = item.get_local_id(1);
          float sum = 0.0f;

          for (int tile_k = 0; tile_k < k; tile_k += TILE_K) {
            int a_col = tile_k + local_col;
            int b_row = tile_k + local_row;
            A_tile[local_row][local_col] =
                (row < m && a_col < k)
                    ? A[row * k + a_col]
                    : (float)0;
            B_tile[local_row][local_col] =
                (b_row < k && col < n)
                    ? B[b_row * n + col]
                    : (float)0;
            item.barrier(access::fence_space::local_space);

#pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
          sum += static_cast<float>(A_tile[local_row][kk]) *
                 static_cast<float>(B_tile[kk][local_col]);
            }
            item.barrier(access::fence_space::local_space);
          }

          if (row < m && col < n) {
            C[row * n + col] = sum;
          }
        });
  });
}