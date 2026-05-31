#include <sycl/sycl.hpp>
using namespace sycl;

void gemm(float *A, float *B, float *C, int m, int k, int n, sycl::queue &q) {
    q.submit([&](handler &h) {
        h.parallel_for(range<2>(m, n), [=](item<2> item) {
            int row = item.get_id(0);
            int col = item.get_id(1);
            float sum = 0.0f;
            #pragma operation(write(output[sum]))
            for (int inner = 0; inner < k; ++inner) {
                #pragma operation(mul(input[A, B], output[sum]))
                sum += A[row * k + inner] * B[inner * n + col];
            }
            #pragma operation(memory(input[sum], output[C]))
            C[row * n + col] = sum;
        });
    });
    q.wait();
}