#include <sycl/sycl.hpp>
using namespace sycl;

void gemm(float *A, float *B, float *C, int m, int k, int n, sycl::queue &q) {
    constexpr int TILE_SIZE = 16;
    q.submit([&](handler &h) {
        local_accessor<float, 2> tileA(range<2>(TILE_SIZE, TILE_SIZE), h);
        local_accessor<float, 2> tileB(range<2>(TILE_SIZE, TILE_SIZE), h);
        
        h.parallel_for(nd_range<2>(range<2>(m, n), range<2>(TILE_SIZE, TILE_SIZE)), 
                      [=](nd_item<2> item) {
            int row = item.get_global_id(0);
            int col = item.get_global_id(1);
            int localRow = item.get_local_id(0);
            int localCol = item.get_local_id(1);
            
            float sum = 0.0f;
            #pragma operation(write(output[sum]))
            
            for (int tile = 0; tile < k; tile += TILE_SIZE) {
                #pragma operation(memory(input[A], output[tileA]))
                if (row < m && (tile + localCol) < k) {
                    tileA[localRow][localCol] = A[row * k + tile + localCol];
                } else {
                    tileA[localRow][localCol] = 0.0f;
                }
                
                #pragma operation(memory(input[B], output[tileB]))
                if ((tile + localRow) < k && col < n) {
                    tileB[localRow][localCol] = B[(tile + localRow) * n + col];
                } else {
                    tileB[localRow][localCol] = 0.0f;
                }
                
                item.barrier(access::fence_space::local_space);
                
                #pragma operation(matmul(input[tileA, tileB], output[sum]))
                for (int inner = 0; inner < TILE_SIZE && (tile + inner) < k; ++inner) {
                    sum += tileA[localRow][inner] * tileB[inner][localCol];
                }
                
                item.barrier(access::fence_space::local_space);
            }
            
            #pragma operation(memory(input[sum], output[C]))
            if (row < m && col < n) {
                C[row * n + col] = sum;
            }
        });
    });
    q.wait();
}