#include <sycl/sycl.hpp>
using namespace sycl;

void gemm(float *A, float *B, float *C, int m, int k, int n, sycl::queue &q) {
    constexpr int TILE_SIZE = 16;
    
    auto global_range = nd_range<2>(range<2>(m, n), range<2>(TILE_SIZE, TILE_SIZE));
    
    q.submit([&](handler &h) {
        local_accessor<float, 2> tileA(range<2>(TILE_SIZE, TILE_SIZE), h);
        local_accessor<float, 2> tileB(range<2>(TILE_SIZE, TILE_SIZE), h);
        
        h.parallel_for(global_range, [=](nd_item<2> item) {
            int row = item.get_global_id(0);
            int col = item.get_global_id(1);
            int local_row = item.get_local_id(0);
            int local_col = item.get_local_id(1);
            
            float sum = 0.0f;
            #pragma operation(write(output[sum]))
            
            int num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;
            
            for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
                int inner_start = tile_idx * TILE_SIZE;
                
                // Load tile from A
                int a_row = row;
                int a_col = inner_start + local_col;
                #pragma operation(memory(input[A], output[tileA]))
                if (a_row < m && a_col < k) {
                    tileA[local_row][local_col] = A[a_row * k + a_col];
                } else {
                    tileA[local_row][local_col] = 0.0f;
                }
                
                // Load tile from B
                int b_row = inner_start + local_row;
                int b_col = col;
                #pragma operation(memory(input[B], output[tileB]))
                if (b_row < k && b_col < n) {
                    tileB[local_row][local_col] = B[b_row * n + b_col];
                } else {
                    tileB[local_row][local_col] = 0.0f;
                }
                
                item.barrier(access::fence_space::local_space);
                
                // Compute partial sum using local memory
                #pragma operation(matmul(input[tileA, tileB], output[sum]))
                for (int inner = 0; inner < TILE_SIZE; ++inner) {
                    sum += tileA[local_row][inner] * tileB[inner][local_col];
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