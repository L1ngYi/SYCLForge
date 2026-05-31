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
                int inner_base = tile_idx * TILE_SIZE;
                
                int load_row = row;
                int load_col = inner_base + local_col;
                if (load_row < m && load_col < k) {
                    #pragma operation(memory(input[A], output[tileA]))
                    tileA[local_row][local_col] = A[load_row * k + load_col];
                } else {
                    #pragma operation(write(output[tileA]))
                    tileA[local_row][local_col] = 0.0f;
                }
                
                int load_row_b = inner_base + local_row;
                int load_col_b = col;
                if (load_row_b < k && load_col_b < n) {
                    #pragma operation(memory(input[B], output[tileB]))
                    tileB[local_row][local_col] = B[load_row_b * n + load_col_b];
                } else {
                    #pragma operation(write(output[tileB]))
                    tileB[local_row][local_col] = 0.0f;
                }
                
                item.barrier(access::fence_space::local_space);
                
                #pragma operation(matmul(input[tileA, tileB], output[sum]))
                for (int inner = 0; inner < TILE_SIZE; ++inner) {
                    int global_inner = inner_base + inner;
                    if (global_inner < k) {
                        sum += tileA[local_row][inner] * tileB[inner][local_col];
                    }
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