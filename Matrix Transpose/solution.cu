#include <cuda_runtime.h>

#define TILE_DIM 16
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float sdata[TILE_DIM][TILE_DIM+1];
    int ix = blockIdx.x * TILE_DIM + threadIdx.x;
    int iy = blockIdx.y * TILE_DIM + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (ix < cols && iy < rows) {
        sdata[ty][tx] = input[iy * cols + ix];
    } else {
        sdata[ty][tx] = 0.0f;
    }
    __syncthreads();

    ix = blockIdx.y * TILE_DIM + threadIdx.x;
    iy = blockIdx.x * TILE_DIM + threadIdx.y;
    if (ix < rows && iy < cols) {
        output[iy * rows + ix] = sdata[tx][ty];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
