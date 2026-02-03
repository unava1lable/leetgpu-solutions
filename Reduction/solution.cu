#include <cuda_runtime.h>

__global__ void reduce(const float *input, float *output, int N) {
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        sdata[threadIdx.x] = input[idx];
    } else {
        sdata[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(output, sdata[threadIdx.x]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    reduce<<<grid, block, block.x * sizeof(float), 0>>>(input, output, N);
    cudaDeviceSynchronize();
}
