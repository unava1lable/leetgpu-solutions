#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < width * height * 4) {
        image[idx+0] = 255 - image[idx+0];
        image[idx+1] = 255 - image[idx+1];
        image[idx+2] = 255 - image[idx+2];
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
