%%writefile Lab5.cu
#include <iostream>
#include <cuda_runtime.h>

#define N 16

__global__ void transposeKernel(float *in, float *out, int n) {
    __shared__ float tile[16][16];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n) {
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    }
    __syncthreads();

    x = blockIdx.y * blockDim.x + threadIdx.x;
    y = blockIdx.x * blockDim.y + threadIdx.y;

    if (x < n && y < n) {
        out[y * n + x] = tile[threadIdx.x][threadIdx.y];
    }
}

void initMatrix(float *matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = static_cast<float>(i);
    }
}

void printMatrix(const float *matrix, int n) {
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            std::cout << matrix[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    float *h_in, *h_out;
    float *d_in, *d_out;

    size_t size = N * N * sizeof(float);
    
    h_in = (float*)malloc(size);
    h_out = (float*)malloc(size);

    initMatrix(h_in, N);
    printMatrix(h_in, N);
    std::cout << std::endl;

    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 grid(N / 16, N / 16);


    // Создание событий для замера времени
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    transposeKernel<<<grid, threads>>>(d_in, d_out, N);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);


    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    printMatrix(h_out, N);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    cout << "\nTime: " << elapsedTime << " ms" << endl;

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}







