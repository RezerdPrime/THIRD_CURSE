#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;


__global__ void dotProductKernel(const float* A, const float* B, float* C, int N, int M) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    if (row < N && col < M) {
        sum = A[row * M + col] * B[row * M + col];
    }

    __shared__ float sharedData[256];
    int threadIndex = threadIdx.x + threadIdx.y * blockDim.x;
    sharedData[threadIndex] = sum;

    __syncthreads();

    if (threadIndex == 0) {
        float blockSum = 0.0f;
        for (int i = 0; i < blockDim.x * blockDim.y; i++) {
            blockSum += sharedData[i];
        }
        atomicAdd(C, blockSum);
    }
}

float dotProduct(const float* A, const float* B, int N, int M) {
    float *d_A, *d_B, *d_C;
    float result;

    cudaMalloc((void**)&d_A, N * M * sizeof(float));
    cudaMalloc((void**)&d_B, N * M * sizeof(float));
    cudaMalloc((void**)&d_C, sizeof(float));

    cudaMemcpy(d_A, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * M * sizeof(float), cudaMemcpyHostToDevice);

    float initialValue = 0.0f;
    cudaMemcpy(d_C, &initialValue, sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((M + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    dotProductKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, M);
    
    cudaMemcpy(&result, d_C, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return result;
}

int main() {
    const int N = 3;
    const int M = 4;
    float A[N * M], B[N * M];

    for (int i = 0; i < N * M; ++i) {
        A[i] = i + 1;
        B[i] = i + 1;
    }

    for (int j = 0; j < M; j++){
        for (int i = 0; i < N; i++){
            cout << A[j*N + i] << " ";
        } cout << endl;
    }

    float result = dotProduct(A, B, N, M);
    cout << "Скалярное произведение: " << result << endl;

    return 0;
}