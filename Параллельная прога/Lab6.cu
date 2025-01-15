#include <iostream>
#include <cuda_runtime.h>

#define N 16

__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int n) {
    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0.0;

    for (int i = 0; i < n / 16; ++i) {
        tileA[threadIdx.y][threadIdx.x] = A[row * n + (i * 16 + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(i * 16 + threadIdx.y) * n + col];
        __syncthreads();

        for (int j = 0; j < 16; ++j) {
            value += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = value;
    }
}

void initMatrix(float *matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        //matrix[i] = static_cast<float>(i);
        if (i % (N - 1) == 0) { 
            matrix[i] = 1; 
        } else {
            matrix[i] = 0;
        }
    }
}

void printMatrix(const float *matrix, int n) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << matrix[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    size_t size = N * N * sizeof(float);
    
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    initMatrix(h_A, N);
    initMatrix(h_B, N);

    printMatrix(h_A, N);
    std::cout << "\n\n";

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 grid(N / 16, N / 16);

    matrixMultiplyKernel<<<grid, threads>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printMatrix(h_C, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}





// #include <stdio.h>
// #define BLOCK_SIZE 16 // submatrix size
// #define N 1024 // matrix size is N*N
// __global__ void matMult ( float * a, float * b, int n, float * c )
// {
// int bx = blockIdx.x; // block index
// int by = blockIdx.y;
// int tx = threadIdx.x; // thread index
// int ty = threadIdx.y;
// float sum = 0.0f; // computed subelement
// int ia = n * BLOCK_SIZE * by + n * ty; // a [i][0]
// int ib = BLOCK_SIZE * bx + tx;
// // Multiply the two matrices together;
// for ( int k = 0; k < n; k++ )
// sum += a [ia + k] * b [ib + k*n];
// // Write the block sub-matrix to global memory;
// // each thread writes one element
// int ic = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;46
// c [ic + n * ty + tx] = sum;
// }
// int main ( int argc, char * argv [] )
// {
// int numBytes = N * N * sizeof ( float );
// // выделение памяти на хосте
// float * a = new float [N*N];
// float * b = new float [N*N];
// float * c = new float [N*N];
// for ( int i = 0; i < N; i++ )
// for ( int j = 0; j < N; j++ )
// {
// int k = N*i + j;
// a [k] = 0.0f;
// b [k] = 1.0f;
// }
// // выделение памяти на девайсе
// float * adev = NULL;
// float * bdev = NULL;
// float * cdev = NULL;
// cudaMalloc ( (void**)&adev, numBytes );
// cudaMalloc ( (void**)&bdev, numBytes );
// cudaMalloc ( (void**)&cdev, numBytes );
// // Установка конфигурации запуска ядра47
// dim3 threads ( BLOCK_SIZE, BLOCK_SIZE );
// dim3 blocks ( N / threads.x, N / threads.y);
// // Создание обработчика событий CUDA
// cudaEvent_t start, stop;
// float gpuTime = 0.0f;
// cudaEventCreate ( &start );
// cudaEventCreate ( &stop );
// // асинхронно выдаваем работу на GPU (все в поток 0)
// cudaEventRecord ( start, 0 );
// cudaMemcpy ( adev, a, numBytes, cudaMemcpyHostToDevice );
// cudaMemcpy ( bdev, b, numBytes, cudaMemcpyHostToDevice );
// matMult<<<blocks, threads>>> ( adev, bdev, N, cdev );
// cudaMemcpy ( c, cdev, numBytes, cudaMemcpyDeviceToHost );
// cudaEventRecord ( stop, 0 );
// cudaEventSynchronize ( stop );
// cudaEventElapsedTime ( &gpuTime, start, stop );
// // Печатаем время работы на GPU и CPU
// printf(«time spent executing by the GPU: %.2f millseconds\n», gpuTime );
// // Освобождение ресурсов
// cudaEventDestroy ( start );
// cudaEventDestroy ( stop );
// cudaFree ( adev );48
// cudaFree ( bdev );
// cudaFree ( cdev );
// delete a;
// delete b;
// delete c;
// return 0;
// }