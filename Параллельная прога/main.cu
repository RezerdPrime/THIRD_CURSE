#include <iostream>
#include <cuda_runtime.h>

using namespace std;

const int N = 2;
const int M = 3;


__global__ void MatAdd(float A[N][M], float B[N][M], float *result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N * M) {
        int i = idx / M;
        int j = idx % M;
        atomicAdd(result, A[i][j] * B[i][j]);
    }
}


int main() {
    float (*A)[M] = new float[N][M];
    float (*B)[M] = new float[N][M];
    float h_result = 0.0f;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            A[i][j] = static_cast<float>(i * j); 
            B[i][j] = static_cast<float>(i * j);
        }
    }

    float (*dev_A)[M], (*dev_B)[M];
    float *dev_result;

    cudaMalloc((void**)&dev_A, N * M * sizeof(float));
    cudaMalloc((void**)&dev_B, N * M * sizeof(float));
    cudaMalloc((void**)&dev_result, sizeof(float));

    cudaMemcpy(dev_A, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N * M + threadsPerBlock - 1) / threadsPerBlock;
    MatAdd<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_B, dev_result);

    cudaMemcpy(&h_result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; ++i){
        for(int j = 0; j < M; ++j){
            cout << A[i][j] << ' ';
        }
        cout << '\n';
    } cout << '\n';

    for(int i = 0; i < N; ++i){
        for(int j = 0; j < M; ++j){
            cout << B[i][j] << ' ';
        }
        cout << '\n';
    } cout << '\n';

    cout << "Скалярное произведение: " << h_result << endl;

    delete[] A;
    delete[] B;
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_result);

    return 0;
}
