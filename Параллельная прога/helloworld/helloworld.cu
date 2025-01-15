#include<iostream>
#include<cuda_runtime.h>
using namespace std;

#define N 5

// int A[N] = {1, 1, 1, 1, 1};
// int B[N] = {1, 2, 3, 4, 5};
// int C[N];

__global__ void VecAdd(float* A, float* B, float* C)
{
  
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}


int main(){

    float A[N] = {1, 1, 1, 1, 1};
    float B[N] = {1, 2, 3, 4, 5};
     C[N];

    float *dev_A, *dev_B, *dev_C;
    cudaMalloc( (void**)&dev_A, sizeof(float)*N );
    cudaMalloc( (void**)&dev_B, sizeof(float)*N );
    cudaMalloc( (void**)&dev_C, sizeof(float)*N );
  
    cudaMemcpy( dev_A, A, sizeof(float)*N, cudaMemcpyHostToDevice );
    cudaMemcpy( dev_B, B, sizeof(float)*N, cudaMemcpyHostToDevice );

    VecAdd<<<1, N>>>(A, B, C);

    for (int i = 0; i < N; i++){
      cout << C[i] << endl;
    }

   cudaDeviceSynchronize();
  return 0;
}


// #include <mpi.h>
// #include <iostream>
// #include <string>
// using namespace std;

// #define SIZE 10

// int main(int argc, char *argv[]) {
//     MPI_Init(&argc, &argv);

//     int rank, size;
//     int arr[SIZE];
//     int local_sum = 0, global_sum = 0;
//     int local_max = 0, global_max = 0;

//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     if (rank == 0) {
//         cout << "Process 0 broadcasting data...\n";
//         for (int i = 0; i < SIZE; i++) {
//             arr[i] = i + 1;
//             //cout << arr[i] << " ";
//         }
//     }

//     MPI_Bcast(arr, SIZE, MPI_INT, 0, MPI_COMM_WORLD);

//     if (rank != 0){
//         for (int i = 0; i < SIZE; i++) {
//             arr[i] *= rank;

//             //cout << arr[i] << " ";
//         }

//         for(int i = 0; i < SIZE; i++) {
//             local_sum += arr[i];

//             if (arr[i] > local_max) {
//                 local_max = arr[i];
//             }
//         }
//     }

//     MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
//     MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

//     MPI_Barrier(MPI_COMM_WORLD);

//     if(rank == 0) {
//         cout << endl << "max: " << global_max << endl;
//         cout << "sum: " << global_sum << endl << endl;
//     }

//     MPI_Finalize();
// }


// int main(int argc, char *argv[]) {
//     MPI_Init(&argc, &argv);

//     string msg;
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     const int MSG_LEN = 20;
//     char message[MSG_LEN];

//     if (rank == 0) {
//         for (int i = 1; i < size; ++i) {
//             msg = "Hello from process 0 to ";
//             msg += to_string(i);
//             MPI_Send(msg.c_str(), MSG_LEN, MPI_CHAR, i, 0, MPI_COMM_WORLD);
//         }

//     } else {
//         MPI_Recv(message, MSG_LEN, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         cout << "Process " << rank << " received message: " << message << endl;
//     }

//     MPI_Finalize();
//     return 0;
// }

// #include <iostream>
// #include <mpi.h>


//int main(int argc, char** argv) {

    // // Инициализация MPI
    // MPI_Init(&argc, &argv);

    // // Получаем количество процессов
    // int treads_num;
    // MPI_Comm_size(MPI_COMM_WORLD, &treads_num);

    // // Получаем ранг (идентификатор) текущего процесса
    // int treads_rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &treads_rank);

    // // Каждый процесс выводит свое сообщение
    // std::cout << "Hello from process " << treads_rank << " out of " << treads_num << " processes" << std::endl;

    // // Завершение работы MPI
    // MPI_Finalize();

    // return 0;
//}

// mpiexec -n 4 helloworld.exe