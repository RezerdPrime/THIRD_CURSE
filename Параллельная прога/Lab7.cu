#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define EPS 0.0001f
#define N (16*1024)
#define BLOCK_SIZE 256

__global__ void integrateBodies(float3 * newPos, float3 * newVel, 
                                float3 * oldPos, float3 * oldVel, float dt) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float3 pos = oldPos[index];
    float3 f = make_float3(0.0f, 0.0f, 0.0f);

    for(int i = 0; i < N; i++){

        float3 pi = oldPos[i];
        float3 r;

        r.x = pi.x - pos.x;
        r.y = pi.y - pos.y;
        r.z = pi.z - pos.z;

        float invDist = 1.0f / sqrtf(r.x*r.x + r.y*r.y + r.z*r.z + EPS*EPS);
        float s = invDist * invDist * invDist;

        f.x += r.x * s;
        f.y += r.y * s;
        f.z += r.z * s;
    }

    float3 vel = oldVel[index];

    vel.x += f.x * dt;
    vel.y += f.y * dt;
    vel.z += f.z * dt;
    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;

    newPos[index] = pos;
    newVel[index] = vel;
}

void randomInit(float3 * a, int n){

    for(int i = 0; i < n; i++){
        a[i].x = rand() / (float) RAND_MAX - 0.5f;
        a[i].y = rand() / (float) RAND_MAX - 0.5f;
        a[i].z = rand() / (float) RAND_MAX - 0.5f;
    }
}

int main(int argc, char * argv[]){

    float3 * p = new float3[N];
    float3 * v = new float3[N];
    float3 * pDev[2] = {NULL, NULL};
    float3 * vDev[2] = {NULL, NULL};

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    int index = 0;

    randomInit(p, N);
    randomInit(v, N);

    std::cout << p.x << " " << p.y << " " << p.z << std::endl;
    std::cout << v.x << " " << v.y << " " << v.z << std::endl;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    cudaMalloc( (void**) &pDev[0], N * sizeof(float3) );
    cudaMalloc( (void**) &vDev[0], N * sizeof(float3) );
    cudaMalloc( (void**) &pDev[1], N * sizeof(float3) );
    cudaMalloc( (void**) &vDev[1], N * sizeof(float3) );

    cudaMemcpy( pDev[0], p, N * sizeof(float3), cudaMemcpyHostToDevice );
    cudaMemcpy( vDev[0], v, N * sizeof(float3), cudaMemcpyHostToDevice );

    for (int i=0; i < 2; i++, index ^= 1) {
        integrateBodies<<<dim3(N/BLOCK_SIZE), dim3(BLOCK_SIZE)>>>( pDev[index^1],
            vDev[index^1], pDev[index], vDev[index], 0.01f);
    }

    cudaMemcpy( p, pDev[index^1], N * sizeof(float3), cudaMemcpyDeviceToHost );
    cudaMemcpy( v, vDev[index^1], N * sizeof(float3), cudaMemcpyDeviceToHost );

    std::cout << pDev[0].x << " " << pDev[0].y << " " << pDev[0].z << std::endl;
    std::cout << vDev[0].x << " " << vDev[0].y << " " << vDev[0].z << std::endl;
    std::cout << std::endl;

    std::cout << pDev[1].x << " " << pDev[1].y << " " << pDev[1].z << std::endl;
    std::cout << vDev[1].x << " " << vDev[1].y << " " << vDev[1].z << std::endl;
    std::cout << std::endl;
    

    cudeFree(pDev[0]);
    cudeFree(vDev[0]);
    cudeFree(pDev[1]);
    cudeFree(vDev[1]);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    printf("Time: %.2f\n", gpuTime);

    delete p;
    delete v;

    return 0;
}