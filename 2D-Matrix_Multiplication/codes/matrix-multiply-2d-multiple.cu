#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <stdint.h>
#include <cstdint>
#include <iostream>
#include <cmath>


#define N  256

// function to check for CUDA errors
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

// kernel to perform 2D matrix multiplication
__global__ void matrixMulGPU( int * a, int * b, int * c, uint64_t width, int start )
{
  int indexWithinTheGridX = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStrideX = gridDim.x * blockDim.x;
 
  int indexWithinTheGridY = threadIdx.y + blockIdx.y * blockDim.y;
  int gridStrideY = gridDim.y * blockDim.y;


  int s = N;
  for (int i = indexWithinTheGridX; i < s; i += gridStrideX)
  {
    for (int j = indexWithinTheGridY; j < s; j += gridStrideY)
      {
        int val = 0;
        for ( int k = 0; k < N; ++k ){
            val += a[(i * s + k) + width*((start))] * b[(k * s + j) + width*((start))];
        }
        c[i*s + j] = (start*val)+val;
      }
  }
}

/*
 * This CPU function already works, and will run to create a solution matrix
 * against which to verify your work building out the matrixMulGPU kernel.
 */

void matrixMulCPU( int * a, int * b, int * c )
{
  int val = 0;

  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      val = 0;
      for ( int k = 0; k < N; ++k )
        val += a[row * N + k] * b[k * N + col];
      c[row * N + col] = val;
    }
}


int main()
{
  clock_t start_time = clock();

  int *aa, *bb, *cc_cpu, *cc_gpu; // Allocate a solution matrix for both the CPU and the GPU operations

  const uint64_t size_r = N * N * sizeof (int); // Number of bytes of an N x N matrix

  const uint64_t size = N*N;

  int num_gpus;
  cudaGetDeviceCount(&num_gpus);


  int x = size;
  int y = num_gpus;
  const uint64_t chunk_size = x/y + (x % y != 0);

  int *c_gpu[num_gpus]; // One pointer for each GPU.

  for (int gpu = 0; gpu < num_gpus; gpu++) {

    cudaSetDevice(gpu);

    const uint64_t lower = chunk_size*gpu;
    const uint64_t upper = min(lower+chunk_size, size);
    const uint64_t width = upper-lower;

    cudaMalloc(&c_gpu[gpu], sizeof(int)*width); // Allocate chunk of data for current GPU.

  }


  // Allocate memory
  cudaMallocManaged (&aa, size_r);
  cudaMallocManaged (&bb, size_r);
  cudaMallocManaged (&cc_cpu, size_r);
  cudaMallocManaged (&cc_gpu, size_r);


  // Initialize memory; create 2D matrices
  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      aa[row*N + col] = row;
      bb[row*N + col] = col+2;
      cc_cpu[row*N + col] = 0;
      cc_gpu[row*N + col] = 0;
    }
 
 
  for (int gpu = 0; gpu < num_gpus; gpu++) {
    cudaSetDevice(gpu);

    dim3 threads_per_block(16,16,1);
    dim3 number_of_blocks(16,16,1);

    const uint64_t lower = chunk_size*gpu;
    const uint64_t upper = min(lower+chunk_size, size);
    const uint64_t width = upper-lower;

    const uint64_t width_sq = sqrt(width);

    matrixMulGPU <<< number_of_blocks, threads_per_block >>> ( aa, bb, c_gpu[gpu], width, gpu);
   
    checkCuda( cudaGetLastError() );
    checkCuda( cudaDeviceSynchronize() );
  }

  clock_t end_time = clock();

  float execution_time = ((float)(end_time - start_time)) / CLOCKS_PER_SEC;
  printf("Execution Time: %f seconds\n", execution_time);

  // Call the CPU version to check our work
  matrixMulCPU( aa, bb, cc_cpu );

  for (int gpu = 0; gpu < num_gpus; gpu++) {

    cudaSetDevice(gpu);

    const uint64_t lower = chunk_size*gpu;
    const uint64_t upper = min(lower+chunk_size, size);
    const uint64_t width = upper-lower;

    // Note use of `cudaMemcpy` and not `cudaMemcpyAsync` since we are not
    // presently using non-default streams.
    cudaMemcpy(cc_gpu+lower, c_gpu[gpu],
           sizeof(int)*width, cudaMemcpyDeviceToHost); // ...or cudaMemcpyDeviceToHost
  }
 
 
  // Compare the two answers to make sure they are equal
  bool error = false;
  for( int row = 0; row < N && !error; ++row )
    for( int col = 0; col < N && !error; ++col ){
      if (cc_cpu[row * N + col] != cc_gpu[row * N + col])
      {
        bool test = cc_cpu[row * N + col] != cc_gpu[row * N + col];
        printf("%d %d %d \n",test,cc_cpu[row * N + col], cc_gpu[row * N + col]);
        printf("FOUND ERROR at c[%d][%d]\n", row, col);
        error = true;
        break;
      }
    }
  if (!error)
    printf("Success!\n");

  
  // Free all our allocated memory
  cudaFree( cc_cpu );
  cudaFree(aa); cudaFree(bb);
  cudaFree( cc_gpu );
  cudaFree( c_gpu );
  
}






