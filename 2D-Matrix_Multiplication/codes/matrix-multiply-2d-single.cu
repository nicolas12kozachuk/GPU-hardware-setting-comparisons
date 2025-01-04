#include <stdio.h>
#include <assert.h>
#include <time.h>

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
__global__ void matrixMulGPU( int * a, int * b, int * c )
{

  int indexWithinTheGridX = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStrideX = gridDim.x * blockDim.x;
  
  int indexWithinTheGridY = threadIdx.y + blockIdx.y * blockDim.y;
  int gridStrideY = gridDim.y * blockDim.y;

  for (int i = indexWithinTheGridX; i < N; i += gridStrideX)
  {
    for (int j = indexWithinTheGridY; j < N; j += gridStrideY)
      {
        int val = 0;
        for ( int k = 0; k < N; ++k ){
            val += a[i * N + k] * b[k * N + j];
        }
        c[i * N + j] = val;
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

  int *a, *b, *c_cpu, *c_gpu; // Allocate a solution matrix for both the CPU and the GPU operations

  int size = N * N * sizeof (int); // Number of bytes of an N x N matrix

  // Allocate memory
  cudaMallocManaged (&a, size);
  cudaMallocManaged (&b, size);
  cudaMallocManaged (&c_cpu, size);
  cudaMallocManaged (&c_gpu, size);

  // Initialize memory; create 2D matrices
  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      a[row*N + col] = 5;
      b[row*N + col] = 5;
      c_cpu[row*N + col] = 0;
      c_gpu[row*N + col] = 0;
    }

  /*
   * Assign `threads_per_block` and `number_of_blocks` 2D values
   * that can be used in matrixMulGPU above.
   */

  dim3 threads_per_block(16,16,1); 
  dim3 number_of_blocks(16,16,1);
  
  matrixMulGPU <<< number_of_blocks, threads_per_block >>> ( a, b, c_gpu );

  checkCuda( cudaGetLastError() );
  checkCuda( cudaDeviceSynchronize() );

  clock_t end_time = clock();

  float execution_time = ((float)(end_time - start_time)) / CLOCKS_PER_SEC;
  printf("Execution Time: %f seconds\n", execution_time);

  
  // Call the CPU version to check our work
  matrixMulCPU( a, b, c_cpu );
  
  
  // Compare the two answers to make sure they are equal
  bool error = false;
  for( int row = 0; row < N && !error; ++row )
    for( int col = 0; col < N && !error; ++col )
      if (c_cpu[row * N + col] != c_gpu[row * N + col])
      {
        printf("FOUND ERROR at c[%d][%d]\n", row, col);
        error = true;
        break;
      }
  if (!error)
    printf("Success!\n");

  cudaFree( c_cpu );

  // Free all our allocated memory
  cudaFree(a); cudaFree(b);
  cudaFree( c_gpu );
}
