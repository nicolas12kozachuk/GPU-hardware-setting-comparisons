#include <stdio.h>
#include <assert.h>

// function to check for CUDA errors
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

// function to initialize values
void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

// kernel to add vector values
__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{
  int indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;

  for (int i = indexWithinTheGrid; i < N; i += gridStride)
  {
    result[i] = a[i] + b[i];
  }
}

// function to check if elements were correctly added
void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main()
{
  // define sizes and variables
  const int N = 2<<20;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;
  
  // allocate memory
  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  // initialize vectors with values
  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);
  
  size_t threads_per_block = 256;
  size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
  
  // call kernel for execution
  addVectorsInto<<<number_of_blocks, threads_per_block>>>(c,a,b,N);
  
  checkCuda( cudaGetLastError() );
  checkCuda( cudaDeviceSynchronize() );

  // check if elements are correctly added
  checkElementsAre(7, c, N);
  
  // free memory
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
