#include <stdio.h>
#include <time.h>

// function to initialize values
void init(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

// kernel to double vector elements
__global__
void doubleElements(int *a, int N)
{

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < N + stride; i += stride)
  {
    a[i] *= 2;
  }
}

// function to check if vector elements were correctly doubled
bool checkElementsAreDoubled(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    if (a[i] != i*2) return false;
  }
  return true;
}

int main()
{
  clock_t start_time = clock();

  // define sizes and cpu and gpu data variables
  int N = 100000;
  int *a;
  size_t size = N * sizeof(int);

  cudaMallocManaged(&a, size); // allocated memory

  init(a,N); // initialize values

  size_t threads_per_block = 1024;  // set threads per block and # of blocks
  size_t number_of_blocks = 32;

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N); // call kernel
  
  cudaError_t err11 = cudaGetLastError(); // returns the error from above.
  cudaError_t err22 = cudaDeviceSynchronize(); // ensure GPU threads all finish

  clock_t end_time = clock();

  float execution_time = ((float)(end_time - start_time)) / CLOCKS_PER_SEC;
  printf("Execution Time: %f seconds\n", execution_time);
  
  if (err11 != cudaSuccess)
  {
    printf("Error: %s\n", cudaGetErrorString(err11));
  }
  
  if (err22 != cudaSuccess)
  {
    printf("Error: %s\n", cudaGetErrorString(err22));
  }


  bool areDoubled = checkElementsAreDoubled(a, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

  cudaFree(a);
}
