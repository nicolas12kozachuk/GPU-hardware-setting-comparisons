#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <cstdint>

// function to initialize values
void init(int *a, const uint64_t N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

// kernel to double vector elements
__global__
void doubleElements(uint64_t *a, int N)
{

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < N + stride; i += stride)
  {
    a[i] *= 2;
  }
}

// function to check if vector elements were correctly doubled
bool checkElementsAreDoubled(int *a, const uint64_t N)
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
  const uint64_t N = 100000;
  uint64_t size = N * sizeof(int);

  // get # of GPUs
  int num_gpus;
  cudaGetDeviceCount(&num_gpus); 

  size_t threads_per_block = 1024; // set threads per block and # of blocks
  size_t number_of_blocks = 32;

  // get data per gpu
  int x = N;
  int y = num_gpus;
  const uint64_t chunk_size = x/y + (x % y != 0);

  uint64_t *data_gpu[num_gpus]; // One pointer for each GPU.

  for (int gpu = 0; gpu < num_gpus; gpu++) {

    cudaSetDevice(gpu); // set gpu to be used
    
    const uint64_t lower = chunk_size*gpu; 
    const uint64_t upper = min(lower+chunk_size, size);
    const uint64_t width = upper-lower; // get data width

    cudaMalloc(&data_gpu[gpu], sizeof(uint64_t)*width); // Allocate chunk of data for current GPU.
  }

  int *a;

  cudaMallocManaged(&a, size);

  init(a, N);


  for (int gpu = 0; gpu < num_gpus; gpu++) {
    cudaSetDevice(gpu); // set gpu to be used

    const uint64_t lower = chunk_size*gpu; // get data width
    const uint64_t upper = min(lower+chunk_size, N);
    const uint64_t width = upper-lower; 
    
    cudaMemcpy(data_gpu[gpu], a+lower, // copy data from host to device
           sizeof(uint64_t)*width, cudaMemcpyHostToDevice);
  }


  for (int gpu = 0; gpu < num_gpus; gpu++) {
    cudaSetDevice(gpu); // get data per gpu

    const uint64_t lower = chunk_size*gpu; // get data width 
    const uint64_t upper = min(lower+chunk_size, N);
    const uint64_t width = upper-lower;

    // Pass chunk of data for current GPU to work on.
    doubleElements<<<number_of_blocks, threads_per_block>>>(data_gpu[gpu], width); 
    
    cudaError_t err11 = cudaGetLastError(); // return the error from above.
    cudaError_t err22 = cudaDeviceSynchronize(); // ensure GPU threads all finish
  }


  clock_t end_time = clock();


  for (int gpu = 0; gpu < num_gpus; gpu++) {
    cudaSetDevice(gpu); // set gpu to be used

    const uint64_t lower = chunk_size*gpu; // get data width
    const uint64_t upper = min(lower+chunk_size, N);
    const uint64_t width = upper-lower;

    cudaMemcpy(a+lower, data_gpu[gpu], // copy data from device to host
           sizeof(uint64_t)*width, cudaMemcpyDeviceToHost); 
  }



  double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
  printf("Execution Time: %f seconds\n", execution_time);


  bool areDoubled = checkElementsAreDoubled(a, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");
  
  // free memory
  cudaFree(a);
  cudaFree(data_gpu);
}
