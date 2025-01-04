#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <cstdint>
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
  const uint64_t N = 100000;
  size_t size = N * sizeof(int);
  int *data_cpu;
  int *data_gpu;

  // allocate memory for cpu and gpu data
  cudaMallocHost(&data_cpu, size);
  cudaMalloc    (&data_gpu, size);

  size_t threads_per_block = 1024; // set threads per block and # of blocks
  size_t number_of_blocks = 32;

  // Set the number of streams to not evenly divide num_entries.
  const uint64_t num_streams = 2;

  cudaStream_t streams[num_streams]; // create stream array and streams
  for (uint64_t stream = 0; stream < num_streams; stream++)
      cudaStreamCreate(&streams[stream]);
  
  // get amount of data per stream
  int x = N;
  int y = num_streams;
  const uint64_t chunk_size = x/y + (x % y != 0);

  init(data_cpu, N); // initiallize data

  for (uint64_t stream = 0; stream < num_streams; stream++) {
    // get data width
    const uint64_t lower = chunk_size*stream;
    const uint64_t upper = min(lower+chunk_size, N);
    const uint64_t width = upper-lower;

     // asynchronously copy data from host to device
    cudaMemcpyAsync(data_gpu+lower, data_cpu+lower, 
           sizeof(int)*width, cudaMemcpyHostToDevice, 
           streams[stream]);

    doubleElements<<<number_of_blocks, threads_per_block, 0, streams[stream]>>>(data_gpu+lower, N);
  
    cudaError_t err11 = cudaGetLastError(); // return the error from above.
    cudaError_t err22 = cudaDeviceSynchronize(); // ensure GPU threads all finish

    // asynchronously copy data from device to host
    cudaMemcpyAsync(data_cpu+lower, data_gpu+lower, 
           sizeof(int)*width, cudaMemcpyDeviceToHost, 
           streams[stream]);
  }

  // Destroy streams.
  for (uint64_t stream = 0; stream < num_streams; stream++)
      cudaStreamDestroy(streams[stream]);


  clock_t end_time = clock();

  double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
  printf("Execution Time: %f seconds\n", execution_time);


  bool areDoubled = checkElementsAreDoubled(data_cpu, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

  // free memory
  cudaFree(data_gpu);
  cudaFree(data_cpu);
}
