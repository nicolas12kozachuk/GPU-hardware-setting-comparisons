#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <cstdint>

// function to check for CUDA errors
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

// function to initialize vector values
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

// function to check elements are correctly added
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
  const uint64_t N = 2<<20;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  
  float *data_cpu;
  float *data_gpu;

  // allocate memory
  cudaMallocHost(&data_cpu, size);
  cudaMalloc    (&data_gpu, size);

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);

  // initialize values
  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, data_cpu, N);

  // Set the number of streams to not evenly divide num_entries.
  const uint64_t num_streams = 2;

  cudaStream_t streams[num_streams];
  for (uint64_t stream = 0; stream < num_streams; stream++)
      cudaStreamCreate(&streams[stream]);
  
  size_t threads_per_block = 256;
  size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

  // get data size per gpu chunk
  int x = N;
  int y = num_streams;
  const uint64_t chunk_size = x/y + (x % y != 0);

  for (uint64_t stream = 0; stream < num_streams; stream++) {

    const uint64_t lower = chunk_size*stream;
    // For tail stream `lower+chunk_size` could be out of range, so here we guard against that.
    const uint64_t upper = min(lower+chunk_size, N);
    // Since the tail stream width may not be `chunk_size`,
    // we need to calculate a separate `width` value.
    const uint64_t width = upper-lower;

    // copy memory from host to device
    cudaMemcpyAsync(data_gpu+lower, data_cpu+lower, 
           sizeof(float)*width, cudaMemcpyHostToDevice, 
           streams[stream]);

    addVectorsInto<<<number_of_blocks, threads_per_block, 0, streams[stream]>>>(data_gpu+lower,a,b,width);

    // check for CUDA errors
    checkCuda( cudaGetLastError() );
    checkCuda( cudaDeviceSynchronize() );

    // copy memory from device to host
    cudaMemcpyAsync(data_cpu+lower, data_gpu+lower, 
           sizeof(float)*width, cudaMemcpyDeviceToHost, 
           streams[stream]);
  }

  // Destroy streams.
  for (uint64_t stream = 0; stream < num_streams; stream++)
      cudaStreamDestroy(streams[stream]);
  
  // Check elements are correctly doubled
  checkElementsAre(7, data_cpu, N);

  // free memory
  cudaFree(a);
  cudaFree(b);
  cudaFree(data_cpu);
  cudaFree(data_gpu);

}
