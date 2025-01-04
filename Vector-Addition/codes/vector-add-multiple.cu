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

// function to initialize values
void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

// kernel to add vector elemnts
__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{
  int indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;

  for (int i = indexWithinTheGrid; i < N; i += gridStride)
  {
    result[i] = a[i] + b[i];
  }
}

// function to check elements are added correctly
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

  // get number of GPUS
  int num_gpus;
  cudaGetDeviceCount(&num_gpus);

  float *aa[num_gpus];
  float *bb[num_gpus];
  float *cc[num_gpus];

  uint64_t size_r = N;

  int x = N;
  int y = num_gpus;
  const uint64_t chunk_size = x/y + (x % y != 0);

  for (int gpu = 0; gpu < num_gpus; gpu++) {

    cudaSetDevice(gpu);

    const uint64_t lower = chunk_size*gpu;
    const uint64_t upper = min(lower+chunk_size, size_r);
    const uint64_t width = upper-lower;

    cudaMalloc(&aa[gpu], sizeof(uint64_t)*width); // Allocate chunk of data for current GPU.
    cudaMalloc(&bb[gpu], sizeof(uint64_t)*width); 
    cudaMalloc(&cc[gpu], sizeof(uint64_t)*width); 
  }

  // initialize value
  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  for (int gpu = 0; gpu < num_gpus; gpu++) {

    cudaSetDevice(gpu);

    const uint64_t lower = chunk_size*gpu;
    const uint64_t upper = min(lower+chunk_size, size_r);
    const uint64_t width = upper-lower;

    //copy memory from host to device
    cudaMemcpy(aa[gpu], a+lower, 
           sizeof(float)*width, cudaMemcpyHostToDevice); // ...or cudaMemcpyDeviceToHost
    cudaMemcpy(bb[gpu], b+lower, 
           sizeof(float)*width, cudaMemcpyHostToDevice);
    cudaMemcpy(cc[gpu], c+lower, 
           sizeof(float)*width, cudaMemcpyHostToDevice);
  }
  
  size_t threads_per_block = 256;

  size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

  for (int gpu = 0; gpu < num_gpus; gpu++) {

    cudaSetDevice(gpu);

    const uint64_t lower = chunk_size*gpu;
    const uint64_t upper = min(lower+chunk_size, size_r);
    const uint64_t width = upper-lower;

    int width_int = width;
    
    addVectorsInto<<<number_of_blocks, threads_per_block>>>(cc[gpu],aa[gpu],bb[gpu],width_int);
  
    checkCuda( cudaGetLastError() );
    checkCuda( cudaDeviceSynchronize() );

  }

  for (int gpu = 0; gpu < num_gpus; gpu++) {

    cudaSetDevice(gpu);

    const uint64_t lower = chunk_size*gpu;
    const uint64_t upper = min(lower+chunk_size, size_r);
    const uint64_t width = upper-lower;

    // copy memory from device to host
    cudaMemcpy(a+lower, aa[gpu],
           sizeof(float)*width, cudaMemcpyDeviceToHost); // ...or cudaMemcpyDeviceToHost
    cudaMemcpy(b+lower, bb[gpu],
           sizeof(float)*width, cudaMemcpyDeviceToHost);
    cudaMemcpy(c+lower, cc[gpu],
           sizeof(float)*width, cudaMemcpyDeviceToHost);
}

  // check elements are added correctly
  checkElementsAre(7, c, N);

  // free memory
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaFree(aa);
  cudaFree(bb);
  cudaFree(cc);
}
