
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
static void runTest(void);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
	runTest();

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.

	checkCudaErrors(cudaDeviceReset());
 
    return 0;
}

__global__ void testKernel(float* g_idata, float* g_odata)
{
	extern __shared__ float sdata[]; // the 3rd parameter of the host kernel func. 
	const unsigned int bid = blockIdx.x;
	const unsigned int tid_in_block = threadIdx.x;
	const unsigned int tid_in_grid = blockDim.x * blockIdx.x + threadIdx.x;

#if 1	
	sdata[tid_in_block] = g_idata[tid_in_grid];
	__syncthreads();

	//sdata[tid_in_block] *= (float)bid;
	//sdata[tid_in_block] *= (float)tid_in_block;
	sdata[tid_in_block] *= (float)tid_in_grid;

	__syncthreads();
	g_odata[tid_in_grid] = sdata[tid_in_block];
#else
	__syncthreads();
	g_idata[tid_in_grid] *= (float)bid;
#endif
}


static void runTest()
{
	// set GPU device
	checkCudaErrors(cudaSetDevice(0));

	unsigned int num_blocks = 4;
	unsigned int num_threads = 4;
	unsigned int mem_size = sizeof(float) * num_threads * num_blocks;

	// allocate host arrays
	float* h_idata = (float*)malloc(mem_size);
	assert(h_idata != NULL);
	float* h_odata = (float*)malloc(mem_size);
	assert(h_odata != NULL);

	// alllocate GPU arrays
	float* d_idata = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_idata, mem_size));
	float* d_odata = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_odata, mem_size));

	// initialize the host array (input)
	for (unsigned int i = 0; i < num_threads * num_blocks; i++)
		h_idata[i] = 1.0f;
	cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice);


	// The model will be like this:
	// 
	//  Grid#0
	//	Blk#0	Blk#1	Blk#2	Blk#3
	//  T0..T3  T4..T7  T8..T11 T12..T15
	//  
	// define the CUDA runtime parameters
	dim3 grid(num_blocks, 1, 1);
	dim3 threads(num_threads, 1, 1);

	// run the kernel
	testKernel <<<grid, threads, mem_size >>>(d_idata, d_odata);
	checkCudaErrors(cudaDeviceSynchronize());
	cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost);

	// output 
	for (unsigned int i = 0; i < num_blocks; i++) {
		for (unsigned int j = 0; j < num_threads; j++) {
			printf("%5.0f", h_odata[i * num_threads + j]);
		}
		printf("\n");
	}

	// release host arrays memory
	free(h_idata);
	free(h_odata);

	checkCudaErrors(cudaFree(d_idata));
	checkCudaErrors(cudaFree(d_odata));
	checkCudaErrors(cudaDeviceReset());
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
