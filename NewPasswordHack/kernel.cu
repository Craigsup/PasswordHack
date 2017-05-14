#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <fstream>

#define POSSIBLE_CHARS 46
#define STARTING_POINT 47
#define THREADS 64//32
#define Z 32
#define XY 32//128
#define OFFSET 512

struct found {
	int yes;
	char password[8];
	unsigned long long attempts;
	unsigned long long lastId;
};
__device__ struct found *foundGlobal; // device (global)
unsigned long hackPassword(const int passwordSize, char actualPassword[8]);

__global__ void addKernel(char actualPassword[8])
{
	if (foundGlobal[0].yes != 1) {
		const unsigned long long idx = (blockIdx.x + blockIdx.y * gridDim.x
			+ gridDim.x * gridDim.y * blockIdx.z) * (blockDim.x * blockDim.y * blockDim.z)
			+ (threadIdx.z * (blockDim.x * blockDim.y))
			+ (threadIdx.y * blockDim.x) + threadIdx.x;

		unsigned long long nextId = foundGlobal[0].lastId;
		bool localGotcha = false;
		for (int i = 0; i < OFFSET && !localGotcha; i++) {
			char answer[8];
			unsigned long long newId = idx*OFFSET + i + nextId;

			int location = 0;
			unsigned long long idx2 = newId;
			
			while (idx2 >= POSSIBLE_CHARS) {
				location++;
				idx2 /= POSSIBLE_CHARS;
			}

			idx2 = newId;
			char pos2;
			while (location > 0) {
				pos2 = (idx2 % POSSIBLE_CHARS) + STARTING_POINT;
				answer[location] = pos2;
				location--;

				idx2 /= POSSIBLE_CHARS;
			}

			answer[0] = idx2 + STARTING_POINT;

			bool right = true;
			for (int z = 0; z < 8 && right; z++) {
				if (answer[z] != actualPassword[z]) {
					right = false;
				}
			}

			if (right) {
				localGotcha = true;
				foundGlobal[0].yes = 1;
				foundGlobal[0].attempts = newId;
				for (int i = 0; i < 8; i++) {
					foundGlobal[0].password[i] = answer[i];
				}

				return;
			}

		}

		if (idx == XY*XY*Z*THREADS - 1) {
			foundGlobal[0].lastId = nextId + (XY*XY*Z*THREADS*OFFSET);
		}
	}
}

int main()
{
	const int repetitions = 1;
	unsigned long times[repetitions];
	unsigned long attempts[repetitions];
	int passwordSize = 8;
	cudaError_t cudaStatus;
	char input[8] = { 0 };
	std::cout << "Please enter a password, no greater than 7 characters long, followed by the enter key.\n";
	std::cin.getline(input, 8, '\n');

	for (int i = 0; i < 8; i++) {
		if (input[i] == '\0') {
			passwordSize--;
		}
	}

    // Add vectors in parallel.
	for (int i = 0; i < repetitions; i++) {
		auto begin = std::chrono::high_resolution_clock::now();
		attempts[i] = hackPassword(passwordSize, input);
		auto end = std::chrono::high_resolution_clock::now();
		times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		cudaDeviceSynchronize();
	}

	std::ofstream out("cudaData.txt");
	out << "Time ns, attempts\n";
	for (int i = 0; i < repetitions; i++) {
		out << times[i] << " , " << attempts[i] << "\n";
	}
	out.close();


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	system("pause");
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
unsigned long hackPassword(int passwordSize, char actualPassword[8])
{
    cudaError_t cudaStatus;

	char* d_localactualPassword;
	cudaMalloc(&d_localactualPassword, 8 * sizeof(char));
	cudaMemcpy(d_localactualPassword, actualPassword, 8 * sizeof(char), cudaMemcpyHostToDevice);

	size_t size = 1 * sizeof(struct found);
	size_t sizep = 1 * sizeof(struct found*);
	struct found *localDeviceFound; // device (local)
	cudaMalloc(&localDeviceFound, size);
	cudaMemcpyToSymbol(foundGlobal, &localDeviceFound, sizep);
	struct found *newFound = (struct found*)malloc(size);

	dim3 grid(XY, XY, Z);

	while (newFound[0].yes != 1) {
		// Launch a kernel on the GPU with one thread for each element.
		addKernel << <grid, THREADS >> >(d_localactualPassword);
		cudaDeviceSynchronize();
		cudaMemcpy(newFound, localDeviceFound, size, cudaMemcpyDeviceToHost);	
		if (newFound[0].yes == 1) {
			printf("Found Password, it took: %llu attempts. Password = %s\n\n", newFound[0].attempts, newFound[0].password);
			return newFound[0].attempts;
		}
	}


    // Launch a kernel on the GPU with one thread for each element.

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

Error:
    
    return cudaStatus;
}
