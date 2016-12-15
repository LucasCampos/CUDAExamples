#include <iostream>
#include <cstdlib>

__global__ void addVec (const int* A, const int * B, int* C) {

	int i=blockIdx.x*blockDim.x+threadIdx.x;

	C[i] = A[i]+B[i];

}

int main() {

	int *A,*B,*C; //Creates pointers of int type. We will use these for host arrays
	int *A_d,*B_d,*C_d; //Creates pointers of int type. We will use these for device arrays
	int N = 8; //Chooses array size as 8

	int buffer_size = sizeof(int)*N; //array size, in bytes.

	//Allocates arrays on host	
	A = new int[N];
	B = new int[N];
	C = new int[N];

	//Allocates buffer_size bytes on GPU's global memory for each array
	cudaMalloc((void**) &A_d, buffer_size);
	cudaMalloc((void**) &B_d, buffer_size);
	cudaMalloc((void**) &C_d, buffer_size);
	
	for (int i=0; i<N; i++) {//Initialize A and B
		A[i] = N - i;
		B[i] = i;
	}
	
	//Copies A and B to A_d and B_d. We won't copy C because it won't be read.
	cudaMemcpy( A_d, A, buffer_size, cudaMemcpyHostToDevice ); 
	cudaMemcpy( B_d, B, buffer_size, cudaMemcpyHostToDevice ); 

	int Blocks = 1;
	int ThreadsPerBlock = 8;

	addVec<<<Blocks,ThreadsPerBlock>>> (A_d, B_d, C_d); //Launch the kernel.

	cudaMemcpy( C, C_d, buffer_size, cudaMemcpyDeviceToHost ); //Copies from C_d to C.
	
	for (int i=0; i<N; i++) { //Checks if all the operations were done right
		std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl;
	}
	
	//Clean arrays on host
	free(A);
	free(B);
	free(C);

	//Clean arrays on device
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);

	return 0;
}
