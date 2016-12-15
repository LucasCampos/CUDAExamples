#include <iostream>
#include <cstdlib>

__global__ void setVec (int* array) {

	int i=blockIdx.x*blockDim.x+threadIdx.x;

	array[i] = 42;

}

int main() {

	int* array; //Creates a pointer of int. This will be used on host
	int* array_d; //Creates a pointer of int. This will be used on device
	int N = 8; //Sets the array size as 8

	int buffer_size = sizeof(int)*N; //array size, in bytes
	array = new int[N]; //Allocates array on host 
	cudaMalloc((void**) &array_d, buffer_size); //Allocates buffer_size bytes on GPU's global memory
	
	for (int i=0; i<N; i++) {
		array[i] = N - i;//Initializes array
	}

	std::cout << "Array before kernel" << std::endl;
	for (int i=0; i<N; i++) { //Writes array on screen
		std::cout << "array[" << i << "]: " << array[i] << std::endl;
	}

	cudaMemcpy( array_d, array, buffer_size, cudaMemcpyHostToDevice ); //This will copy buffer_size bytes to device

	int Block = 1; //How many blocks we will use
	int Threads = N; //How many threads per block we will use
	setVec<<<Block,Threads>>> (array_d); //Launches the kernel, with Blocks blocks and Threads threads per block.
	
	cudaMemcpy( array, array_d, buffer_size, cudaMemcpyDeviceToHost ); //Copies array back to the host	
	
	std::cout << std::endl << "Array after kernel" << std::endl;
	for (int i=0; i<N; i++) { //Writes array on screen
		std::cout << "array[" << i << "]: " << array[i] << std::endl;
	}

	cudaFree(array_d);
	delete[] array;
	
	return 0;
}
