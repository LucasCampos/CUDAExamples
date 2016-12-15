#include <iostream>
#include <cstdlib>

int main() {

	int* vetor; //Declares a integer pointer
	int* vetor_d; //Declares a integer pointer
    int N = 10; //Declares and initializes N to 10

	int buffer_size = sizeof(int)*N; //Number of bytes in our array
	vetor = (int*) malloc (buffer_size); //Allocates the host vector
	cudaMalloc((void**) &vetor_d, buffer_size); //Allocates the device vector
	
	for (int i=0; i<N; i++) {
		vetor[i] = N - i;
	}

    //Copies from the host to the device
	cudaMemcpy( vetor_d, vetor, buffer_size, cudaMemcpyHostToDevice ); 
    //Sets the values of the device vector to zero
	cudaMemset( vetor_d, 0, buffer_size); 
    //Copies the vector back from the device to the host
	cudaMemcpy( vetor, vetor_d, buffer_size, cudaMemcpyDeviceToHost ); 
	
	for (int i=0; i<N; i++) {
		std::cout << "vetor em " << i << ": " << vetor[i] << std::endl;
	}
	
	return 0;
}
	
