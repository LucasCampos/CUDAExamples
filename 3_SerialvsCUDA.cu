#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cassert>

using namespace std;

__global__ void euler1 (float2 *pos, float2* vel, float2 *acc, float dt, float box) {

	int i=threadIdx.x+blockDim.x*blockIdx.x;

	//Moves a particle using Euler

	pos[i].x += vel[i].x * dt;
	pos[i].y += vel[i].y * dt;

	vel[i].x += acc[i].x * dt;
	vel[i].y += acc[i].y * dt;

}

void Euler_Serial ( float2 *pos, float2* vel, float2 *acc, float dt, float box, int nParticles)
{

	for (int i=0; i<nParticles; i++) {

        //Moves a particle using Euler
		pos[i].x += vel[i].x * dt;
		pos[i].y += vel[i].y * dt;

		vel[i].x += acc[i].x * dt;
		vel[i].y += acc[i].y * dt;
	
	}
}

inline void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
                                  cudaGetErrorString( err) );

        exit(EXIT_FAILURE);
    }                         
}

int main() {

	cudaEvent_t start, stop;
	float time1, time2;
	int nTests = 99999;
	float box = 40.0f;
	float dt =0.0001f;
	int nBlocks = 49;
	int nThreads = 192;
	int N=nBlocks*nThreads;
	int buffer_size = N*sizeof(float2);

	srand(time(NULL));

	float2 pos[N];
	float2 vel[N];
	float2 acc[N];	

	float2 posserial[N],
	       accserial[N],
	       velserial[N];

	for (int i=0; i<N; i++){

		pos[i].x = 5*(float)rand()/RAND_MAX;
		pos[i].y = 5*(float)rand()/RAND_MAX;

		vel[i].x = 5*(float)rand()/RAND_MAX;
		vel[i].y = 5*(float)rand()/RAND_MAX;

		acc[i].x = (float)rand()/RAND_MAX;
		acc[i].y = (float)rand()/RAND_MAX;

	}

	for (int i=0; i<N; i++){
		posserial[i] = pos[i];
		velserial[i] = vel[i];
		accserial[i] = acc[i];
	}
	
    //Declares the CUDA pointers
	float2 *pos_d=0, *vel_d, *acc_d=0;
	
    //Allocates the device pointers
	cudaMalloc((void**)&pos_d,buffer_size);
	cudaMalloc((void**)&vel_d,buffer_size);
	cudaMalloc((void**)&acc_d,buffer_size);
	
    //Copies the contents of the vector to the device
	cudaMemcpy( pos_d, pos, buffer_size, cudaMemcpyHostToDevice );
	cudaMemcpy( vel_d, vel, buffer_size, cudaMemcpyHostToDevice );
	cudaMemcpy( acc_d, acc, buffer_size, cudaMemcpyHostToDevice );

	//Creates the events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Sets the inital time
	cudaEventRecord(start, 0);
	checkCUDAError("Start1");

	for (int i=0; i<nTests; i++){
		euler1 <<<nBlocks, nThreads>>> (pos_d, vel_d, acc_d, dt, box);
		checkCUDAError("Euler1");
	}
	//Sets the final time
	cudaEventRecord( stop, 0 );
	checkCUDAError("Stop1");
	//Sync threads
	cudaEventSynchronize( stop );
    //Calculates the elapsed time
	cudaEventElapsedTime( &time1, start, stop );
	checkCUDAError("Time2");

	//Sets the inital time
	cudaEventRecord(start, 0);
	checkCUDAError("Start2");

	for (int i=0; i<nTests; i++){
		Euler_Serial (posserial, velserial, accserial, dt, box, N);
		checkCUDAError("Euler1");
	}
	//Sets the final time
	cudaEventRecord( stop, 0 );
	checkCUDAError("Stop2");
	//Sync threads
	cudaEventSynchronize( stop );
    //Calculates the elapsed time
	cudaEventElapsedTime( &time2, start, stop );
	checkCUDAError("Time2");

	cout <<"Tempo em CUDA: " << time1/nTests << "ms\n";
	cout <<"Tempo Serial: " << time2/nTests << "ms\n";
	cout <<"Aumento de " << time2/time1 << "x\n";

	return 0;
}
	
