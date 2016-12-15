#include <iostream>
#include <cstdlib>
#include <cstdio>

using namespace std;

__global__ void euler1 (float2 *pos, float2* vel, float2 *acc, float dt, float box) {

	int i=threadIdx.x+blockDim.x*blockIdx.x;

	//Moves a particle using Euler

	pos[i].x += vel[i].x * dt;
	pos[i].y += vel[i].y * dt;

	vel[i].x += acc[i].x * dt;
	vel[i].y += acc[i].y * dt;

	/*
	if (pos[i].x > box) {
		pos[i].x -= box;
	}
	else if (pos[i].x < 0.0f) {
		pos[i].x += box;
	}

	if (pos[i].y > box) {
		pos[i].y -= box;
	}
	else if (pos[i].y < 0.0f) {
		pos[i].y += box;
	}
	*/
	//set acceleration to zero

	acc[i] = (float2){0.0f,0.0f};

}

__global__ void euler2 (float2 *pos, float2* vel, float2 *acc, float dt, float box) {

	int i=threadIdx.x+blockDim.x*blockIdx.x;
	int locali = threadIdx.x;

	//Copies to shared memory
	
	__shared__ float2 poslocal[192];
	__shared__ float2 acclocal[192];
	__shared__ float2 vellocal[192];

	poslocal[locali] = pos[i];
	vellocal[locali] = vel[i];
	acclocal[locali] = acc[i];

	poslocal[locali].x += vellocal[locali].x * dt;
	poslocal[locali].y += vellocal[locali].y * dt;

	vellocal[locali].x += acclocal[locali].x * dt;
	vellocal[locali].y += acclocal[locali].y * dt;

	/*
	if (pos[i].x > box) {
		pos[i].x -= box;
	}
	else if (pos[i].x < 0.0f) {
		pos[i].x += box;
	}

	if (pos[i].y > box) {
		pos[i].y -= box;
	}
	else if (pos[i].y < 0.0f) {
		pos[i].y += box;
	}
	*/
	
	//Copies new values to global memory
	pos[i] = poslocal[locali];
	vel[i] = vellocal[locali];
	acc[i] = (float2){0.0f,0.0f};

}

__global__ void euler3 (float2 *pos, float2* vel, float2 *acc, float dt, float box) {
 
        int i=threadIdx.x+blockDim.x*blockIdx.x;
 
        //Copia para a memória compartilhada
        float2 poslocal = pos[i], vellocal = vel[i], acclocal = acc[i];
 
        poslocal.x += vellocal.x * dt;
        poslocal.y += vellocal.y * dt;
 
        vellocal.x += acclocal.x * dt;
        vellocal.y += acclocal.y * dt;

	/*
	if (pos[i].x > box) {
		pos[i].x -= box;
	}
	else if (pos[i].x < 0.0f) {
		pos[i].x += box;
	}

	if (pos[i].y > box) {
		pos[i].y -= box;
	}
	else if (pos[i].y < 0.0f) {
		pos[i].y += box;
	}
	*/
		       
        //Copia os novos valores para a memória global
        pos[i] = poslocal;
        vel[i] = vellocal;
        acc[i] = (float2){0.0f,0.0f};
 
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
	float time1, time2, time3;

	float dt = 0.001f;	
	int nTests = 99;
	float box = 40.0f;
	int N = 10752;
	int buffer_size = N*sizeof(float2);

	float2 pos[N];
	float2 vel[N];
	float2 acc[N];

	for (int i=0; i<N; i++){

		pos[i].x = (float)rand()/RAND_MAX;
		pos[i].y = (float)rand()/RAND_MAX;

		vel[i].x = (float)rand()/RAND_MAX;
		vel[i].y = (float)rand()/RAND_MAX;

		acc[i].x = (float)rand()/RAND_MAX;
		acc[i].y = (float)rand()/RAND_MAX;

	}
	
	//Cria os ponteiros CUDA	
	float2 *pos_d=0, *vel_d, *acc_d=0;
	
	//Aloca os ponteiros no device
	cudaMalloc((void**)&pos_d,buffer_size);
	cudaMalloc((void**)&vel_d,buffer_size);
	cudaMalloc((void**)&acc_d,buffer_size);
	
	//Copia os vetores para o device		
	cudaMemcpy( pos_d, pos, buffer_size, cudaMemcpyHostToDevice );
	cudaMemcpy( vel_d, vel, buffer_size, cudaMemcpyHostToDevice );
	cudaMemcpy( acc_d, acc, buffer_size, cudaMemcpyHostToDevice );

	//Cria os eventos
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Marca o tempo inicial
	cudaEventRecord(start, 0);
	checkCUDAError("Start1");

	//Roda o kernel1 nTests vezes	
	for (int i=0; i<nTests; i++){
		euler1 <<<56, 192>>> (pos_d, vel_d, acc_d, dt, box);
		checkCUDAError("Euler1");
	}

	//Marca o evento de parada
	cudaEventRecord( stop, 0 );
	checkCUDAError("Stop1");
	//Para tudo até que o evento de parada seja marcado
	cudaEventSynchronize( stop );

	//Calcula a diferença de tempo entre start e stop
	cudaEventElapsedTime( &time1, start, stop );
	checkCUDAError("Time1");

	//Marca o tempo inicial
	cudaEventRecord(start, 0);
	checkCUDAError("Start2");	

	//Roda o kernel2 nTests vezes	
	for (int i=0; i<nTests; i++){
		euler2 <<<56, 192>>> (pos_d, vel_d, acc_d, dt, box);
		checkCUDAError("Euler2");
	}

	//Marca o evento de parada
	cudaEventRecord( stop, 0 );
	checkCUDAError("Stop2");
	//Para tudo até que o evento de parada seja marcado
	cudaEventSynchronize( stop );

	//Calcula a diferença de tempo entre start e stop
	cudaEventElapsedTime( &time2, start, stop );
	checkCUDAError("Time2");

	//Marca o tempo inicial
	cudaEventRecord(start, 0);
	checkCUDAError("Start3");

	//Roda o kernel3 nTests vezes	
	for (int i=0; i<nTests; i++){
		euler3 <<<56, 192>>> (pos_d, vel_d, acc_d, dt, box);
		checkCUDAError("Euler3");
	}

	//Marca o evento de parada
	cudaEventRecord( stop, 0 );
	checkCUDAError("Stop3");
	//Para tudo até que o evento de parada seja marcado
	cudaEventSynchronize( stop );

	//Calcula a diferença de tempo entre start e stop
	cudaEventElapsedTime( &time3, start, stop );
	checkCUDAError("Time3");

	//Mostra os tempos e quanto melhorou
	cout << "Time without optimization:         " << time1/nTests << "ms" << endl;
	cout << "Time with shared mem optimization: " << time2/nTests << "ms" << endl;
	cout << "Time with register optimization:   " << time3/nTests << "ms" << endl;
	cout << "Improvement of shared mem: " << (time1-time2)/time1*100 << "%" << endl;
	cout << "Improvement of register mem: " << (time1-time3)/time1*100 << "%" << endl;


	cudaEventDestroy( start );
	cudaEventDestroy( stop );
}
