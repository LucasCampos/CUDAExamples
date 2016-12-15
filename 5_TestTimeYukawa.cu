#include <iostream>
#include <cstdlib>
#include <cstdio>

using namespace std;

void YukSerial(float2* pos, float2* acc, float k, int N){
	
	float2 del;
	float r2, r;
	float kr;
	float termo;

	for (int i=0; i<N; i++){
		for (int j=0; j<N; j++){
		
			if (i!=j){

				//Calculate distances
				del.x = pos[i].x - pos[j].x;
				del.y = pos[i].y - pos[j].y;

				r2 = (del.x * del.x + del.y *del.y);
				r = sqrt(r2);

				kr = k*r;

				termo = (1+kr)*exp(-kr)/(r2*r);

				acc[i].x += del.x*termo;
				acc[i].y += del.y*termo; 
			}
		}
	}
}

void YukSerialNewton(float2* pos, float2* acc, float k, int N){
	
	float2 del;
	float r2, r;
	float kr;
	float termo;

	for (int i=0; i<N-1; i++){
		for (int j=i+1; j<N; j++){
		
			//Calculate distances
			del.x = pos[i].x - pos[j].x;
			del.y = pos[i].y - pos[j].y;

			r2 = (del.x * del.x + del.y *del.y);
			r = sqrt(r2);

			kr = k*r;

			termo = (1+kr)*exp(-kr)/(r2*r);

			acc[i].x += del.x*termo;
			acc[i].y += del.y*termo; 
			
			acc[j].x -= del.x*termo;
			acc[j].y -= del.y*termo; 
		}
	}
}

__global__ void Yuk1 (float2 *pos, float2 *acc, float k,int N) {

	int i=threadIdx.x+blockDim.x*blockIdx.x;

	float2 del;
	float r2, r;
	float kr;
	float termo;

	for (int j=0; j<N; j++){
	
		if (i!=j){

			//Calculate distances
			del.x = pos[i].x - pos[j].x;
			del.y = pos[i].y - pos[j].y;

			r2 = (del.x * del.x + del.y *del.y);
			r = sqrt(r2);

			kr = k*r;

			termo = (1+kr)*exp(-kr)/(r2*r);

			acc[i].x += del.x*termo;
			acc[i].y += del.y*termo; 
		}
	}
}

__global__ void Yuk2 (float2 *pos, float2 *acc, float k, int N) {

	int i=threadIdx.x+blockDim.x*blockIdx.x;
	int locali = threadIdx.x;

	//Copies to shared memory
	
	__shared__ float2 poslocal[192];
	__shared__ float2 acclocal[192];

	poslocal[locali] = pos[i];
	acclocal[locali] = acc[i];

	float2 del;
	float r2, r;
	float kr;
	float termo;

	for (int j=0; j<N; j++){
	
	if (i!=j){

			//Calculates distances
			del.x = poslocal[locali].x - pos[j].x;
			del.y = poslocal[locali].y - pos[j].y;

			r2 = (del.x * del.x + del.y *del.y);
			r = sqrt(r2);

			kr = k*r;

			termo = (1+kr)*exp(-kr)/(r2*r);

			acclocal[locali].x += del.x*termo;
			acclocal[locali].y += del.y*termo; 
		}
	}
	acc[i] = acclocal [locali];
}

__global__ void Yuk3 (float2 *pos, float2 *acc, float k, int N){

	float2 del;
	float r2;
	float r;
	float kr;
	float termo;

	int i=blockIdx.x*blockDim.x+threadIdx.x;

	//Copies from global to local memory
	float2 poslocal = pos[i];
	float2 acclocal = acc[i];

	for (int j=0; j < N; j++) {
				
		if (i != j){ 

			//Calculates distances
			del.x = poslocal.x - pos[j].x;
			del.y = poslocal.y - pos[j].y;

			r2=del.x*del.x+del.y*del.y;
				
			r = sqrt(r2);

			kr = k*r;

			termo = (kr+1)*exp(-kr)/(r2*r);

			acclocal.x+= termo*del.x;
			acclocal.y+= termo*del.y;			
		}
	}
		
	//Copies acceleration back from register to global memory
	acc[i] = acclocal;
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

	//Creates CUDA events
	cudaEvent_t start, stop;
	float time1, time2, time3, timeSerial, timeSerialNewton;

	//Chooses parameters
	float k = 1.0f;	
	int nTests = 99;
	int nBlocks = 7;
	int threadsPerBlock = 192;
	int N = nBlocks*threadsPerBlock;
	float box = 40.0f, hbox = box/2.0f;
	int buffer_size = N*sizeof(float2);

	float2 pos[N];
	float2 acc[N];

	//Set values randomly
	for (int i=0; i<N; i++){

		pos[i].x = (float)rand()/RAND_MAX*box;
		pos[i].y = (float)rand()/RAND_MAX*box;

		acc[i].x = (float)rand()/RAND_MAX;
		acc[i].y = (float)rand()/RAND_MAX;

	}
	
	//Create device vectors
	float2 *pos_d=0, *acc_d=0;
	
	//Allocate device arrays
	cudaMalloc((void**)&pos_d,buffer_size);
	cudaMalloc((void**)&acc_d,buffer_size);
	
	//Copie arrays from host to device
	cudaMemcpy( pos_d, pos, buffer_size, cudaMemcpyHostToDevice );
	cudaMemcpy( acc_d, acc, buffer_size, cudaMemcpyHostToDevice );

	//Create events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Set the initial time
	cudaEventRecord(start, 0);
	checkCUDAError("StartSerial");

	//Runs kernel 1 nTests times
	for (int i=0; i<nTests; i++){
		YukSerial (pos, acc, k, N);
	}

	//Set stop time
	cudaEventRecord( stop, 0 );
	checkCUDAError("StopSerial");
	//Synchonizes everything
	cudaEventSynchronize( stop );

	//Calculates the elapsed time
	cudaEventElapsedTime( &timeSerial, start, stop );
	checkCUDAError("TimeSerial");
	
	//Set the initial time
	cudaEventRecord(start, 0);
	checkCUDAError("StartSerialNewton");

	//Runs kernel 1 nTests times
	for (int i=0; i<nTests; i++){
		YukSerialNewton (pos, acc, k, N);
	}

	//Set stop time
	cudaEventRecord( stop, 0 );
	checkCUDAError("StopSerialNewton");
	//Synchonizes everything
	cudaEventSynchronize( stop );

	//Calculates the elapsed time
	cudaEventElapsedTime( &timeSerialNewton, start, stop );
	checkCUDAError("TimeSerialNewton");

	//Set the initial time
	cudaEventRecord(start, 0);
	checkCUDAError("Start1");

	//Runs kernel 1 nTests times
	for (int i=0; i<nTests; i++){
		Yuk1 <<<nBlocks,threadsPerBlock>>> (pos_d, acc_d, k, N);
		checkCUDAError("Yuk1");
	}

	//Set stop time
	cudaEventRecord( stop, 0 );
	checkCUDAError("Stop1");
	//Synchonizes everything
	cudaEventSynchronize( stop );

	//Calculates the elapsed time
	cudaEventElapsedTime( &time1, start, stop );
	checkCUDAError("Time1");
	
	//Set the initial time
	cudaEventRecord(start, 0);
	checkCUDAError("Start2");

	//Runs kernel 1 nTests times
	for (int i=0; i<nTests; i++){
		Yuk2 <<<nBlocks, threadsPerBlock>>> (pos_d, acc_d, k, N);
		checkCUDAError("Yuk2");
	}

	//Set stop time
	cudaEventRecord( stop, 0 );
	checkCUDAError("Stop2");
	//Synchonizes everything
	cudaEventSynchronize( stop );

	//Calculates the elapsed time
	cudaEventElapsedTime( &time2, start, stop );
	checkCUDAError("Time2");
	
	//Set the initial time
	cudaEventRecord(start, 0);
	checkCUDAError("Start3");

	//Runs kernel 1 nTests times
	for (int i=0; i<nTests; i++){
		Yuk3 <<<nBlocks, threadsPerBlock>>> (pos_d, acc_d, k, N);
		checkCUDAError("Yuk3");
	}

	//Set stop time
	cudaEventRecord( stop, 0 );
	checkCUDAError("Stop3");
	//Synchonizes everything
	cudaEventSynchronize( stop );

	//Calculates the elapsed time
	cudaEventElapsedTime( &time3, start, stop );
	checkCUDAError("Time3");


	//Shows timings
	cout << "Time on serial                     " << timeSerial/nTests << "ms" << endl;
	cout << "Time on serial Newton              " << timeSerialNewton/nTests << "ms" << endl;
	cout << "Time on CUDA without optimization:         " << time1/nTests << "ms" << endl;
	cout << "Time on CUDA with shared mem optimization: " << time2/nTests << "ms" << endl;
	cout << "Time on CUDA with register optimization:   " << time3/nTests << "ms" << endl;
	cout << "\nSimple CUDA velocity increase:     " << timeSerial/time1 << "x"  << endl;
	cout << "Improvement of shared mem:         " << (time1-time2)/time1*100 << "%" << endl;
	cout << "Improvement of register mem:       " << (time1-time3)/time1*100 << "%" << endl;

	cudaEventDestroy( start );
	cudaEventDestroy( stop );
}
