# CUDAExamples
A set of simple CUDA examples

Any of these files can be run using

``` sh
nvcc filelame.cu -run
```

These examples were written with GTX465 in mind. It might be a good idea do test a few different choices of `nBlocks` and `nThreads`, depending on your GPU.

## SimpleVector.cu

Allocates a vector on device and sets it to zero. This file is intended to teach how allocate a vector.

## SetVector.cu

Allocates a vector on device and sets it to a given value, using a simple kernel.

## AddVector.cu

Allocates three vectos on the device and sum the first two.

## SerialVsCUDA.cu

Compare the timings of two implementations of the Euler integration scheme, one in Serial and another in CUDA.

## TestTimeEuler.cu/TestTimeYukawa.cu

Measures the timing for trivial optimazions, using local and shared memory naively.
