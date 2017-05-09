#pragma once

#include <cuda_runtime_api.h>
#include <cstdlib>

#define cudaCheck(code) { cudaAssert((code), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s, at %s %d\n", 
            cudaGetErrorString(code), file, line);
    exit(code);
  }
}
