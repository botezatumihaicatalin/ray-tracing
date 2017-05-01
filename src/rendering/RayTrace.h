#pragma once

#include "Sphere.h"

class RayTrace {

private:
  const Sphere* surface_;
  float tnear_;

public:

  __host__ __device__ 
  RayTrace(const Sphere* surface, float tnear) : 
    surface_(surface), tnear_(tnear) {}

  __host__ __device__ 
  const Sphere* surface() const { return surface_;  }
  
  __host__ __device__ 
  float tnear() const { return tnear_; }

  __host__ __device__ 
  bool has_trace() const { return surface_ != nullptr;  }
};




