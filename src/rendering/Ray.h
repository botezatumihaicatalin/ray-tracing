#pragma once

#include "glm/vec3.hpp"

class Ray {

private:
  glm::vec3 origin_;
  glm::vec3 direction_;

public:
  __host__ __device__ 
  Ray(const glm::vec3& origin, const glm::vec3& direction)
    : origin_(origin), direction_(glm::normalize(direction)) {}

  __host__ __device__ 
  const glm::vec3& direction() const { return direction_; }

  __host__ __device__ 
  const glm::vec3& origin() const { return origin_;  }
};



