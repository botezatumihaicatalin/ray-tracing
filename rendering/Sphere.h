#pragma once

#include "../glm/vec3.hpp"
#include "../glm/glm.hpp"
#include "Ray.h"

class Sphere {

private:
  glm::vec3 center_;
  float radius_;
  // TODO: add transparency, reflection, etc.

public:
  __host__ __device__ Sphere(const glm::vec3& center, float radius) : center_(center), radius_(radius) {}
  __host__ __device__ Sphere() : Sphere(glm::vec3(0), 0.f) {};

  __host__ __device__ const glm::vec3& center() const;
  __host__ __device__ const float& radius() const;

  __host__ __device__ float intersects(const Ray& ray) const;
};

inline const glm::vec3& Sphere::center() const {
  return center_;
}

inline const float& Sphere::radius() const {
  return radius_;
}

inline float Sphere::intersects(const Ray& ray) const {
  glm::vec3 l = center_ - ray.origin();
  float tca = glm::dot(l, ray.direction());
  if (tca < 0) {
    return -INFINITY;
  }
  float d2 = glm::dot(l, l) - tca * tca;
  float radius2 = radius_ * radius_;
  if (d2 > radius2) {
    return -INFINITY;
  }
  float thc = sqrt(radius2 - d2);
  float t0 = tca - thc;
  float t1 = tca + thc;
  return (t0 < 0 || t0 > t1) ? t1 : t0;
}

