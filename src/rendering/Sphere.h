#pragma once

#include "glm/vec3.hpp"
#include "glm/glm.hpp"

#include "Ray.h"
#include "Surface.h"

class Sphere {

private:
  glm::vec3 center_;
  float radius_;
  // TODO: add material.

public:
  __host__ __device__ 
  Sphere(const glm::vec3& center, float radius) : center_(center), radius_(radius) {}
  
  __host__ __device__ 
  Sphere() : Sphere(glm::vec3(0), 0.f) {}

  __host__ __device__ 
  const glm::vec3& center() const { return center_;  }
  
  __host__ __device__ 
  const float& radius() const { return radius_; }

  __host__ __device__ 
  float intersection(const Ray& ray) const;

  __host__ __device__
  SurfaceProperties properties(const glm::vec3& properties) const;
};

inline float Sphere::intersection(const Ray& ray) const {
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

inline SurfaceProperties Sphere::properties(const glm::vec3& point) const {
  return SurfaceProperties(point, point - center_);
}

