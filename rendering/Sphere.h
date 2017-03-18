#pragma once

#include "glm/vec3.hpp"
#include "glm/glm.hpp"
#include "Ray.h"

class Sphere {

private:
  glm::vec3 center_;
  float radius_;
  // TODO: add transparency, reflection, etc.

public:
  Sphere(const glm::vec3& center, float radius) : center_(center), radius_(radius) {}

  bool intersects(const Ray& ray) const;
};

inline bool Sphere::intersects(const Ray& ray) const {
  glm::vec3 l = center_ - ray.origin();
  float tca = glm::dot(l, ray.direction());
  if (tca < 0) {
    return false;
  }
  float d2 = glm::dot(l, l) - tca * tca;
  return d2 <= radius_ * radius_;
}

