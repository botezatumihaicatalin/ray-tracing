#pragma once

#include "Ray.h"
#include "Light.h"
#include "../glm/glm.hpp"
#include <algorithm>
#include <cmath>

class Phong {

private:
  const Ray& source_ray_;
  const glm::vec3 normal_;

public:

  __host__ __device__ Phong(const Ray& source_ray, glm::vec3 normal) : source_ray_(source_ray), normal_(normal) {};

  __host__ __device__ glm::vec3 calc_colour(const Light& light, const Ray& shadow_ray, bool in_shadow) const;
};


inline glm::vec3 Phong::calc_colour(const Light& light, const Ray& shadow_ray, bool in_shadow) const {
  glm::vec3 surface_color(0.f);

  float light_dot_normal = fmaxf(0.f, glm::dot(shadow_ray.direction(), normal_));
  glm::vec3 h_direction = glm::normalize(shadow_ray.direction() - source_ray_.direction());
  float reflection_dot_normal = fmaxf(0.f, glm::dot(h_direction, normal_));

  surface_color += light.ambient() * glm::vec3(0.2, 0.2, 0.22); // light_ambient * mat_ambient
  surface_color += (1 - in_shadow) * light_dot_normal * light.diffuse() * glm::vec3(0.6, 0.7, 0.8);
  surface_color += (1 - in_shadow) * pow(reflection_dot_normal, 10) * light.specular() * glm::vec3(1, 1, 1);

  return surface_color;
}



