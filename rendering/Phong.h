#pragma once

#include <cmath>

#include "../glm/glm.hpp"

#include "Ray.h"
#include "Light.h"


class Phong {

private:
  const Ray& source_ray_;
  const glm::vec3 normal_;

public:

  __host__ __device__ Phong(const Ray& source_ray, glm::vec3 normal) : source_ray_(source_ray), normal_(normal) {};

  __host__ __device__ glm::vec3 calc_colour(const Light& light, const Ray& shadow_ray, bool in_shadow) const;
};

// Phong illumination: http://gta.math.unibuc.ro/stup/suport_curs_CGGC.pdf
inline glm::vec3 Phong::calc_colour(const Light& light, const Ray& shadow_ray, bool in_shadow) const {
  glm::vec3 surface_color(0.f);

  float light_dot_normal = fmaxf(0.f, glm::dot(shadow_ray.direction(), normal_));
  glm::vec3 h_direction = glm::normalize(shadow_ray.direction() - source_ray_.direction());
  float reflection_dot_normal = fmaxf(0.f, glm::dot(h_direction, normal_));

  surface_color += light.ambient() * glm::vec3(0.294117f, 0, 0.5098039f); // light_ambient * mat_ambient
  surface_color += (1 - in_shadow) * light_dot_normal * light.diffuse() * glm::vec3(0.6, 0.7, 0.8);
  surface_color += (1 - in_shadow) * pow(reflection_dot_normal, 10) * light.specular() * glm::vec3(1, 1, 1);

  const glm::vec3& atn_vec = light.attenuation();
  float light_dist = glm::distance(shadow_ray.origin(), light.position());
  float attenuation_factor = 1.f / (atn_vec[0] + atn_vec[1] * light_dist + atn_vec[2] * light_dist * light_dist);

  float spotlight_effect = 1.f;

  return spotlight_effect * attenuation_factor * surface_color;
}



