#pragma once

#include <cmath>

#include "glm/vec3.hpp"
#include "glm/glm.hpp"

#include "Ray.h"
#include "Light.h"


class PhongIllumination {

private:
  const Ray& source_ray_;
  const SurfaceProperties& properties_;

public:

  __host__ __device__ 
  PhongIllumination(const Ray& source_ray, const SurfaceProperties& properties):
    source_ray_(source_ray), properties_(properties) {}

  __host__ __device__ 
  glm::vec3 calc_colour(const Light& light, const Ray& shadow_ray, bool in_shadow) const;
};

// Phong illumination: http://gta.math.unibuc.ro/stup/suport_curs_CGGC.pdf
inline glm::vec3 PhongIllumination::calc_colour(const Light& light, const Ray& shadow_ray, bool in_shadow) const {
  glm::vec3 surface_color(0.f);
  const Material& material = properties_.material();

  surface_color += light.ambient() * material.ambient(); // light_ambient * mat_ambient

  if (!in_shadow) {
    float light_dot_normal = fmaxf(0.f, glm::dot(shadow_ray.direction(), properties_.normal()));
        
    surface_color += light_dot_normal * light.diffuse() * material.diffuse();
    
    glm::vec3 h_direction = glm::normalize(shadow_ray.direction() - source_ray_.direction());
    float reflection_dot_normal = fmaxf(0.f, glm::dot(h_direction, properties_.normal()));

    surface_color += pow(reflection_dot_normal, material.shininess()) * light.specular() * material.specular();
  }

  const glm::vec3& atn_vec = light.attenuation();
  float light_dist = glm::distance(shadow_ray.origin(), light.position());
  float attenuation_factor = 1.f / (atn_vec[0] + atn_vec[1] * light_dist + atn_vec[2] * light_dist * light_dist);

  float spotlight_effect = 1.f;

  return spotlight_effect * attenuation_factor * surface_color;
}

/*
inline glm::vec3 Phong::calc_colour(const Light& light, bool in_shadow) const {
  Ray shadow_ray(properties_.point(), light.position() - properties_.point());
  return calc_colour(light, shadow_ray, in_shadow);
}
*/



