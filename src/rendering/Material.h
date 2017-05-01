#pragma once

#include "glm/vec3.hpp"
#include <host_defines.h>

class Material {

private:
  glm::vec3 ambient_;
  glm::vec3 diffuse_;
  glm::vec3 specular_;
  float shininess_;

public:

  __host__ __device__
  explicit Material(const glm::vec3& ambient, const glm::vec3& diffuse, const glm::vec3& specular, float shininess) :
    ambient_(ambient), diffuse_(diffuse), specular_(specular), shininess_(shininess) {}

  __host__ __device__
  explicit Material(const glm::vec3& ambient, const glm::vec3& diffuse, const glm::vec3& specular) 
    : Material(ambient, diffuse, specular, 10) {}

  __host__ __device__
  explicit Material() : Material(glm::vec3(0), glm::vec3(0), glm::vec3(0), 0) {}

  __host__ __device__
  const glm::vec3& ambient() const { return ambient_; }

  __host__ __device__
  const glm::vec3& diffuse() const { return diffuse_; }

  __host__ __device__
  const glm::vec3& specular() const { return specular_; }

  __host__ __device__
  const float& shininess() const { return shininess_; }
};
