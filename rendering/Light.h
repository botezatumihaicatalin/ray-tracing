#pragma once

#include "../glm/vec3.hpp"

class Light {

private:
  glm::vec3 position_;
  glm::vec3 ambient_;
  glm::vec3 diffuse_;
  glm::vec3 specular_;

public:

  __host__ __device__ explicit Light(const glm::vec3& position, const glm::vec3& ambient, const glm::vec3& diffuse, const glm::vec3& specular) :
    position_(position), ambient_(ambient), diffuse_(diffuse), specular_(specular) {};
  __host__ __device__ explicit Light() : Light(glm::vec3(0), glm::vec3(0), glm::vec3(0), glm::vec3(0)) {};

  __host__ __device__ const glm::vec3& position() const;
  __host__ __device__ const glm::vec3& ambient() const;
  __host__ __device__ const glm::vec3& diffuse() const;
  __host__ __device__ const glm::vec3& specular() const;
};

inline const glm::vec3& Light::position() const {
  return position_;
}

inline const glm::vec3& Light::ambient() const {
  return ambient_;
}

inline const glm::vec3& Light::diffuse() const {
  return diffuse_;
}

inline const glm::vec3& Light::specular() const {
  return specular_;
}
