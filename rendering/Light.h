#pragma once

#include "../glm/vec3.hpp"

class Light {

private:
  glm::vec3 position_;
  glm::vec3 ambient_;
  glm::vec3 diffuse_;
  glm::vec3 specular_;
  glm::vec3 attenuation_;

public:

  __host__ __device__ explicit Light(const glm::vec3& position, const glm::vec3& ambient, const glm::vec3& diffuse, const glm::vec3& specular, const glm::vec3& attenuation) :
    position_(position), ambient_(ambient), diffuse_(diffuse), specular_(specular), attenuation_(attenuation) {};
  __host__ __device__ explicit Light(const glm::vec3& position, const glm::vec3& ambient, const glm::vec3& diffuse, const glm::vec3& specular) : Light(position, ambient, diffuse, specular, glm::vec3(1, 0, 0)) {};
  __host__ __device__ explicit Light() : Light(glm::vec3(0), glm::vec3(0), glm::vec3(0), glm::vec3(0)) {};

  __host__ __device__ const glm::vec3& position() const;
  __host__ __device__ const glm::vec3& ambient() const;
  __host__ __device__ const glm::vec3& diffuse() const;
  __host__ __device__ const glm::vec3& specular() const;
  __host__ __device__ const glm::vec3& attenuation() const;
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

inline const glm::vec3 & Light::attenuation() const {
  return attenuation_;
}
