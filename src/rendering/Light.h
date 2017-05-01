#pragma once

#include "glm/vec3.hpp"

class Light {

private:
  glm::vec3 position_;
  glm::vec3 ambient_;
  glm::vec3 diffuse_;
  glm::vec3 specular_;
  glm::vec3 attenuation_;

public:

  __host__ __device__ 
  explicit Light(const glm::vec3& position, const glm::vec3& ambient, const glm::vec3& diffuse, const glm::vec3& specular, const glm::vec3& attenuation) :
    position_(position), ambient_(ambient), diffuse_(diffuse), specular_(specular), attenuation_(attenuation) {};

  __host__ __device__ 
  explicit Light(const glm::vec3& position, const glm::vec3& ambient, const glm::vec3& diffuse, const glm::vec3& specular) 
   : Light(position, ambient, diffuse, specular, glm::vec3(1, 0, 0)) {};
  
  __host__ __device__ 
  explicit Light() : Light(glm::vec3(0), glm::vec3(0), glm::vec3(0), glm::vec3(0)) {};

  __host__ __device__ 
  const glm::vec3& position() const { return position_; }

  __host__ __device__ 
  const glm::vec3& ambient() const { return ambient_; }

  __host__ __device__ 
  const glm::vec3& diffuse() const { return diffuse_; }

  __host__ __device__ 
  const glm::vec3& specular() const { return specular_; }

  __host__ __device__ 
  const glm::vec3& attenuation() const { return attenuation_; }
};
