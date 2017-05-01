#pragma once

#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#endif

#include "glm/vec3.hpp"
#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/rotate_vector.hpp"

class Camera {

private:
  glm::vec3 eye_;
  glm::vec3 target_;
  glm::vec3 up_;

public:

  __host__ __device__ Camera(): Camera(glm::vec3(0, 0, 0), glm::vec3(0, 0, -1)) {};
  __host__ __device__ Camera(const glm::vec3& eye, const glm::vec3& target, const glm::vec3 up): eye_(eye), target_(glm::normalize(target)), up_(up) {}
  __host__ __device__ Camera(const glm::vec3& eye, const glm::vec3& target): Camera(eye, target, glm::vec3(0, 1, 0)) {}

  glm::mat4 viewMatrix() const;

  __host__ __device__ const glm::vec3& eye() const;
  __host__ __device__ const glm::vec3& target() const;
  __host__ __device__ const glm::vec3& up() const;

  void move_forward(const float& delta);
  void move_backward(const float& delta);
  void rotate(const float& angle);
};

inline glm::mat4 Camera::viewMatrix() const {
  return glm::lookAt(eye_, target_, up_);
}

inline const glm::vec3& Camera::eye() const {
  return eye_;
}

inline const glm::vec3& Camera::target() const {
  return target_;
}

inline const glm::vec3& Camera::up() const {
  return up_;
}

inline void Camera::move_forward(const float& delta) {
  eye_ += target_ * delta;
}

inline void Camera::move_backward(const float& delta) {
  eye_ -= target_ * delta;
}

inline void Camera::rotate(const float& angle) {
  target_ = glm::rotateY(target_, angle);
}






