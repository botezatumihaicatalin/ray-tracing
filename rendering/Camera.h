#pragma once

#include "glm/vec3.hpp"
#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"

class Camera {

private:
  glm::vec3 eye_;
  glm::vec3 target_;
  glm::vec3 up_;

public:

  Camera(): Camera(glm::vec3(0, 0, 0), glm::vec3(0, 0, -1)) {};
  Camera(const glm::vec3& eye, const glm::vec3& target, const glm::vec3 up): eye_(eye), target_(target), up_(up) {}
  Camera(const glm::vec3& eye, const glm::vec3& target): Camera(eye, target, glm::vec3(0, 1, 0)) {}

  glm::mat4 viewMatrix() const;

  glm::vec3 eye() const;
  glm::vec3 target() const;
  glm::vec3 up() const;
};

inline glm::mat4 Camera::viewMatrix() const {
  return glm::lookAt(eye_, target_, up_);
}

inline glm::vec3 Camera::eye() const {
  return eye_;
}

inline glm::vec3 Camera::target() const {
  return target_;
}

inline glm::vec3 Camera::up() const {
  return up_;
}



