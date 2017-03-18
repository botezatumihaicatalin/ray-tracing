#pragma once

#include "glm/vec3.hpp"
#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"

class Camera {

public:
  glm::vec3 eye;
  glm::vec3 target;
  glm::vec3 up;

  Camera(): Camera(glm::vec3(0, 0, 0), glm::vec3(0, 0, -1)) {};
  Camera(const glm::vec3& eye, const glm::vec3& target, const glm::vec3 up): eye(eye), target(target), up(up) {}
  Camera(const glm::vec3& eye, const glm::vec3& target): Camera(eye, target, glm::vec3(0, 1, 0)) {}

  glm::mat4 viewMatrix() const;
};

inline glm::mat4 Camera::viewMatrix() const {
  return glm::lookAt(eye, target, up);
}
