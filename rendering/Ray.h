#pragma once

#include "../glm/vec3.hpp"

class Ray {

private:
  glm::vec3 origin_;
  glm::vec3 direction_;

public:
  Ray(const glm::vec3& origin, const glm::vec3& direction)
    : origin_(origin), direction_(glm::normalize(direction)) {}

  const glm::vec3& direction() const;
  const glm::vec3& origin() const;
};

inline const glm::vec3& Ray::direction() const {
  return direction_;
}

inline const glm::vec3& Ray::origin() const {
  return origin_;
}



