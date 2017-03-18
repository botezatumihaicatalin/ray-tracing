#pragma once

#include "glm/glm.hpp"
#include "glm/vec3.hpp"
#include "Camera.h"
#include "Ray.h"
#include "Sphere.h"

#define M_PI 3.141592653589793

class Scene {

private:
  Camera camera_;
  size_t width_;
  size_t height_;
  float fov_;
  float ratio_;

public:

  Scene(size_t width, size_t height);

  Ray makeRay(size_t x, size_t y) const;
  glm::vec3 trace(const Ray& ray) const;
  glm::vec3* render() const;

  size_t width() const;
  size_t height() const;
};

inline Scene::Scene(size_t width, size_t height) {
  width_ = width;
  height_ = height;
  fov_ = 60;
  camera_ = Camera(glm::vec3(0, 0, 0), glm::vec3(0, 0, 1));
  ratio_ = width / float(height);
}

inline Ray Scene::makeRay(size_t x, size_t y) const {
  float nx = (x / float(width_)) - 0.5f;
  float ny = (y / float(height_)) - 0.5f;

  glm::vec3 camera_right = glm::cross(camera_.target, camera_.up);
  glm::vec3 camera_up = glm::cross(camera_right, camera_.target);

  glm::vec3 image_point = nx * camera_right + ny * camera_up + camera_.eye + camera_.target;
  glm::vec3 ray_direction = glm::normalize(image_point - camera_.eye);
  return Ray(camera_.eye, ray_direction);
}

inline glm::vec3 Scene::trace(const Ray& ray) const {
  Sphere one(glm::vec3(0, 0, 5), 0.5f);
  if (one.intersects(ray)) {
    return glm::vec3(68, 115, 19);
  }
  return glm::vec3(0, 0, 0);
}

inline glm::vec3* Scene::render() const {
  glm::vec3* buffer = new glm::vec3[width_ * height_];

  unsigned i = 0;
  for (unsigned y = 0; y < height_; ++y) {
    for (unsigned x = 0; x < width_; ++x, i++) {
      buffer[i] = trace(makeRay(x, y));
    }
  }
  return buffer;
}

inline size_t Scene::width() const {
  return width_;
}

inline size_t Scene::height() const {
  return height_;
}




