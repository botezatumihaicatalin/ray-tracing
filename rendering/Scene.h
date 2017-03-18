#pragma once

#include "glm/glm.hpp"
#include "glm/vec3.hpp"
#include "Camera.h"
#include "Ray.h"
#include "Sphere.h"
#include <vector>

class Scene {

private:
  Camera camera_;
  size_t width_;
  size_t height_;
  float fov_;
  float ratio_;
  std::vector<Sphere> spheres_;

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

  spheres_.push_back(Sphere(glm::vec3(0, 0, 5), 0.5f));
  spheres_.push_back(Sphere(glm::vec3(1, 1, 4), 0.7f));
}

inline Ray Scene::makeRay(size_t x, size_t y) const {
  float nx = (x / float(width_)) - 0.5f;
  float ny = (y / float(height_)) - 0.5f;

  glm::vec3 camera_eye = camera_.eye();
  glm::vec3 camera_target = camera_.target();
  glm::vec3 camera_up = camera_.up();

  glm::vec3 camera_right = glm::cross(camera_target, camera_up);

  glm::vec3 image_point = nx * camera_right + ny * camera_up + camera_eye + camera_target;
  glm::vec3 ray_direction = glm::normalize(image_point - camera_eye);
  return Ray(camera_eye, ray_direction);
}

inline glm::vec3 Scene::trace(const Ray& ray) const {
  float min_tnear = INFINITY;
  const Sphere* sphere = nullptr;

  for (Sphere const& value : spheres_) {
    float tnear = value.intersects(ray);
    if (tnear >= 0 && tnear <= min_tnear) {
      min_tnear = tnear, sphere = &value;
    }
  }

  if (!sphere) {
    return glm::vec3(0, 0, 0);
  }

  glm::vec3 surface_color(0.0f);
  glm::vec3 int_point = ray.origin() + ray.direction() * min_tnear;
  glm::vec3 int_normal = glm::normalize(int_point - sphere->center());

  return glm::vec3(68, 115, 19);
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




