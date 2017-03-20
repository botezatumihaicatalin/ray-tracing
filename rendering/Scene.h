#pragma once

#include <algorithm>
#include <vector>

#include "../glm/glm.hpp"
#include "../glm/vec3.hpp"

#include "Camera.h"
#include "Ray.h"
#include "Sphere.h"
#include "Light.h"
#include "RayTrace.h"

class Scene {

private:
  Camera camera_;
  size_t width_;
  size_t height_;
  float fov_;
  float ratio_;
  std::vector<Sphere> spheres_;
  std::vector<Light> lights_;

public:

  Scene(size_t width, size_t height);

  Ray make_ray(size_t x, size_t y) const;
  RayTrace trace_ray(const Ray& ray) const;
  glm::vec3 cast_ray(const Ray& ray) const;
  glm::vec3* render() const;

  const size_t& width() const;
  const size_t& height() const;
};

inline Scene::Scene(size_t width, size_t height) {
  width_ = width;
  height_ = height;
  fov_ = 60;
  camera_ = Camera(glm::vec3(0, 0, 0), glm::vec3(0, 0, 1));
  ratio_ = width / float(height);

  spheres_.push_back(Sphere(glm::vec3(0, 0, 5), 0.5f));
  spheres_.push_back(Sphere(glm::vec3(1, 1, 4), 0.7f));

  //lights_.push_back(Light(glm::vec3(1, 0, 4)));
  lights_.push_back(Light(glm::vec3(0, -3, 5), glm::vec3(0.5f), glm::vec3(1.0f), glm::vec3(1.0f)));
}

inline Ray Scene::make_ray(size_t x, size_t y) const {
  float nx = (x / float(width_)) - 0.5f;
  float ny = (y / float(height_)) - 0.5f;

  glm::vec3 camera_right = glm::cross(camera_.target(), camera_.up());

  glm::vec3 image_point = nx * camera_right + ny * camera_.up() + camera_.eye() + camera_.target();
  glm::vec3 ray_direction = image_point - camera_.eye();
  return Ray(camera_.eye(), ray_direction);
}

inline RayTrace Scene::trace_ray(const Ray& ray) const {
  float min_tnear = INFINITY;
  const Sphere* sphere = nullptr;

  for (Sphere const& value : spheres_) {
    float tnear = value.intersects(ray);
    if (tnear >= 0 && tnear <= min_tnear) {
      min_tnear = tnear, sphere = &value;
    }
  }

  return RayTrace(sphere, min_tnear);
}

inline glm::vec3 Scene::cast_ray(const Ray& ray) const {
  const RayTrace trace1 = trace_ray(ray);

  if (!trace1.has_trace()) {
    return glm::vec3(62, 174, 218);
  }

  const Sphere& sphere = trace1.sphere();
  float ray_tnear = trace1.tnear();

  glm::vec3 surface_color = glm::vec3(0.03f, 0.03f, 0.03f) * glm::vec3(0.9f);
  glm::vec3 int_point = ray.origin() + ray.direction() * ray_tnear;
  glm::vec3 int_normal = glm::normalize(int_point - sphere.center());

  for (Light const& light : lights_) {
    glm::vec3 light_dir = light.position() - int_point;
    float light_distance2 = glm::dot(light_dir, light_dir);

    Ray shadow_ray(int_point, light_dir);
    RayTrace shadow_trace = trace_ray(shadow_ray);
    float shadow_tnear = shadow_trace.tnear();

    bool isInShadow = shadow_trace.has_trace() && (shadow_tnear * shadow_tnear) < light_distance2;

    float light_dot_normal = std::max(0.f, glm::dot(shadow_ray.direction(), int_normal));

    glm::vec3 h_direction = glm::normalize(shadow_ray.direction() - ray.direction());
    float reflection_dot_normal = std::max(0.f, glm::dot(h_direction, int_normal));

    surface_color += light.ambient() * glm::vec3(0.2, 0.2, 0.22);
    surface_color += (1 - isInShadow) * light_dot_normal * light.diffuse() * glm::vec3(0.6, 0.7, 0.8);
    surface_color += (1 - isInShadow) * pow(reflection_dot_normal, 10) * light.specular() * glm::vec3(1, 1, 1);
  }

  return surface_color * 255.0f;
}

inline glm::vec3* Scene::render() const {
  glm::vec3* buffer = new glm::vec3[width_ * height_];

  unsigned i = 0;
  for (unsigned x = 0; x < width_; ++x) {
    for (unsigned y = 0; y < height_; ++y, i++) {
      buffer[i] = cast_ray(make_ray(x, y));
    }
  }
  return buffer;
}

inline const size_t& Scene::width() const {
  return width_;
}

inline const size_t& Scene::height() const {
  return height_;
}




