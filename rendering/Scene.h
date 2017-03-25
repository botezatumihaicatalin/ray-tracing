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
  bool antialiasing_;
  Camera camera_;
  size_t width_;
  size_t height_;
  float fov_;
  float ratio_;
  std::vector<Sphere> spheres_;
  std::vector<Light> lights_;

public:

  Scene(size_t width, size_t height);

  Ray make_ray(float x, float y) const;
  RayTrace trace_ray(const Ray& ray) const;
  glm::vec3 cast_ray(const Ray& ray) const;
  glm::vec3 pixel_color(float x, float y) const;
  glm::vec3* render() const;

  const size_t& width() const;
  const size_t& height() const;
  const bool& antialiasing() const;
  void antialiasing(bool value);
};

inline Scene::Scene(size_t width, size_t height) {
  width_ = width;
  height_ = height;

  fov_ = 60;
  antialiasing_ = true;
  
  camera_ = Camera(glm::vec3(0, 0, 0), glm::vec3(0, 0, 1));
  ratio_ = width / float(height);

  spheres_.push_back(Sphere(glm::vec3(0, 0, 5), 0.5f));
  spheres_.push_back(Sphere(glm::vec3(1, 1, 4), 0.7f));

  //lights_.push_back(Light(glm::vec3(1, 0, 4)));
  lights_.push_back(Light(glm::vec3(0, -3, 5), glm::vec3(0.5f), glm::vec3(1.0f), glm::vec3(1.0f)));
}

inline Ray Scene::make_ray(float x, float y) const {
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

  const Phong phong(ray, int_normal);

  for (Light const& light : lights_) {
    glm::vec3 light_dir = light.position() - int_point;
    float light_distance2 = glm::dot(light_dir, light_dir);

    Ray shadow_ray(int_point, light_dir);
    RayTrace shadow_trace = trace_ray(shadow_ray);
    float shadow_tnear = shadow_trace.tnear();

    bool isInShadow = shadow_trace.has_trace() && (shadow_tnear * shadow_tnear) < light_distance2;

    surface_color += phong.calc_colour(light, shadow_ray, isInShadow);
  }

  return surface_color * 255.0f;
}

inline glm::vec3 Scene::pixel_color(float x, float y) const {
  if (!antialiasing_) {
    return cast_ray(make_ray(x, y));
  }
  glm::vec3 total_color(0);

  // Supersampling
  for (float dx = -0.5f; dx <= 0.5f; dx += 0.5f) {
    for (float dy = -0.5f; dy <= 0.5f; dy += 0.5f) {
      total_color += cast_ray(make_ray(x + dx, y + dy));
    }
  }

  return total_color / 9.0f;
}

inline glm::vec3* Scene::render() const {
  glm::vec3* buffer = new glm::vec3[width_ * height_];

  unsigned i = 0;
  for (float x = 0; x < width_; ++x) {
    for (float y = 0; y < height_; ++y, i++) {
      buffer[i] = pixel_color(x, y);
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

inline const bool& Scene::antialiasing() const {
  return antialiasing_;
}

inline void Scene::antialiasing(bool value) {
  this->antialiasing_ = value;
}




