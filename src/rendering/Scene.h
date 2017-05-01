#pragma once

#include <algorithm>
#include <vector>
#include <ctime>

#include "glm/glm.hpp"
#include "glm/vec3.hpp"

#include "../utils/cuda_scoped_ptr.hpp"

#include "Camera.h"
#include "Ray.h"
#include "Sphere.h"
#include "Light.h"
#include "RayTrace.h"
#include "Phong.h"

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
  glm::vec3* render() const;

  const size_t& width() const { return width_; }
  const size_t& height() const { return height_; }
  const bool& antialiasing() const { return antialiasing_;  }
  void antialiasing(bool value) { antialiasing_ = value; }

  Camera& camera() { return camera_; }
};

inline Scene::Scene(size_t width, size_t height) {
  width_ = width;
  height_ = height;

  fov_ = 60;
  antialiasing_ = true;
  
  camera_ = Camera(glm::vec3(0, 0, 0), glm::vec3(0, 0, 1));
  ratio_ = width / float(height);

  const Material sphere_material1(glm::vec3(0.2, 0.2, 0.22), glm::vec3(0.6, 0.7, 0.8), 
                                 glm::vec3(1, 1, 1), 8);
  const Material sphere_material2(glm::vec3(0.45, 0, 0), glm::vec3(0.8, 0, 0),
                                  glm::vec3(1, 1, 1), 40);

  spheres_.push_back(Sphere(glm::vec3(0, 0, 5), 0.5f, sphere_material1));
  spheres_.push_back(Sphere(glm::vec3(1, 1, 4), 0.7f, sphere_material2));

  //lights_.push_back(Light(glm::vec3(1, 0, 4)));
  lights_.push_back(Light(glm::vec3(0, -3, 5), glm::vec3(1.0f), glm::vec3(1.0f), 
                          glm::vec3(1.0f), glm::vec3(1.f, .3f, 0.0f)));
}

__device__ inline Ray make_ray(const Camera& camera, float x, float y, size_t width, size_t height) {
  float nx = (x / float(width)) - 0.5f;
  float ny = (y / float(height)) - 0.5f;

  glm::vec3 camera_right = glm::cross(camera.target(), camera.up());

  glm::vec3 image_point = nx * camera_right + ny * camera.up() + camera.eye() + camera.target();
  glm::vec3 ray_direction = image_point - camera.eye();
  return Ray(camera.eye(), ray_direction);
}

__device__ inline RayTrace trace_ray(const Ray& ray, Sphere* spheres, size_t spheres_count) {
  float min_tnear = INFINITY;
  const Sphere* sphere = nullptr;

  for (size_t i = 0; i < spheres_count; i++) {
    Sphere const& value = spheres[i];
    float tnear = value.intersection(ray);

    if (tnear >= 0 && tnear < min_tnear) {
      min_tnear = tnear, sphere = &value;
    }
  }

  return RayTrace(sphere, min_tnear);
}

__device__ inline glm::vec3 cast_ray(const Ray& ray, Sphere* spheres, size_t spheres_count, 
                                     Light* lights, size_t lights_count) {
  const RayTrace trace1 = trace_ray(ray, spheres, spheres_count);

  if (!trace1.has_trace()) {
    return glm::vec3(37/255.f, 148/255.f, 192/255.f);
  }

  const Sphere& sphere = *trace1.surface();
  const SurfaceProperties sphere_properties = sphere.properties(ray);
  const PhongIllumination phong(ray, sphere_properties);

  glm::vec3 surface_color = glm::vec3(0.03f, 0.03f, 0.03f) * glm::vec3(0.9f);

  for (size_t l_idx = 0; l_idx < lights_count; l_idx++) {
    Light const& light = lights[l_idx];

    glm::vec3 light_dir = light.position() - sphere_properties.point();
    float light_distance2 = glm::dot(light_dir, light_dir);

    Ray shadow_ray(sphere_properties.point(), light_dir);
    RayTrace shadow_trace = trace_ray(shadow_ray, spheres, spheres_count);
    float shadow_tnear2 = shadow_trace.tnear() * shadow_trace.tnear();

    bool isInShadow = shadow_trace.has_trace() && shadow_tnear2 < light_distance2;

    surface_color += phong.calc_colour(light, shadow_ray, isInShadow);
  }

  return glm::clamp(surface_color, glm::vec3(0.f), glm::vec3(1.f));
}

__global__ inline void render_kernel(size_t width, size_t height, Camera* camera, 
                                     Light* lights, size_t lights_count, 
                                     Sphere* spheres, size_t spheres_count, 
                                     bool antialiasing, glm::vec3* buffer) {
  
  // Compute the pixel position from the kernel arguments
  size_t x = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t y = (blockIdx.y * blockDim.y) + threadIdx.y;
  size_t buf_index = x + y * width;

  // First check if the x and y are inside the scene bounds.
  if (x >= width || y >= height) {
    return;
  }

  // When not antialiasing, just make one ray and trace it.
  if (!antialiasing) {
    const Ray the_ray = make_ray(*camera, x, y, width, height);
    buffer[buf_index] = cast_ray(the_ray, spheres, spheres_count, lights, lights_count);
    return;
  }

  glm::vec3 total_color(0);

  // Supersampling antialiasing (http://paulbourke.net/miscellaneous/aliasing/)
  // Create multiple rays and compute the mean color value.
  for (float dx = -0.5f; dx <= 0.5f; dx += 0.5f) {
    for (float dy = -0.5f; dy <= 0.5f; dy += 0.5f) {
      const Ray the_ray = make_ray(*camera, x + dx, y + dy, width, height);
      total_color += cast_ray(the_ray, spheres, spheres_count, lights, lights_count);
    }
  }

  buffer[buf_index] = total_color / 9.0f;
}

inline glm::vec3* Scene::render() const {

  cuda_scoped_ptr<glm::vec3> d_buffer(width_ * height_);
  cuda_scoped_ptr<Camera> d_camera(1);
  cuda_scoped_ptr<Light> d_lights(lights_.size());
  cuda_scoped_ptr<Sphere> d_spheres(spheres_.size());

  cudaMemcpy(d_camera.get(), &camera_, sizeof(Camera), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lights.get(), &lights_[0], lights_.size() * sizeof(Light), cudaMemcpyHostToDevice);
  cudaMemcpy(d_spheres.get(), &spheres_[0], spheres_.size() * sizeof(Sphere), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((width_ / threadsPerBlock.x) + 1, (height_ / threadsPerBlock.y) + 1);

  clock_t t0 = clock();
  render_kernel <<<numBlocks, threadsPerBlock>>> (width_, height_, d_camera.get(), d_lights.get(), 
                                                  lights_.size(), d_spheres.get(), spheres_.size(), 
                                                  antialiasing_, d_buffer.get());

  cudaDeviceSynchronize();
  clock_t t1 = clock();
  printf("Kernel render = %f secs\n", float(t1 - t0) / 1000);

  glm::vec3* buffer = new glm::vec3[width_ * height_];
  cudaMemcpy(buffer, d_buffer.get(), width_ * height_ * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  return buffer;
}




