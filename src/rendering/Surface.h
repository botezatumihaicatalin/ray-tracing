#pragma once
#include "Ray.h"
#include <glm/vec3.hpp>
#include "Material.h"

class SurfaceProperties {

private:
  const glm::vec3 point_;
  const glm::vec3 normal_;
  const Material& material_;

public:

  __host__ __device__
  SurfaceProperties(const glm::vec3& point, const glm::vec3& normal, const Material& material) :
    point_(point), normal_(glm::normalize(normal)), material_(material) { }

  __host__ __device__
  const glm::vec3& point() const { return point_; }

  __host__ __device__
  const glm::vec3& normal() const { return normal_; }

  __host__ __device__
  const Material& material() const { return material_; }
};

class Surface {

public:

  __host__ __device__
  virtual ~Surface() = default;

  __host__ __device__
  virtual float intersection(const Ray& ray) const = 0;

  __host__ __device__
  virtual SurfaceProperties properties(const glm::vec3& point) const = 0;
};
