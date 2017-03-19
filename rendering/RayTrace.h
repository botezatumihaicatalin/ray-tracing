#pragma once
#include "Sphere.h"

class RayTrace {

private:
  const Sphere* sphere_;
  float tnear_;

public:

  RayTrace(const Sphere* sphere, float tnear) : sphere_(sphere), tnear_(tnear) {}

  const Sphere& sphere() const;
  float tnear() const;

  bool has_trace() const;
};

inline const Sphere& RayTrace::sphere() const {
  return *sphere_;
}

inline float RayTrace::tnear() const {
  return tnear_;
}

inline bool RayTrace::has_trace() const {
  return sphere_ != nullptr;
}




