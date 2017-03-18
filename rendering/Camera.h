#pragma once

#include "../geometry/Vector3.h"

class Camera
{

private:
  Vector3 point_;
  Vector3 direction_;
  double fieldOfView_;

public:
  Camera(Vector3& point, Vector3& direction, double fov) : 
    point_(point), direction_(direction), fieldOfView_(fov) {};

};

