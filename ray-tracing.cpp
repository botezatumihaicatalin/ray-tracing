// ray-tracing.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "geometry/Vector3.h"
#include "rendering/Camera.h"

#include <iostream>

int main()
{

  Vector3 a(10, 10, 10);
  Vector3 b(20, 21, 22);
  Vector3 cross = a * b;

  printf("%f, %f, %f \r\n", cross[0], cross[1], cross[2]);
  system("pause");

  return 0;
}

