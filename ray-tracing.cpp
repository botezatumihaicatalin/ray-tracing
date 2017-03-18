// ray-tracing.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "rendering/Scene.h"
#include "glm/ext.hpp"
#include <fstream>

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

#ifndef INFINITY
#define INFINITY 1e8
#endif

int main()
{

  Scene scene(640, 480);
  glm::vec3* pixels = scene.render();
  system("pause");

  std::ofstream ofs("./out.ppm", std::ios::out | std::ios::binary);
  ofs << "P6\r\n" << scene.width() << " " << scene.height() << "\r\n255\r\n";
  for (uint32_t i = 0; i < scene.height() * scene.width(); ++i) {
    char r = (char)(pixels[i].x);
    char g = (char)(pixels[i].y);
    char b = (char)(pixels[i].z);
    ofs << r << g << b;
  }

  ofs.close();
  delete[] pixels;

  return 0;
}

