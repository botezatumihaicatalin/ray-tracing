// ray-tracing.cpp : Defines the entry point for the console application.
//

#include <fstream>

#include "rendering/Scene.h"
#include "glm/ext.hpp"
#include "cimg/CImg.h"

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

#ifndef INFINITY
#define INFINITY 1e8
#endif

int main()
{

  Scene scene(640 * 2, 480 * 2);
  glm::vec3* pixels = scene.render();

  cimg_library::CImg<float> image(scene.width(),scene.height(),1,3,0);

  uint32_t i = 0;
  for (size_t x = 0; x < scene.width(); x++) {
    for (size_t y = 0; y < scene.height(); y++, i++) {
      image(x, y, 0, 0) = pixels[i][0];
      image(x, y, 0, 1) = pixels[i][1];
      image(x, y, 0, 2) = pixels[i][2];
    }
  }

  cimg_library::CImgDisplay main_disp(image,"Click a point");
  while (!main_disp.is_closed()) {
    main_disp.wait();
  }

  delete[] pixels;

  return 0;
}

