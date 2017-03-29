#define GLM_FORCE_CUDA

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "rendering/Scene.h"
#include <ctime>
#include "cimg/CImg.h"
#include <memory>

int main() {
  Scene scene(1280, 768);
  scene.antialiasing(true);
  cimg_library::CImg<float> image(scene.width(), scene.height(), 1, 3, 0);

  cimg_library::CImgDisplay main_disp(image, "Click a point");
  while (!main_disp.is_closed()) {
    
    
    clock_t t0 = clock();
    std::unique_ptr<glm::vec3[]> pixels(scene.render());

    uint32_t i = 0;
    for (size_t y = 0; y < scene.height(); y++) {
      for (size_t x = 0; x < scene.width(); x++, i++) {
        for (size_t c = 0; c < 3; c++) {
          image(x, y, 0, c) = pixels[i][c];
        }
      }
    }
    clock_t t1 = clock();
    printf("Render = %f secs\n", float(t1 - t0) / 1000);

    image.display(main_disp);

    if (main_disp.is_keyW()) {
      scene.camera().move_forward(0.3f);
    }

    if (main_disp.is_keyS()) {
      scene.camera().move_backward(0.3f);
    }

    if (main_disp.is_keyR()) {
      scene.camera().rotate(-0.1f);
    }

    if (main_disp.is_keyE()) {
      scene.camera().rotate(0.1f);
    }

    main_disp.wait();
  }

  return 0;
}
