#define GLM_FORCE_CUDA
#define CUDA_VERSION 8000

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <memory>
#include <ctime>

#include "rendering/Scene.h"
#include <cimg/CImg.h>

#include <tbb/tbb.h>
#include <tbb/tick_count.h>

void load_pixels(cimg_library::CImg<uint8_t>& image, glm::vec3* pixels_buf) {
  if (image.spectrum() != 3) {
    throw std::runtime_error("Can't copy");
  }

  size_t pixels_size = image.width() * image.height();

  tbb::parallel_for(tbb::blocked_range<size_t>(0, pixels_size), 
                    [&](const tbb::blocked_range<size_t>& range) {
    for (size_t idx = range.begin(), idx_end = range.end(); idx < idx_end; idx++) {
      size_t y = idx / image.width(), x = idx % image.width();
      for (size_t c = 0; c < 3; c++) {
        image(x, y, 0, c) = uint8_t(pixels_buf[idx][c] * 255);
      }
    }
  });
}

int main() {

  // Hack, because first malloc is very slow.
  int * t;
  cudaMalloc(&t, sizeof(int) * 1000);
  cudaFree(t);

  Scene scene(1280, 768);
  scene.antialiasing(true);
  cimg_library::CImg<uint8_t> image(scene.width(), scene.height(), 1, 3, 0);

  tbb::task_scheduler_init init(1);

  double render_avg_s = 0;
  double overall_avg_s = 0;
  size_t measures = 0;

  tbb::tick_count t_start, t_stop1, t_stop2;

  cimg_library::CImgDisplay main_disp(image, "W, A, S, D to move camera and E, R to rotate camera");
  while (!main_disp.is_closed()) {
    
    t_start = tbb::tick_count::now();
    std::unique_ptr<glm::vec3[]> pixels(scene.render());
    t_stop1 = tbb::tick_count::now();
    load_pixels(image, pixels.get());
    t_stop2 = tbb::tick_count::now();

    double duration1 = (t_stop1 - t_start).seconds();
    double duration2 = (t_stop2 - t_start).seconds();

    render_avg_s = (render_avg_s * measures + duration1) / (measures + 1);
    overall_avg_s = (overall_avg_s * measures + duration2) / (measures + 1);

    measures = measures + 1;

    printf("Render = %lf secs\n", render_avg_s);
    printf("Overall = %lf secs\n", overall_avg_s);

    image.display(main_disp);

    if (main_disp.is_keyW()) {
      scene.camera().move_forward(0.3f);
    }

    if (main_disp.is_keyS()) {
      scene.camera().move_backward(0.3f);
    }

    if (main_disp.is_keyA()) {
      scene.camera().move_left(0.3f);
    }

    if (main_disp.is_keyD()) {
      scene.camera().move_right(0.3f);
    }

    if (main_disp.is_keyR()) {
      scene.camera().rotate(0.1f);
    }

    if (main_disp.is_keyE()) {
      scene.camera().rotate(-0.1f);
    }

    if (main_disp.is_keyP()) {
      scene.antialiasing(!scene.antialiasing());
    }

    main_disp.wait();
  }

  return 0;
}
