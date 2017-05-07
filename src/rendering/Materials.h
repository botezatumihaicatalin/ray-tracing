#pragma once

#include "Material.h"
#include <glm/vec3.hpp>

class Materials {

public:
  static const Material EMERALD;
  static const Material JADE;
  static const Material OBSIDIAN;
  static const Material PEARL;
  static const Material RUBY;
  static const Material TURQUOISE;
  
  static const Material BRASS;
  static const Material BRONZE;
  static const Material CHROME;
  static const Material COPPER;
  static const Material GOLD;
  static const Material SILVER;

  static const Material BLACKPLASTIC;
  static const Material CYANPLASTIC;
  static const Material GREENPLASTIC;
  static const Material REDPLASTIC;
  static const Material WHITEPLASTIC;
  static const Material YELLOWPLASTIC;
  
  static const Material BLACKRUBBER;
  static const Material CYANRUBBER;
  static const Material GREENRUBBER;
  static const Material REDRUBBER;
  static const Material WHITERUBBER;
  static const Material YELLOWRUBBER;
};

const Material Materials::EMERALD = Material(glm::vec3(0.0215, 0.1745, 0.0215), glm::vec3(0.07568, 0.61424, 0.07568), glm::vec3(0.633, 0.727811, 0.633), 76.8);
const Material Materials::JADE = Material(glm::vec3(0.135, 0.2225, 0.1575), glm::vec3(0.54, 0.89, 0.63), glm::vec3(0.316228, 0.316228, 0.316228), 12.8);
const Material Materials::OBSIDIAN = Material(glm::vec3(0.05375, 0.05, 0.06625), glm::vec3(0.18275, 0.17, 0.22525), glm::vec3(0.332741, 0.328634, 0.346435), 38.4);
const Material Materials::PEARL = Material(glm::vec3(0.25, 0.20725, 0.20725), glm::vec3(1, 0.829, 0.829), glm::vec3(0.296648, 0.296648, 0.296648), 11.264);
const Material Materials::RUBY = Material(glm::vec3(0.1745, 0.01175, 0.01175), glm::vec3(0.61424, 0.04136, 0.04136), glm::vec3(0.727811, 0.626959, 0.626959), 76.8);
const Material Materials::TURQUOISE = Material(glm::vec3(0.1, 0.18725, 0.1745), glm::vec3(0.396, 0.74151, 0.69102), glm::vec3(0.297254, 0.30829, 0.306678), 12.8);

const Material Materials::BRASS = Material(glm::vec3(0.329412, 0.223529, 0.027451), glm::vec3(0.780392, 0.568627, 0.113725), glm::vec3(0.992157, 0.941176, 0.807843), 27.89743616);
const Material Materials::BRONZE = Material(glm::vec3(0.2125, 0.1275, 0.054), glm::vec3(0.714, 0.4284, 0.18144), glm::vec3(0.393548, 0.271906, 0.166721), 25.6);
const Material Materials::CHROME = Material(glm::vec3(0.25, 0.25, 0.25), glm::vec3(0.4, 0.4, 0.4), glm::vec3(0.774597, 0.774597, 0.774597), 76.8);
const Material Materials::COPPER = Material(glm::vec3(0.19125, 0.0735, 0.0225), glm::vec3(0.7038, 0.27048, 0.0828), glm::vec3(0.256777, 0.137622, 0.086014), 12.8);
const Material Materials::GOLD = Material(glm::vec3(0.24725, 0.1995, 0.0745), glm::vec3(0.75164, 0.60648, 0.22648), glm::vec3(0.628281, 0.555802, 0.366065), 51.2);
const Material Materials::SILVER = Material(glm::vec3(0.19225, 0.19225, 0.19225), glm::vec3(0.50754, 0.50754, 0.50754), glm::vec3(0.508273, 0.508273, 0.508273), 51.2);

const Material Materials::BLACKPLASTIC = Material(glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.01, 0.01, 0.01), glm::vec3(0.50, 0.50, 0.50), 32);
const Material Materials::CYANPLASTIC = Material(glm::vec3(0.0, 0.1, 0.06), glm::vec3(0.0, 0.50980392, 0.50980392), glm::vec3(0.50196078, 0.50196078, 0.50196078), 32);
const Material Materials::GREENPLASTIC = Material(glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.1, 0.35, 0.1), glm::vec3(0.45, 0.55, 0.45), 32);
const Material Materials::REDPLASTIC = Material(glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.5, 0.0, 0.0), glm::vec3(0.7, 0.6, 0.6), 32);
const Material Materials::WHITEPLASTIC = Material(glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.55, 0.55, 0.55), glm::vec3(0.70, 0.70, 0.70), 32);
const Material Materials::YELLOWPLASTIC = Material(glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.5, 0.5, 0.0), glm::vec3(0.60, 0.60, 0.50), 32);

const Material Materials::BLACKRUBBER = Material(glm::vec3(0.02, 0.02, 0.02), glm::vec3(0.01, 0.01, 0.01), glm::vec3(0.4, 0.4, 0.4), 10);
const Material Materials::CYANRUBBER = Material(glm::vec3(0.0, 0.05, 0.05), glm::vec3(0.4, 0.5, 0.5), glm::vec3(0.04, 0.7, 0.7), 10);
const Material Materials::GREENRUBBER = Material(glm::vec3(0.0, 0.05, 0.0), glm::vec3(0.4, 0.5, 0.4), glm::vec3(0.04, 0.7, 0.04), 10);
const Material Materials::REDRUBBER = Material(glm::vec3(0.05, 0.0, 0.0), glm::vec3(0.5, 0.4, 0.4), glm::vec3(0.7, 0.04, 0.04), 10);
const Material Materials::WHITERUBBER = Material(glm::vec3(0.05, 0.05, 0.05), glm::vec3(0.5, 0.5, 0.5), glm::vec3(0.7, 0.7, 0.7), 10);
const Material Materials::YELLOWRUBBER = Material(glm::vec3(0.05, 0.05, 0.0), glm::vec3(0.5, 0.5, 0.4), glm::vec3(0.7, 0.7, 0.04), 10);
