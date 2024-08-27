#ifndef RECLIB_UTILS_COLORS_H
#define RECLIB_UTILS_COLORS_H

#include "reclib/data_types.h"
#include "reclib/math/eigen_glm_interface.h"

namespace reclib {

inline vec3 hsv2rgb(const vec3& hsv) {
  float hue = hsv[0];
  if (hue < 0) {
    hue = 360.f + hue;
  }

  float C = hsv[2] * hsv[1];
  float X = C * (1 - abs(std::fmod(hue / 60.f, 2.f) - 1));
  float m = hsv[2] - C;

  vec3 rgb_prime = vec3::Zero();

  if (hue >= 0 && hue < 60) {
    rgb_prime = vec3(C, X, 0);
  } else if (hue >= 60 && hue < 120) {
    rgb_prime = vec3(X, C, 0);
  } else if (hue >= 120 && hue < 180) {
    rgb_prime = vec3(0, C, X);
  } else if (hue >= 180 && hue < 240) {
    rgb_prime = vec3(0, X, C);
  } else if (hue >= 240 && hue < 300) {
    rgb_prime = vec3(X, 0, C);
  } else if (hue >= 300 && hue < 360) {
    rgb_prime = vec3(C, 0, X);
  }

  vec3 rgb = (rgb_prime + reclib::make_vec3(m));
  return rgb;
}

inline vec3 rgb2hsv(vec3 rgb) {
  if ((rgb.array() > Eigen::Array3f::Ones()).any()) {
    rgb = rgb / 255.f;
  }

  float Cmax = rgb.maxCoeff();
  float Cmin = rgb.minCoeff();
  float delta = Cmax - Cmin;

  // hue calculation
  float hue = 0;
  float saturation = 0;
  if (delta > 0) {
    if (Cmax == rgb[0]) {
      hue = 60 * std::fmod((rgb[1] - rgb[2]) / delta, 6.f);
    } else if (Cmax == rgb[1]) {
      hue = 60 * (((rgb[2] - rgb[0]) / delta) + 2.f);
    } else {
      hue = 60 * (((rgb[0] - rgb[1]) / delta) + 4.f);
    }
  }

  if (hue < 0) {
    hue = 360.f + hue;
  }

  if (Cmax > 0) {
    saturation = delta / Cmax;
  }

  float value = Cmax;

  vec3 hsv(hue, saturation, value);
  return hsv;
}

}  // namespace reclib

#endif