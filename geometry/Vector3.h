#pragma once
#include <cmath>

class Vector3 {

private:
  float buffer_[3];

public:
  explicit Vector3(float e = .0f) : Vector3(e, e, e) {};
  explicit Vector3(float x, float y, float z);

  float x() const;
  float y() const;
  float z() const;

  void x(float val);
  void y(float val);
  void z(float val);

  float length() const;
  void normalize();

  float operator [](size_t i) const;
  float& operator [](size_t i);

  static Vector3 add(const Vector3& l, const Vector3& r);
  static Vector3 sub(const Vector3& l, const Vector3& r);
  static Vector3 mult(const Vector3& l, float r);
  static Vector3 mult(float l, const Vector3& r);
  static float dot(const Vector3& l, const Vector3& r);
  static Vector3 cross(const Vector3& l, const Vector3& r);

  static const Vector3 unit_x;
  static const Vector3 unit_y;
  static const Vector3 unit_z;
  static const Vector3 zero;
  static const Vector3 one;

  ~Vector3() {}
};

const Vector3 Vector3::unit_x = Vector3(1.0f, 0.0f, 0.0f);
const Vector3 Vector3::unit_y = Vector3(0.0f, 1.0f, 0.0f);
const Vector3 Vector3::unit_z = Vector3(0.0f, 0.0f, 1.0f);
const Vector3 Vector3::one = Vector3(1.0f);
const Vector3 Vector3::zero = Vector3(0.0f);

inline Vector3::Vector3(float x, float y, float z) {
  buffer_[0] = x, buffer_[1] = y, buffer_[2] = z;
}

inline float Vector3::x() const {
  return buffer_[0];
}

inline float Vector3::y() const {
  return buffer_[1];
}

inline float Vector3::z() const {
  return buffer_[2];
}

inline void Vector3::x(float val) {
  buffer_[0] = val;
}

inline void Vector3::y(float val) {
  buffer_[1] = val;
}

inline void Vector3::z(float val) {
  buffer_[2] = val;
}

inline float Vector3::length() const {
  float sum = 0;
  for (size_t i = 0; i < 3; i++) {
    sum += buffer_[i] * buffer_[i];
  }
  return sqrt(sum);
}

inline void Vector3::normalize() {
  float scale = 1.0 / length();
  for (size_t i = 0; i < 3; i++) {
    buffer_[i] *= scale;
  }
}

inline float Vector3::operator[](size_t i) const {
  return buffer_[i];
}

inline float& Vector3::operator[](size_t i) {
  return buffer_[i];
}

inline Vector3 Vector3::add(const Vector3& l, const Vector3& r) {
  Vector3 result(0.0f, 0.0f, 0.0f);
  for (size_t i = 0; i < 3; i++) {
    result[i] = l[i] + r[i];
  }
  return result;
}

inline Vector3 Vector3::sub(const Vector3& l, const Vector3& r) {
  Vector3 result(0.0f, 0.0f, 0.0f);
  for (size_t i = 0; i < 3; i++) {
    result[i] = l[i] - r[i];
  }
  return result;
}

inline Vector3 Vector3::mult(const Vector3& l, float r) {
  Vector3 result(0.0f, 0.0f, 0.0f);
  for (size_t i = 0; i < 3; i++) {
    result[i] = l[i] * r;
  }
  return result;
}

inline Vector3 Vector3::mult(float l, const Vector3& r) {
  return mult(r, l);
}

inline float Vector3::dot(const Vector3& l, const Vector3& r) {
  float result = 0.0f;
  for (size_t i = 0; i < 3; i++) {
    result += r[i] * l[i];
  }
  return result;
}

inline Vector3 Vector3::cross(const Vector3& l, const Vector3& r) {
  return Vector3(
    l.y() * l.z() - l.z() * r.y(),
    l.z() * r.x() - l.x() * r.z(),
    l.z() * r.y() - l.y() * r.z()
  );
}


inline Vector3 operator+(const Vector3& l, const Vector3& r) {
  return Vector3::add(l, r);
}

inline Vector3 operator-(const Vector3& l, const Vector3& r) {
  return Vector3::sub(l, r);
}

inline Vector3 operator*(const Vector3& l, float r) {
  return Vector3::mult(l, r);
}

inline Vector3 operator*(float l, const Vector3& r) {
  return Vector3::mult(l, r);
}

inline Vector3 operator*(const Vector3& l , const Vector3& r) {
  return Vector3::cross(l, r);
}









