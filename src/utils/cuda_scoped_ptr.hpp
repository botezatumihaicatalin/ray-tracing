#pragma once

#include "cuda_utils.hpp"

#include <cuda_runtime_api.h>
#include <assert.h>
#include <cstddef>

template <class T>
class cuda_scoped_ptr // noncopyable
{
private:
  T* ptr_;

  cuda_scoped_ptr(cuda_scoped_ptr const&) = delete;
  cuda_scoped_ptr& operator=(cuda_scoped_ptr const&) = delete;

  void operator==(cuda_scoped_ptr const&) const = delete;
  void operator!=(cuda_scoped_ptr const&) const = delete;

public:
  explicit cuda_scoped_ptr() {
    cudaCheck(cudaMalloc(&ptr_, sizeof(T)));
  }

  explicit cuda_scoped_ptr(size_t size) {
    if (size == 0) {
      ptr_ = nullptr;
    }
    else {
      cudaCheck(cudaMalloc(&ptr_, size * sizeof(T)));
    }
  }

  explicit cuda_scoped_ptr(T* p = nullptr) : ptr_(p) {}

  ~cuda_scoped_ptr() {
    cudaCheck(cudaFree(ptr_));
  }

  void reset(T* p = nullptr) {
    assert(p == nullptr || p != ptr_);
    cuda_scoped_ptr<T>(p).swap(*this);
  }

  T& operator*() const {
    assert(ptr_ != nullptr);
    return *ptr_;
  }

  T* operator->() const {
    assert(ptr_ != nullptr);
    return ptr_;
  }

  T* get() const {
    return ptr_;
  }

  operator bool() const {
    return ptr_ != nullptr;
  }

  void swap(cuda_scoped_ptr& csp) noexcept {
    T* tmp = csp.ptr_;
    csp.ptr_ = ptr_;
    ptr_ = tmp;
  }
};

template <class T>
inline bool operator==(
  cuda_scoped_ptr<T> const& p, std::nullptr_t) noexcept {
  return p.get() == 0;
}

template <class T>
inline bool operator==(
  std::nullptr_t, cuda_scoped_ptr<T> const& p) noexcept {
  return p.get() == 0;
}

template <class T>
inline bool operator!=(
  cuda_scoped_ptr<T> const& p, std::nullptr_t) noexcept {
  return p.get() != 0;
}

template <class T>
inline bool operator!=(
  std::nullptr_t, cuda_scoped_ptr<T> const& p) noexcept {
  return p.get() != 0;
}
