#pragma once

/*
* (C) Copyright Karol Dzitkowski 2015
*
*/

#define CUDA_CALL(x) do { assert((x)==cudaSuccess); } while(0)

#include <cuda_runtime_api.h>
#include <assert.h>
#include <algorithm>            // for std::swap
#include <atomic>
#include <cstddef>

template <class T>
class cuda_shared_ptr {
private:
  T* ptr_;
  std::atomic_int* cnt_;

public:
  explicit cuda_shared_ptr() {
    init_();
    CUDA_CALL(cudaMalloc(&ptr_, sizeof(T)));
  }

  explicit cuda_shared_ptr(size_t size) {
    init_();
    if (size == 0) {
      ptr_ = nullptr;
    }
    else {
      CUDA_CALL(cudaMalloc(&ptr_, size * sizeof(T)));
    }
  }

  explicit cuda_shared_ptr(T* p = nullptr) : ptr_(p) {
    init_();
  }

  ~cuda_shared_ptr() {
    if (dec_counter_()) {
      CUDA_CALL(cudaFree(ptr_));
      delete cnt_;
    }
  }

  void reset(T* p = nullptr) {
    assert(p == nullptr || p != ptr_);
    cuda_shared_ptr<T>(p).swap(*this);
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

  void swap(cuda_shared_ptr& other) noexcept {
    std::swap(ptr_, other.ptr_);
    std::swap(cnt_, other.cnt_);
  }

  int use_count() const {
    return *cnt_;
  }

private:
  bool dec_counter_() const {
    return --(*cnt_) == 0;
  }

  void inc_counter_() const {
    ++(*cnt_);
  }

  void init_() {
    cnt_ = new std::atomic<int>();
    cnt_->store(1);
  }

public:
  cuda_shared_ptr(cuda_shared_ptr const& r) : ptr_(r.ptr_), cnt_(r.cnt_) {
    inc_counter_();
  }

  /* NOT YET SUPPORTED
  cuda_shared_ptr( cuda_shared_ptr && r ) noexcept : ptr( r.ptr ), cnt()
  {
  swap(cnt, r.cnt);
  r.px = 0;
  }
  */

  cuda_shared_ptr& operator=(cuda_shared_ptr const& r) noexcept {
    cuda_shared_ptr<T>(r).swap(*this);
    return *this;
  }

  template <class Y>
  cuda_shared_ptr& operator=(cuda_shared_ptr<Y> const& r) noexcept {
    cuda_shared_ptr<T>(r).swap(*this);
    return *this;
  }
};

template <class T, class U>
inline bool operator==(cuda_shared_ptr<T> const& a, cuda_shared_ptr<U> const& b) noexcept {
  return a.get() == b.get();
}

template <class T, class U>
inline bool operator!=(cuda_shared_ptr<T> const& a, cuda_shared_ptr<U> const& b) noexcept {
  return a.get() != b.get();
}

template <class T>
inline bool operator==(cuda_shared_ptr<T> const& p, std::nullptr_t) noexcept {
  return p.get() == 0;
}

template <class T>
inline bool operator==(std::nullptr_t, cuda_shared_ptr<T> const& p) noexcept {
  return p.get() == 0;
}

template <class T>
inline bool operator!=(cuda_shared_ptr<T> const& p, std::nullptr_t) noexcept {
  return p.get() != 0;
}

template <class T>
inline bool operator!=(std::nullptr_t, cuda_shared_ptr<T> const& p) noexcept {
  return p.get() != 0;
}
