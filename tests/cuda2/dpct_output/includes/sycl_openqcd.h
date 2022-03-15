//==---- sycl_openqcd.h ---------------------------*- C++ -*----------------==//
//
// Copyright (C) 2018 - 2021 Intel Corporation
// Copyright (C) 2022 Advanced Research Computing, University College London
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
// Statement of modifications:
// 1. This file has been renamed from atomic.hpp to sycl_openqcd.h
// 2. Header guards of this file have been renamed to reflect the new filename
// 3. From the original 'atomic.hpp', only the templated version of
// 'atomic_fetch_add()' and its float specialisation has been retained in this
// file. Rest of the functions have been deleted
// 4. The namespace 'dpct' has been renamed to 'sycl_openqcd'
//===----------------------------------------------------------------------===//

#ifndef __SYCL_OPENQCD_H__
#define __SYCL_OPENQCD_H__

#include <CL/sycl.hpp>

namespace sycl_openqcd {

/// Atomically add the value operand to the value at the addr and assign the
/// result to the value at addr, Int version.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to add to the value at \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T, cl::sycl::access::address_space addressSpace =
                          cl::sycl::access::address_space::global_space>
inline T atomic_fetch_add(
    T *addr, T operand,
    cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed)
{
  cl::sycl::atomic<T, addressSpace> obj(
      (cl::sycl::multi_ptr<T, addressSpace>(addr)));
  return cl::sycl::atomic_fetch_add(obj, operand, memoryOrder);
}

/// Atomically add the value operand to the value at the addr and assign the
/// result to the value at addr, Float version.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to add to the value at \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <cl::sycl::access::address_space addressSpace =
              cl::sycl::access::address_space::global_space>
inline float atomic_fetch_add(
    float *addr, float operand,
    cl::sycl::memory_order memoryOrder = cl::sycl::memory_order::relaxed)
{
  static_assert(sizeof(float) == sizeof(int), "Mismatched type size");

  cl::sycl::atomic<int, addressSpace> obj(
      (cl::sycl::multi_ptr<int, addressSpace>(reinterpret_cast<int *>(addr))));

  int old_value;
  float old_float_value;

  do {
    old_value = obj.load(memoryOrder);
    old_float_value = *reinterpret_cast<const float *>(&old_value);
    const float new_float_value = old_float_value + operand;
    const int new_value = *reinterpret_cast<const int *>(&new_float_value);
    if (obj.compare_exchange_strong(old_value, new_value, memoryOrder))
      break;
  } while (true);

  return old_float_value;
}

} // namespace sycl_openqcd
#endif // __SYCL_OPENQCD_H__
