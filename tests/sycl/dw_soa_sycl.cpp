// #include "dw_cuda_soa.h"
#include "macros.h"
#include "su3.h"
#include "sycl_openqcd.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <cstring>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace cl;

void vector_add_assign(su3_vector_soa r, su3_vector s, int idx)
{
  sycl_openqcd::atomic_fetch_add(&(r.c1.re[idx]), s.c1.re);
  sycl_openqcd::atomic_fetch_add(&(r.c1.im[idx]), s.c1.im);
  sycl_openqcd::atomic_fetch_add(&(r.c2.re[idx]), s.c2.re);
  sycl_openqcd::atomic_fetch_add(&(r.c2.im[idx]), s.c2.im);
  sycl_openqcd::atomic_fetch_add(&(r.c3.re[idx]), s.c3.re);
  sycl_openqcd::atomic_fetch_add(&(r.c3.im[idx]), s.c3.im);
}

void vector_sub_assign(su3_vector_soa r, su3_vector s, int idx)
{
  sycl_openqcd::atomic_fetch_add(&(r.c1.re[idx]), -s.c1.re);
  sycl_openqcd::atomic_fetch_add(&(r.c1.im[idx]), -s.c1.im);
  sycl_openqcd::atomic_fetch_add(&(r.c2.re[idx]), -s.c2.re);
  sycl_openqcd::atomic_fetch_add(&(r.c2.im[idx]), -s.c2.im);
  sycl_openqcd::atomic_fetch_add(&(r.c3.re[idx]), -s.c3.re);
  sycl_openqcd::atomic_fetch_add(&(r.c3.im[idx]), -s.c3.im);
}

void vector_i_add_assign(su3_vector_soa r, su3_vector s, int idx)
{
  sycl_openqcd::atomic_fetch_add(&(r.c1.re[idx]), -s.c1.im);
  sycl_openqcd::atomic_fetch_add(&(r.c1.im[idx]), s.c1.re);
  sycl_openqcd::atomic_fetch_add(&(r.c2.re[idx]), -s.c2.im);
  sycl_openqcd::atomic_fetch_add(&(r.c2.im[idx]), s.c2.re);
  sycl_openqcd::atomic_fetch_add(&(r.c3.re[idx]), -s.c3.im);
  sycl_openqcd::atomic_fetch_add(&(r.c3.im[idx]), s.c3.re);
}

void vector_i_sub_assign(su3_vector_soa r, su3_vector s, int idx)
{
  sycl_openqcd::atomic_fetch_add(&(r.c1.re[idx]), s.c1.im);
  sycl_openqcd::atomic_fetch_add(&(r.c1.im[idx]), -s.c1.re);
  sycl_openqcd::atomic_fetch_add(&(r.c2.re[idx]), s.c2.im);
  sycl_openqcd::atomic_fetch_add(&(r.c2.im[idx]), -s.c2.re);
  sycl_openqcd::atomic_fetch_add(&(r.c3.re[idx]), s.c3.im);
  sycl_openqcd::atomic_fetch_add(&(r.c3.im[idx]), -s.c3.re);
}

pauli_soa allocPauli2Device(int vol, sycl::queue &q_ct1)
{
  pauli_soa d_m;

  // Allocate memory on device
  d_m.m1 = sycl::malloc_device<float>(36 * vol, q_ct1);
  d_m.m2 = sycl::malloc_device<float>(36 * vol, q_ct1);

  return d_m;
}

su3_soa allocSu32Device(int vol, sycl::queue &q_ct1)
{
  su3_soa d_u;

  d_u.c11.re = sycl::malloc_device<float>(4 * vol, q_ct1);
  d_u.c11.im = sycl::malloc_device<float>(4 * vol, q_ct1);
  d_u.c12.re = sycl::malloc_device<float>(4 * vol, q_ct1);
  d_u.c12.im = sycl::malloc_device<float>(4 * vol, q_ct1);
  d_u.c13.re = sycl::malloc_device<float>(4 * vol, q_ct1);
  d_u.c13.im = sycl::malloc_device<float>(4 * vol, q_ct1);
  d_u.c21.re = sycl::malloc_device<float>(4 * vol, q_ct1);
  d_u.c21.im = sycl::malloc_device<float>(4 * vol, q_ct1);
  d_u.c22.re = sycl::malloc_device<float>(4 * vol, q_ct1);
  d_u.c22.im = sycl::malloc_device<float>(4 * vol, q_ct1);
  d_u.c23.re = sycl::malloc_device<float>(4 * vol, q_ct1);
  d_u.c23.im = sycl::malloc_device<float>(4 * vol, q_ct1);
  d_u.c31.re = sycl::malloc_device<float>(4 * vol, q_ct1);
  d_u.c31.im = sycl::malloc_device<float>(4 * vol, q_ct1);
  d_u.c32.re = sycl::malloc_device<float>(4 * vol, q_ct1);
  d_u.c32.im = sycl::malloc_device<float>(4 * vol, q_ct1);
  d_u.c33.re = sycl::malloc_device<float>(4 * vol, q_ct1);
  d_u.c33.im = sycl::malloc_device<float>(4 * vol, q_ct1);

  return d_u;
}

spinor_soa allocSpinor2Device(int vol, sycl::queue &q_ct1)
{
  spinor_soa d_s;

  d_s.c1.c1.re = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c1.c1.im = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c1.c2.re = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c1.c2.im = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c1.c3.re = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c1.c3.im = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c2.c1.re = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c2.c1.im = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c2.c2.re = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c2.c2.im = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c2.c3.re = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c2.c3.im = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c3.c1.re = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c3.c1.im = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c3.c2.re = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c3.c2.im = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c3.c3.re = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c3.c3.im = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c4.c1.re = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c4.c1.im = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c4.c2.re = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c4.c2.im = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c4.c3.re = sycl::malloc_device<float>(vol, q_ct1);
  d_s.c4.c3.im = sycl::malloc_device<float>(vol, q_ct1);

  return d_s;
}

void destroy_pauli_soa(pauli_soa obj, sycl::queue &q_ct1)
{
  sycl::free(obj.m1, q_ct1);
  sycl::free(obj.m2, q_ct1);
}

void destroy_su3_soa(su3_soa obj, sycl::queue &q_ct1)
{
  sycl::free(obj.c11.re, q_ct1);
  sycl::free(obj.c11.im, q_ct1);
  sycl::free(obj.c12.re, q_ct1);
  sycl::free(obj.c12.im, q_ct1);
  sycl::free(obj.c13.re, q_ct1);
  sycl::free(obj.c13.im, q_ct1);
  sycl::free(obj.c21.re, q_ct1);
  sycl::free(obj.c21.im, q_ct1);
  sycl::free(obj.c22.re, q_ct1);
  sycl::free(obj.c22.im, q_ct1);
  sycl::free(obj.c23.re, q_ct1);
  sycl::free(obj.c23.im, q_ct1);
  sycl::free(obj.c31.re, q_ct1);
  sycl::free(obj.c31.im, q_ct1);
  sycl::free(obj.c32.re, q_ct1);
  sycl::free(obj.c32.im, q_ct1);
  sycl::free(obj.c33.re, q_ct1);
  sycl::free(obj.c33.im, q_ct1);
}

void destroy_spinor_soa(spinor_soa obj, sycl::queue &q_ct1)
{
  sycl::free(obj.c1.c1.re, q_ct1);
  sycl::free(obj.c1.c1.im, q_ct1);
  sycl::free(obj.c1.c2.re, q_ct1);
  sycl::free(obj.c1.c2.im, q_ct1);
  sycl::free(obj.c1.c3.re, q_ct1);
  sycl::free(obj.c1.c3.im, q_ct1);
  sycl::free(obj.c2.c1.re, q_ct1);
  sycl::free(obj.c2.c1.im, q_ct1);
  sycl::free(obj.c2.c2.re, q_ct1);
  sycl::free(obj.c2.c2.im, q_ct1);
  sycl::free(obj.c2.c3.re, q_ct1);
  sycl::free(obj.c2.c3.im, q_ct1);
  sycl::free(obj.c3.c1.re, q_ct1);
  sycl::free(obj.c3.c1.im, q_ct1);
  sycl::free(obj.c3.c2.re, q_ct1);
  sycl::free(obj.c3.c2.im, q_ct1);
  sycl::free(obj.c3.c3.re, q_ct1);
  sycl::free(obj.c3.c3.im, q_ct1);
  sycl::free(obj.c4.c1.re, q_ct1);
  sycl::free(obj.c4.c1.im, q_ct1);
  sycl::free(obj.c4.c2.re, q_ct1);
  sycl::free(obj.c4.c2.im, q_ct1);
  sycl::free(obj.c4.c3.re, q_ct1);
  sycl::free(obj.c4.c3.im, q_ct1);
}

// Kernel to convert pauli from AoS to SoA in GPU
// extern "C"
void pauli_AoS2SoA(int vol, pauli_soa mout, pauli *min, sycl::nd_item<1> item_ct1)
{
  int idx = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0);
  if (idx >= vol)
    return;

  int j;

  // A+
  for (j = 0; j < 36; ++j) {
    mout.m1[j * vol + idx] = (*(min + 2 * idx + 0)).u[j];
  }

  // A-
  for (j = 0; j < 36; ++j) {
    mout.m2[j * vol + idx] = (*(min + 2 * idx + 1)).u[j];
  }
}

// Kernel to convert su3 from AoS to SoA in GPU
extern "C" void su3_AoS2SoA(int vol, su3_soa uout, su3 *uin, sycl::nd_item<1> item_ct1)
{
  int idx = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0);
  if (idx >= vol / 2)
    return;

  int idSoA, idAoS;

  for (int j = 0; j < 8; ++j) {
    idSoA = (vol / 2) * j + idx;
    idAoS = 8 * idx + j;
    uout.c11.re[idSoA] = (*(uin + idAoS)).c11.re;
    uout.c11.im[idSoA] = (*(uin + idAoS)).c11.im;
    uout.c12.re[idSoA] = (*(uin + idAoS)).c12.re;
    uout.c12.im[idSoA] = (*(uin + idAoS)).c12.im;
    uout.c13.re[idSoA] = (*(uin + idAoS)).c13.re;
    uout.c13.im[idSoA] = (*(uin + idAoS)).c13.im;
    uout.c21.re[idSoA] = (*(uin + idAoS)).c21.re;
    uout.c21.im[idSoA] = (*(uin + idAoS)).c21.im;
    uout.c22.re[idSoA] = (*(uin + idAoS)).c22.re;
    uout.c22.im[idSoA] = (*(uin + idAoS)).c22.im;
    uout.c23.re[idSoA] = (*(uin + idAoS)).c23.re;
    uout.c23.im[idSoA] = (*(uin + idAoS)).c23.im;
    uout.c31.re[idSoA] = (*(uin + idAoS)).c31.re;
    uout.c31.im[idSoA] = (*(uin + idAoS)).c31.im;
    uout.c32.re[idSoA] = (*(uin + idAoS)).c32.re;
    uout.c32.im[idSoA] = (*(uin + idAoS)).c32.im;
    uout.c33.re[idSoA] = (*(uin + idAoS)).c33.re;
    uout.c33.im[idSoA] = (*(uin + idAoS)).c33.im;
  }
}

// Kernel to convert spinor from AoS to SoA in GPU
extern "C" void spinor_AoS2SoA(int vol, spinor_soa rout, spinor *rin, sycl::nd_item<1> item_ct1)
{
  int idx = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0);
  if (idx >= vol)
    return;

  rout.c1.c1.re[idx] = (*(rin + idx)).c1.c1.re;
  rout.c1.c1.im[idx] = (*(rin + idx)).c1.c1.im;
  rout.c1.c2.re[idx] = (*(rin + idx)).c1.c2.re;
  rout.c1.c2.im[idx] = (*(rin + idx)).c1.c2.im;
  rout.c1.c3.re[idx] = (*(rin + idx)).c1.c3.re;
  rout.c1.c3.im[idx] = (*(rin + idx)).c1.c3.im;
  rout.c2.c1.re[idx] = (*(rin + idx)).c2.c1.re;
  rout.c2.c1.im[idx] = (*(rin + idx)).c2.c1.im;
  rout.c2.c2.re[idx] = (*(rin + idx)).c2.c2.re;
  rout.c2.c2.im[idx] = (*(rin + idx)).c2.c2.im;
  rout.c2.c3.re[idx] = (*(rin + idx)).c2.c3.re;
  rout.c2.c3.im[idx] = (*(rin + idx)).c2.c3.im;
  rout.c3.c1.re[idx] = (*(rin + idx)).c3.c1.re;
  rout.c3.c1.im[idx] = (*(rin + idx)).c3.c1.im;
  rout.c3.c2.re[idx] = (*(rin + idx)).c3.c2.re;
  rout.c3.c2.im[idx] = (*(rin + idx)).c3.c2.im;
  rout.c3.c3.re[idx] = (*(rin + idx)).c3.c3.re;
  rout.c3.c3.im[idx] = (*(rin + idx)).c3.c3.im;
  rout.c4.c1.re[idx] = (*(rin + idx)).c4.c1.re;
  rout.c4.c1.im[idx] = (*(rin + idx)).c4.c1.im;
  rout.c4.c2.re[idx] = (*(rin + idx)).c4.c2.re;
  rout.c4.c2.im[idx] = (*(rin + idx)).c4.c2.im;
  rout.c4.c3.re[idx] = (*(rin + idx)).c4.c3.re;
  rout.c4.c3.im[idx] = (*(rin + idx)).c4.c3.im;
}

// Kernel to convert spinor from SoA to AoS in GPU
extern "C" void spinor_SoA2AoS(int vol, spinor *rout, spinor_soa rin, sycl::nd_item<1> item_ct1)
{
  int idx = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0);
  if (idx >= vol)
    return;

  (*(rout + idx)).c1.c1.re = rin.c1.c1.re[idx];
  (*(rout + idx)).c1.c1.im = rin.c1.c1.im[idx];
  (*(rout + idx)).c1.c2.re = rin.c1.c2.re[idx];
  (*(rout + idx)).c1.c2.im = rin.c1.c2.im[idx];
  (*(rout + idx)).c1.c3.re = rin.c1.c3.re[idx];
  (*(rout + idx)).c1.c3.im = rin.c1.c3.im[idx];
  (*(rout + idx)).c2.c1.re = rin.c2.c1.re[idx];
  (*(rout + idx)).c2.c1.im = rin.c2.c1.im[idx];
  (*(rout + idx)).c2.c2.re = rin.c2.c2.re[idx];
  (*(rout + idx)).c2.c2.im = rin.c2.c2.im[idx];
  (*(rout + idx)).c2.c3.re = rin.c2.c3.re[idx];
  (*(rout + idx)).c2.c3.im = rin.c2.c3.im[idx];
  (*(rout + idx)).c3.c1.re = rin.c3.c1.re[idx];
  (*(rout + idx)).c3.c1.im = rin.c3.c1.im[idx];
  (*(rout + idx)).c3.c2.re = rin.c3.c2.re[idx];
  (*(rout + idx)).c3.c2.im = rin.c3.c2.im[idx];
  (*(rout + idx)).c3.c3.re = rin.c3.c3.re[idx];
  (*(rout + idx)).c3.c3.im = rin.c3.c3.im[idx];
  (*(rout + idx)).c4.c1.re = rin.c4.c1.re[idx];
  (*(rout + idx)).c4.c1.im = rin.c4.c1.im[idx];
  (*(rout + idx)).c4.c2.re = rin.c4.c2.re[idx];
  (*(rout + idx)).c4.c2.im = rin.c4.c2.im[idx];
  (*(rout + idx)).c4.c3.re = rin.c4.c3.re[idx];
  (*(rout + idx)).c4.c3.im = rin.c4.c3.im[idx];
}

// ---------------------------------------------------------------------------//
extern "C" void mulpauli_kernel(int vol, float mu, spinor_soa s, spinor_soa r,
                                pauli_soa m, sycl::nd_item<1> item_ct1)
{
  int idx = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0);
  if (idx >= vol)
    return;

  float u[36];
  weyl sloc;

  // A+
  sloc.c1.c1.re = s.c1.c1.re[idx];
  sloc.c1.c1.im = s.c1.c1.im[idx];
  sloc.c1.c2.re = s.c1.c2.re[idx];
  sloc.c1.c2.im = s.c1.c2.im[idx];
  sloc.c1.c3.re = s.c1.c3.re[idx];
  sloc.c1.c3.im = s.c1.c3.im[idx];
  sloc.c2.c1.re = s.c2.c1.re[idx];
  sloc.c2.c1.im = s.c2.c1.im[idx];
  sloc.c2.c2.re = s.c2.c2.re[idx];
  sloc.c2.c2.im = s.c2.c2.im[idx];
  sloc.c2.c3.re = s.c2.c3.re[idx];
  sloc.c2.c3.im = s.c2.c3.im[idx];

#pragma unroll
  for (int i = 0; i < 36; ++i) {
    u[i] = m.m1[i * vol + idx];
  }

  r.c1.c1.re[idx] =
      u[0] * sloc.c1.c1.re - mu * sloc.c1.c1.im + u[6] * sloc.c1.c2.re -
      u[7] * sloc.c1.c2.im + u[8] * sloc.c1.c3.re - u[9] * sloc.c1.c3.im +
      u[10] * sloc.c2.c1.re - u[11] * sloc.c2.c1.im + u[12] * sloc.c2.c2.re -
      u[13] * sloc.c2.c2.im + u[14] * sloc.c2.c3.re - u[15] * sloc.c2.c3.im;

  r.c1.c1.im[idx] =
      u[0] * sloc.c1.c1.im + mu * sloc.c1.c1.re + u[6] * sloc.c1.c2.im +
      u[7] * sloc.c1.c2.re + u[8] * sloc.c1.c3.im + u[9] * sloc.c1.c3.re +
      u[10] * sloc.c2.c1.im + u[11] * sloc.c2.c1.re + u[12] * sloc.c2.c2.im +
      u[13] * sloc.c2.c2.re + u[14] * sloc.c2.c3.im + u[15] * sloc.c2.c3.re;

  r.c1.c2.re[idx] =
      u[6] * sloc.c1.c1.re + u[7] * sloc.c1.c1.im + u[1] * sloc.c1.c2.re -
      mu * sloc.c1.c2.im + u[16] * sloc.c1.c3.re - u[17] * sloc.c1.c3.im +
      u[18] * sloc.c2.c1.re - u[19] * sloc.c2.c1.im + u[20] * sloc.c2.c2.re -
      u[21] * sloc.c2.c2.im + u[22] * sloc.c2.c3.re - u[23] * sloc.c2.c3.im;

  r.c1.c2.im[idx] =
      u[6] * sloc.c1.c1.im - u[7] * sloc.c1.c1.re + u[1] * sloc.c1.c2.im +
      mu * sloc.c1.c2.re + u[16] * sloc.c1.c3.im + u[17] * sloc.c1.c3.re +
      u[18] * sloc.c2.c1.im + u[19] * sloc.c2.c1.re + u[20] * sloc.c2.c2.im +
      u[21] * sloc.c2.c2.re + u[22] * sloc.c2.c3.im + u[23] * sloc.c2.c3.re;

  r.c1.c3.re[idx] =
      u[8] * sloc.c1.c1.re + u[9] * sloc.c1.c1.im + u[16] * sloc.c1.c2.re +
      u[17] * sloc.c1.c2.im + u[2] * sloc.c1.c3.re - mu * sloc.c1.c3.im +
      u[24] * sloc.c2.c1.re - u[25] * sloc.c2.c1.im + u[26] * sloc.c2.c2.re -
      u[27] * sloc.c2.c2.im + u[28] * sloc.c2.c3.re - u[29] * sloc.c2.c3.im;

  r.c1.c3.im[idx] =
      u[8] * sloc.c1.c1.im - u[9] * sloc.c1.c1.re + u[16] * sloc.c1.c2.im -
      u[17] * sloc.c1.c2.re + u[2] * sloc.c1.c3.im + mu * sloc.c1.c3.re +
      u[24] * sloc.c2.c1.im + u[25] * sloc.c2.c1.re + u[26] * sloc.c2.c2.im +
      u[27] * sloc.c2.c2.re + u[28] * sloc.c2.c3.im + u[29] * sloc.c2.c3.re;

  r.c2.c1.re[idx] =
      u[10] * sloc.c1.c1.re + u[11] * sloc.c1.c1.im + u[18] * sloc.c1.c2.re +
      u[19] * sloc.c1.c2.im + u[24] * sloc.c1.c3.re + u[25] * sloc.c1.c3.im +
      u[3] * sloc.c2.c1.re - mu * sloc.c2.c1.im + u[30] * sloc.c2.c2.re -
      u[31] * sloc.c2.c2.im + u[32] * sloc.c2.c3.re - u[33] * sloc.c2.c3.im;

  r.c2.c1.im[idx] =
      u[10] * sloc.c1.c1.im - u[11] * sloc.c1.c1.re + u[18] * sloc.c1.c2.im -
      u[19] * sloc.c1.c2.re + u[24] * sloc.c1.c3.im - u[25] * sloc.c1.c3.re +
      u[3] * sloc.c2.c1.im + mu * sloc.c2.c1.re + u[30] * sloc.c2.c2.im +
      u[31] * sloc.c2.c2.re + u[32] * sloc.c2.c3.im + u[33] * sloc.c2.c3.re;

  r.c2.c2.re[idx] =
      u[12] * sloc.c1.c1.re + u[13] * sloc.c1.c1.im + u[20] * sloc.c1.c2.re +
      u[21] * sloc.c1.c2.im + u[26] * sloc.c1.c3.re + u[27] * sloc.c1.c3.im +
      u[30] * sloc.c2.c1.re + u[31] * sloc.c2.c1.im + u[4] * sloc.c2.c2.re -
      mu * sloc.c2.c2.im + u[34] * sloc.c2.c3.re - u[35] * sloc.c2.c3.im;

  r.c2.c2.im[idx] =
      u[12] * sloc.c1.c1.im - u[13] * sloc.c1.c1.re + u[20] * sloc.c1.c2.im -
      u[21] * sloc.c1.c2.re + u[26] * sloc.c1.c3.im - u[27] * sloc.c1.c3.re +
      u[30] * sloc.c2.c1.im - u[31] * sloc.c2.c1.re + u[4] * sloc.c2.c2.im +
      mu * sloc.c2.c2.re + u[34] * sloc.c2.c3.im + u[35] * sloc.c2.c3.re;

  r.c2.c3.re[idx] =
      u[14] * sloc.c1.c1.re + u[15] * sloc.c1.c1.im + u[22] * sloc.c1.c2.re +
      u[23] * sloc.c1.c2.im + u[28] * sloc.c1.c3.re + u[29] * sloc.c1.c3.im +
      u[32] * sloc.c2.c1.re + u[33] * sloc.c2.c1.im + u[34] * sloc.c2.c2.re +
      u[35] * sloc.c2.c2.im + u[5] * sloc.c2.c3.re - mu * sloc.c2.c3.im;

  r.c2.c3.im[idx] =
      u[14] * sloc.c1.c1.im - u[15] * sloc.c1.c1.re + u[22] * sloc.c1.c2.im -
      u[23] * sloc.c1.c2.re + u[28] * sloc.c1.c3.im - u[29] * sloc.c1.c3.re +
      u[32] * sloc.c2.c1.im - u[33] * sloc.c2.c1.re + u[34] * sloc.c2.c2.im -
      u[35] * sloc.c2.c2.re + u[5] * sloc.c2.c3.im + mu * sloc.c2.c3.re;

  // A-
  sloc.c1.c1.re = s.c3.c1.re[idx];
  sloc.c1.c1.im = s.c3.c1.im[idx];
  sloc.c1.c2.re = s.c3.c2.re[idx];
  sloc.c1.c2.im = s.c3.c2.im[idx];
  sloc.c1.c3.re = s.c3.c3.re[idx];
  sloc.c1.c3.im = s.c3.c3.im[idx];
  sloc.c2.c1.re = s.c4.c1.re[idx];
  sloc.c2.c1.im = s.c4.c1.im[idx];
  sloc.c2.c2.re = s.c4.c2.re[idx];
  sloc.c2.c2.im = s.c4.c2.im[idx];
  sloc.c2.c3.re = s.c4.c3.re[idx];
  sloc.c2.c3.im = s.c4.c3.im[idx];

#pragma unroll
  for (int i = 0; i < 36; ++i) {
    u[i] = m.m2[i * vol + idx];
  }

  mu = -mu;

  r.c3.c1.re[idx] =
      u[0] * sloc.c1.c1.re - mu * sloc.c1.c1.im + u[6] * sloc.c1.c2.re -
      u[7] * sloc.c1.c2.im + u[8] * sloc.c1.c3.re - u[9] * sloc.c1.c3.im +
      u[10] * sloc.c2.c1.re - u[11] * sloc.c2.c1.im + u[12] * sloc.c2.c2.re -
      u[13] * sloc.c2.c2.im + u[14] * sloc.c2.c3.re - u[15] * sloc.c2.c3.im;

  r.c3.c1.im[idx] =
      u[0] * sloc.c1.c1.im + mu * sloc.c1.c1.re + u[6] * sloc.c1.c2.im +
      u[7] * sloc.c1.c2.re + u[8] * sloc.c1.c3.im + u[9] * sloc.c1.c3.re +
      u[10] * sloc.c2.c1.im + u[11] * sloc.c2.c1.re + u[12] * sloc.c2.c2.im +
      u[13] * sloc.c2.c2.re + u[14] * sloc.c2.c3.im + u[15] * sloc.c2.c3.re;

  r.c3.c2.re[idx] =
      u[6] * sloc.c1.c1.re + u[7] * sloc.c1.c1.im + u[1] * sloc.c1.c2.re -
      mu * sloc.c1.c2.im + u[16] * sloc.c1.c3.re - u[17] * sloc.c1.c3.im +
      u[18] * sloc.c2.c1.re - u[19] * sloc.c2.c1.im + u[20] * sloc.c2.c2.re -
      u[21] * sloc.c2.c2.im + u[22] * sloc.c2.c3.re - u[23] * sloc.c2.c3.im;

  r.c3.c2.im[idx] =
      u[6] * sloc.c1.c1.im - u[7] * sloc.c1.c1.re + u[1] * sloc.c1.c2.im +
      mu * sloc.c1.c2.re + u[16] * sloc.c1.c3.im + u[17] * sloc.c1.c3.re +
      u[18] * sloc.c2.c1.im + u[19] * sloc.c2.c1.re + u[20] * sloc.c2.c2.im +
      u[21] * sloc.c2.c2.re + u[22] * sloc.c2.c3.im + u[23] * sloc.c2.c3.re;

  r.c3.c3.re[idx] =
      u[8] * sloc.c1.c1.re + u[9] * sloc.c1.c1.im + u[16] * sloc.c1.c2.re +
      u[17] * sloc.c1.c2.im + u[2] * sloc.c1.c3.re - mu * sloc.c1.c3.im +
      u[24] * sloc.c2.c1.re - u[25] * sloc.c2.c1.im + u[26] * sloc.c2.c2.re -
      u[27] * sloc.c2.c2.im + u[28] * sloc.c2.c3.re - u[29] * sloc.c2.c3.im;

  r.c3.c3.im[idx] =
      u[8] * sloc.c1.c1.im - u[9] * sloc.c1.c1.re + u[16] * sloc.c1.c2.im -
      u[17] * sloc.c1.c2.re + u[2] * sloc.c1.c3.im + mu * sloc.c1.c3.re +
      u[24] * sloc.c2.c1.im + u[25] * sloc.c2.c1.re + u[26] * sloc.c2.c2.im +
      u[27] * sloc.c2.c2.re + u[28] * sloc.c2.c3.im + u[29] * sloc.c2.c3.re;

  r.c4.c1.re[idx] =
      u[10] * sloc.c1.c1.re + u[11] * sloc.c1.c1.im + u[18] * sloc.c1.c2.re +
      u[19] * sloc.c1.c2.im + u[24] * sloc.c1.c3.re + u[25] * sloc.c1.c3.im +
      u[3] * sloc.c2.c1.re - mu * sloc.c2.c1.im + u[30] * sloc.c2.c2.re -
      u[31] * sloc.c2.c2.im + u[32] * sloc.c2.c3.re - u[33] * sloc.c2.c3.im;

  r.c4.c1.im[idx] =
      u[10] * sloc.c1.c1.im - u[11] * sloc.c1.c1.re + u[18] * sloc.c1.c2.im -
      u[19] * sloc.c1.c2.re + u[24] * sloc.c1.c3.im - u[25] * sloc.c1.c3.re +
      u[3] * sloc.c2.c1.im + mu * sloc.c2.c1.re + u[30] * sloc.c2.c2.im +
      u[31] * sloc.c2.c2.re + u[32] * sloc.c2.c3.im + u[33] * sloc.c2.c3.re;

  r.c4.c2.re[idx] =
      u[12] * sloc.c1.c1.re + u[13] * sloc.c1.c1.im + u[20] * sloc.c1.c2.re +
      u[21] * sloc.c1.c2.im + u[26] * sloc.c1.c3.re + u[27] * sloc.c1.c3.im +
      u[30] * sloc.c2.c1.re + u[31] * sloc.c2.c1.im + u[4] * sloc.c2.c2.re -
      mu * sloc.c2.c2.im + u[34] * sloc.c2.c3.re - u[35] * sloc.c2.c3.im;

  r.c4.c2.im[idx] =
      u[12] * sloc.c1.c1.im - u[13] * sloc.c1.c1.re + u[20] * sloc.c1.c2.im -
      u[21] * sloc.c1.c2.re + u[26] * sloc.c1.c3.im - u[27] * sloc.c1.c3.re +
      u[30] * sloc.c2.c1.im - u[31] * sloc.c2.c1.re + u[4] * sloc.c2.c2.im +
      mu * sloc.c2.c2.re + u[34] * sloc.c2.c3.im + u[35] * sloc.c2.c3.re;

  r.c4.c3.re[idx] =
      u[14] * sloc.c1.c1.re + u[15] * sloc.c1.c1.im + u[22] * sloc.c1.c2.re +
      u[23] * sloc.c1.c2.im + u[28] * sloc.c1.c3.re + u[29] * sloc.c1.c3.im +
      u[32] * sloc.c2.c1.re + u[33] * sloc.c2.c1.im + u[34] * sloc.c2.c2.re +
      u[35] * sloc.c2.c2.im + u[5] * sloc.c2.c3.re - mu * sloc.c2.c3.im;

  r.c4.c3.im[idx] =
      u[14] * sloc.c1.c1.im - u[15] * sloc.c1.c1.re + u[22] * sloc.c1.c2.im -
      u[23] * sloc.c1.c2.re + u[28] * sloc.c1.c3.im - u[29] * sloc.c1.c3.re +
      u[32] * sloc.c2.c1.im - u[33] * sloc.c2.c1.re + u[34] * sloc.c2.c2.im -
      u[35] * sloc.c2.c2.re + u[5] * sloc.c2.c3.im + mu * sloc.c2.c3.re;
}
// ---------------------------------------------------------------------------//

// ---------------------------------------------------------------------------//
extern "C" void doe_kernel(int vol, spinor_soa s, spinor_soa r, su3_soa u,
                           sycl::int4 *piup, sycl::int4 *pidn, float coe,
                           float gamma_f, float one_over_gammaf,
                           sycl::nd_item<1> item_ct1)
{
  int idx = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0);
  if (idx >= vol / 2)
    return;

  su3 uloc;
  spinor sloc, rloc;
  su3_vector psi, chi;
  sycl::int4 piuploc = piup[idx];
  sycl::int4 pidnloc = pidn[idx];
  int sidx, uidx;

  /***************************** direction +0 *******************************/

  sidx = piuploc.x();
  uidx = 0 * (vol / 2) + idx;
  _spinor_copy2struct(sloc, s, sidx);
  _su3_copy2struct(uloc, u, uidx);

  _vector_add(psi, sloc.c1, sloc.c3);
  _su3_multiply(rloc.c1, uloc, psi);
  rloc.c3 = rloc.c1;

  _vector_add(psi, sloc.c2, sloc.c4);
  _su3_multiply(rloc.c2, uloc, psi);
  rloc.c4 = rloc.c2;

  /***************************** direction -0 *******************************/

  sidx = pidnloc.x();
  uidx = 1 * (vol / 2) + idx;
  _spinor_copy2struct(sloc, s, sidx);
  _su3_copy2struct(uloc, u, uidx);

  _vector_sub(psi, sloc.c1, sloc.c3);
  _su3_inverse_multiply(chi, uloc, psi);
  _vector_add_assign(rloc.c1, chi);
  _vector_sub_assign(rloc.c3, chi);

  _vector_sub(psi, sloc.c2, sloc.c4);
  _su3_inverse_multiply(chi, uloc, psi);
  _vector_add_assign(rloc.c2, chi);
  _vector_sub_assign(rloc.c4, chi);

  _vector_mul_assign(rloc.c1, gamma_f);
  _vector_mul_assign(rloc.c2, gamma_f);
  _vector_mul_assign(rloc.c3, gamma_f);
  _vector_mul_assign(rloc.c4, gamma_f);

  /***************************** direction +1 *******************************/

  sidx = piuploc.y();
  uidx = 2 * (vol / 2) + idx;
  _spinor_copy2struct(sloc, s, sidx);
  _su3_copy2struct(uloc, u, uidx);

  _vector_i_add(psi, sloc.c1, sloc.c4);
  _su3_multiply(chi, uloc, psi);
  _vector_add_assign(rloc.c1, chi);
  _vector_i_sub_assign(rloc.c4, chi);

  _vector_i_add(psi, sloc.c2, sloc.c3);
  _su3_multiply(chi, uloc, psi);
  _vector_add_assign(rloc.c2, chi);
  _vector_i_sub_assign(rloc.c3, chi);

  /***************************** direction -1 *******************************/

  sidx = pidnloc.y();
  uidx = 3 * (vol / 2) + idx;
  _spinor_copy2struct(sloc, s, sidx);
  _su3_copy2struct(uloc, u, uidx);

  _vector_i_sub(psi, sloc.c1, sloc.c4);
  _su3_inverse_multiply(chi, uloc, psi);
  _vector_add_assign(rloc.c1, chi);
  _vector_i_add_assign(rloc.c4, chi);

  _vector_i_sub(psi, sloc.c2, sloc.c3);
  _su3_inverse_multiply(chi, uloc, psi);
  _vector_add_assign(rloc.c2, chi);
  _vector_i_add_assign(rloc.c3, chi);

  /***************************** direction +2 *******************************/

  sidx = piuploc.z();
  uidx = 4 * (vol / 2) + idx;
  _spinor_copy2struct(sloc, s, sidx);
  _su3_copy2struct(uloc, u, uidx);

  _vector_add(psi, sloc.c1, sloc.c4);
  _su3_multiply(chi, uloc, psi);
  _vector_add_assign(rloc.c1, chi);
  _vector_add_assign(rloc.c4, chi);

  _vector_sub(psi, sloc.c2, sloc.c3);
  _su3_multiply(chi, uloc, psi);
  _vector_add_assign(rloc.c2, chi);
  _vector_sub_assign(rloc.c3, chi);

  /***************************** direction -2 *******************************/

  sidx = pidnloc.z();
  uidx = 5 * (vol / 2) + idx;
  _spinor_copy2struct(sloc, s, sidx);
  _su3_copy2struct(uloc, u, uidx);

  _vector_sub(psi, sloc.c1, sloc.c4);
  _su3_inverse_multiply(chi, uloc, psi);
  _vector_add_assign(rloc.c1, chi);
  _vector_sub_assign(rloc.c4, chi);

  _vector_add(psi, sloc.c2, sloc.c3);
  _su3_inverse_multiply(chi, uloc, psi);
  _vector_add_assign(rloc.c2, chi);
  _vector_add_assign(rloc.c3, chi);

  /***************************** direction +3 *******************************/

  sidx = piuploc.w();
  uidx = 6 * (vol / 2) + idx;
  _spinor_copy2struct(sloc, s, sidx);
  _su3_copy2struct(uloc, u, uidx);

  _vector_i_add(psi, sloc.c1, sloc.c3);
  _su3_multiply(chi, uloc, psi);
  _vector_add_assign(rloc.c1, chi);
  _vector_i_sub_assign(rloc.c3, chi);

  _vector_i_sub(psi, sloc.c2, sloc.c4);
  _su3_multiply(chi, uloc, psi);
  _vector_add_assign(rloc.c2, chi);
  _vector_i_add_assign(rloc.c4, chi);

  /***************************** direction -3 *******************************/

  sidx = pidnloc.w();
  uidx = 7 * (vol / 2) + idx;
  _spinor_copy2struct(sloc, s, sidx);
  _su3_copy2struct(uloc, u, uidx);

  _vector_i_sub(psi, sloc.c1, sloc.c3);
  _su3_inverse_multiply(chi, uloc, psi);
  _vector_add_assign(rloc.c1, chi);
  _vector_i_add_assign(rloc.c3, chi);

  _vector_i_add(psi, sloc.c2, sloc.c4);
  _su3_inverse_multiply(chi, uloc, psi);
  _vector_add_assign(rloc.c2, chi);
  _vector_i_sub_assign(rloc.c4, chi);

  _vector_mul_assign(rloc.c1, coe);
  _vector_mul_assign(rloc.c2, coe);
  _vector_mul_assign(rloc.c3, coe);
  _vector_mul_assign(rloc.c4, coe);

  _vector_mul_assign(rloc.c1, one_over_gammaf);
  _vector_mul_assign(rloc.c2, one_over_gammaf);
  _vector_mul_assign(rloc.c3, one_over_gammaf);
  _vector_mul_assign(rloc.c4, one_over_gammaf);

  // Copy from registers to global memory
  sidx = vol / 2 + idx;
  _spinor_add2arrays(r, rloc, sidx);
}
// ---------------------------------------------------------------------------//

// ---------------------------------------------------------------------------//
extern "C" void deo_kernel(int vol, spinor_soa s, spinor_soa r, su3_soa u,
                           sycl::int4 *piup, sycl::int4 *pidn, float ceo,
                           float one_over_gammaf, sycl::nd_item<1> item_ct1)
{
  int idx = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0);
  if (idx >= vol / 2)
    return;

  su3 uloc;
  spinor sloc;
  su3_vector psi, chi;
  sycl::int4 piuploc = piup[idx];
  sycl::int4 pidnloc = pidn[idx];
  int sidx, uidx;

  sidx = vol / 2 + idx;
  _spinor_copy2struct(sloc, s, sidx);

  _vector_mul_assign(sloc.c1, ceo);
  _vector_mul_assign(sloc.c2, ceo);
  _vector_mul_assign(sloc.c3, ceo);
  _vector_mul_assign(sloc.c4, ceo);

  /***************************** direction +0 *******************************/

  sidx = piuploc.x();
  uidx = 0 * (vol / 2) + idx;
  _su3_copy2struct(uloc, u, uidx);

  _vector_sub(psi, sloc.c1, sloc.c3);
  _su3_inverse_multiply(chi, uloc, psi);
  vector_add_assign(r.c1, chi, sidx);
  vector_sub_assign(r.c3, chi, sidx);

  _vector_sub(psi, sloc.c2, sloc.c4);
  _su3_inverse_multiply(chi, uloc, psi);
  vector_add_assign(r.c2, chi, sidx);
  vector_sub_assign(r.c4, chi, sidx);

  /***************************** direction -0 *******************************/

  sidx = pidnloc.x();
  uidx = 1 * (vol / 2) + idx;
  _su3_copy2struct(uloc, u, uidx);

  _vector_add(psi, sloc.c1, sloc.c3);
  _su3_multiply(chi, uloc, psi);
  vector_add_assign(r.c1, chi, sidx);
  vector_add_assign(r.c3, chi, sidx);

  _vector_add(psi, sloc.c2, sloc.c4);
  _su3_multiply(chi, uloc, psi);
  vector_add_assign(r.c2, chi, sidx);
  vector_add_assign(r.c4, chi, sidx);

  /***************************** direction +1 *******************************/

  _vector_mul_assign(sloc.c1, one_over_gammaf);
  _vector_mul_assign(sloc.c2, one_over_gammaf);
  _vector_mul_assign(sloc.c3, one_over_gammaf);
  _vector_mul_assign(sloc.c4, one_over_gammaf);

  sidx = piuploc.y();
  uidx = 2 * (vol / 2) + idx;
  _su3_copy2struct(uloc, u, uidx);

  _vector_i_sub(psi, sloc.c1, sloc.c4);
  _su3_inverse_multiply(chi, uloc, psi);
  vector_add_assign(r.c1, chi, sidx);
  vector_i_add_assign(r.c4, chi, sidx);

  _vector_i_sub(psi, sloc.c2, sloc.c3);
  _su3_inverse_multiply(chi, uloc, psi);
  vector_add_assign(r.c2, chi, sidx);
  vector_i_add_assign(r.c3, chi, sidx);

  /***************************** direction -1 *******************************/

  sidx = pidnloc.y();
  uidx = 3 * (vol / 2) + idx;
  _su3_copy2struct(uloc, u, uidx);

  _vector_i_add(psi, sloc.c1, sloc.c4);
  _su3_multiply(chi, uloc, psi);
  vector_add_assign(r.c1, chi, sidx);
  vector_i_sub_assign(r.c4, chi, sidx);

  _vector_i_add(psi, sloc.c2, sloc.c3);
  _su3_multiply(chi, uloc, psi);
  vector_add_assign(r.c2, chi, sidx);
  vector_i_sub_assign(r.c3, chi, sidx);

  /***************************** direction +2 *******************************/

  sidx = piuploc.z();
  uidx = 4 * (vol / 2) + idx;
  _su3_copy2struct(uloc, u, uidx);

  _vector_sub(psi, sloc.c1, sloc.c4);
  _su3_inverse_multiply(chi, uloc, psi);
  vector_add_assign(r.c1, chi, sidx);
  vector_sub_assign(r.c4, chi, sidx);

  _vector_add(psi, sloc.c2, sloc.c3);
  _su3_inverse_multiply(chi, uloc, psi);
  vector_add_assign(r.c2, chi, sidx);
  vector_add_assign(r.c3, chi, sidx);

  /***************************** direction -2 *******************************/

  sidx = pidnloc.z();
  uidx = 5 * (vol / 2) + idx;
  _su3_copy2struct(uloc, u, uidx);

  _vector_add(psi, sloc.c1, sloc.c4);
  _su3_multiply(chi, uloc, psi);
  vector_add_assign(r.c1, chi, sidx);
  vector_add_assign(r.c4, chi, sidx);

  _vector_sub(psi, sloc.c2, sloc.c3);
  _su3_multiply(chi, uloc, psi);
  vector_add_assign(r.c2, chi, sidx);
  vector_sub_assign(r.c3, chi, sidx);

  /***************************** direction +3 *******************************/

  sidx = piuploc.w();
  uidx = 6 * (vol / 2) + idx;
  _su3_copy2struct(uloc, u, uidx);

  _vector_i_sub(psi, sloc.c1, sloc.c3);
  _su3_inverse_multiply(chi, uloc, psi);
  vector_add_assign(r.c1, chi, sidx);
  vector_i_add_assign(r.c3, chi, sidx);

  _vector_i_add(psi, sloc.c2, sloc.c4);
  _su3_inverse_multiply(chi, uloc, psi);
  vector_add_assign(r.c2, chi, sidx);
  vector_i_sub_assign(r.c4, chi, sidx);

  /***************************** direction -3 *******************************/

  sidx = pidnloc.w();
  uidx = 7 * (vol / 2) + idx;
  _su3_copy2struct(uloc, u, uidx);

  _vector_i_add(psi, sloc.c1, sloc.c3);
  _su3_multiply(chi, uloc, psi);
  vector_add_assign(r.c1, chi, sidx);
  vector_i_sub_assign(r.c3, chi, sidx);

  _vector_i_sub(psi, sloc.c2, sloc.c4);
  _su3_multiply(chi, uloc, psi);
  vector_add_assign(r.c2, chi, sidx);
  vector_i_add_assign(r.c4, chi, sidx);
}
// ---------------------------------------------------------------------------//

extern "C" void Dw_cuda_SoA(int VOLUME, su3 *u, spinor *s, spinor *r, pauli *m, int *piup, int *pidn)
{

  auto platformlist = sycl::platform::get_platforms();

  std::cout << "List of detected devices:" << "\n";

  for (auto p : platformlist)
  {
    auto devicelist = p.get_devices(sycl::info::device_type::all);
    for(auto d : devicelist)
      {
        std::string device_vendor = d.get_info<sycl::info::device::vendor>();
        std::cout<<d.get_info<sycl::info::device::name>()<<"\n";
      }
  }

  sycl::queue q_ct1{ sycl::default_selector{} };

  std::cout << "Selected device: " << q_ct1.get_device().get_info<sycl::info::device::name>() << "\n";

  sycl::event start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

  float milliseconds;

  int block_size, grid_size;

  float mu, coe, ceo;
  float gamma_f, one_over_gammaf;

  mu = 0.0f;
  coe = -0.5f;
  ceo = -0.5f;

  gamma_f = 1.0f;
  one_over_gammaf = 1.0f;

  // Copy pauli m from host to device and convert from Aos to SoA in GPU
  pauli_soa d_m_soa = allocPauli2Device(VOLUME, q_ct1); // Allocate SoA in device
  pauli * m_aos_usm = sycl::malloc_host<pauli>(2 * VOLUME * sizeof(pauli), q_ct1); // AoS_USM as host allocation, but accessible on the device via a PCI-e link
  std::memcpy(m_aos_usm, m, 2*VOLUME*sizeof(pauli)); // in the host side, copy the data pointed to by 'm' into m_aos_usm
  // pauli *d_m_aos;
  // d_m_aos = sycl::malloc_device<pauli>(2 * VOLUME, q_ct1); // Allocate AoS in device
  /*
  DPCT1012:2: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time
  is measured depending on your goals.
  */
  start_ct1 = std::chrono::steady_clock::now(); // Start the timer
  // q_ct1.memcpy(d_m_aos, m, 2 * VOLUME * sizeof(pauli)).wait(); // Mem copy AoS H2D
  block_size = 128;
  grid_size = ceil(VOLUME / static_cast<float>(block_size));
  /*
  DPCT1049:4: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  stop = q_ct1.parallel_for<class pauli_AoS2SoA_kernel>(sycl::nd_range<1>(sycl::range<1>(grid_size)
                                                        * sycl::range<1>(block_size), sycl::range<1>(block_size)),
                                                        [=](sycl::nd_item<1> item_ct1)
                                                        { pauli_AoS2SoA(VOLUME, d_m_soa, m_aos_usm, item_ct1); });
  /*
  DPCT1012:3: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time
  is measured depending on your goals.
  */
  stop.wait();
  stop_ct1 = std::chrono::steady_clock::now(); // Stop the timer
  milliseconds = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  printf("Time for AoS to SoA for pauli m +H2D (GPU) (ms): %.2f\n", milliseconds);
  // sycl::free(d_m_aos, q_ct1); // Free AoS in GPU
  sycl::free(m_aos_usm, q_ct1); // Free the AoS USM allocation

  // Copy su3 u from host to device and convert from Aos to SoA in GPU
  su3_soa d_u_soa = allocSu32Device(VOLUME, q_ct1); // Allocate SoA in device
  su3 *d_u_aos;
  d_u_aos = sycl::malloc_device<su3>(4 * VOLUME, q_ct1); // Allocate AoS in device
  /*
  DPCT1012:5: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time
  is measured depending on your goals.
  */
  start_ct1 = std::chrono::steady_clock::now();              // Start the timer
  q_ct1.memcpy(d_u_aos, u, 4 * VOLUME * sizeof(su3)).wait(); // Mem copy AoS H2D
  block_size = 128;
  grid_size = ceil((VOLUME / 2.0) / static_cast<float>(block_size));
  /*
  DPCT1049:7: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  stop = q_ct1.parallel_for<class su3_AoS2SoA_kernel>(sycl::nd_range<1>(sycl::range<1>(grid_size)
                                                      * sycl::range<1>(block_size), sycl::range<1>(block_size)),
                                                      [=](sycl::nd_item<1> item_ct1)
                                                      { su3_AoS2SoA(VOLUME, d_u_soa, d_u_aos, item_ct1); });
  /*
  DPCT1012:6: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time
  is measured depending on your goals.
  */
  stop.wait();
  stop_ct1 = std::chrono::steady_clock::now(); // Stop the timer
  milliseconds = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  printf("Time for AoS to SoA for su3 u +H2D (GPU) (ms): %.2f\n", milliseconds);
  sycl::free(d_u_aos, q_ct1); // Free AoS in GPU

  // Copy spinor s from host to device and convert from Aos to SoA in GPU
  spinor_soa d_s_soa = allocSpinor2Device(VOLUME, q_ct1); // Allocate SoA in device
  spinor *d_s_aos;
  d_s_aos = sycl::malloc_device<spinor>(VOLUME, q_ct1); // Allocate AoS in device
  /*
  DPCT1012:8: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time
  is measured depending on your goals.
  */
  start_ct1 = std::chrono::steady_clock::now();             // Start the timer
  q_ct1.memcpy(d_s_aos, s, VOLUME * sizeof(spinor)).wait(); // Mem copy AoS H2D
  block_size = 128;
  grid_size = ceil(VOLUME / static_cast<float>(block_size));
  /*
  DPCT1049:10: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  stop = q_ct1.parallel_for<class spinor_AoS2SoA_kernel>(sycl::nd_range<1>(sycl::range<1>(grid_size)
                                                         * sycl::range<1>(block_size), sycl::range<1>(block_size)),
                                                         [=](sycl::nd_item<1> item_ct1)
                                                         { spinor_AoS2SoA(VOLUME, d_s_soa, d_s_aos, item_ct1); });
  /*
  DPCT1012:9: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time
  is measured depending on your goals.
  */
  stop.wait();
  stop_ct1 = std::chrono::steady_clock::now(); // Stop the timer
  milliseconds = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  printf("Time for AoS to SoA for spinor s +H2D (GPU) (ms): %.2f\n", milliseconds);
  sycl::free(d_s_aos, q_ct1);

  // Allocate memory on device for lookup tables and spinor r
  sycl::int4 *d_piup, *d_pidn;
  d_piup = (sycl::int4 *)sycl::malloc_device(2 * VOLUME * sizeof(int), q_ct1);
  d_pidn = (sycl::int4 *)sycl::malloc_device(2 * VOLUME * sizeof(int), q_ct1);
  spinor_soa d_r_soa = allocSpinor2Device(VOLUME, q_ct1);

  // Copy lookup tables from host to device
  /*
  DPCT1012:11: Detected kernel execution time measurement pattern and
  generated an initial code for time measurements in SYCL. You can change the
  way time is measured depending on your goals.
  */
  start_ct1 = std::chrono::steady_clock::now();
  q_ct1.memcpy(d_piup, piup, 2 * VOLUME * sizeof(int));
  q_ct1.memcpy(d_pidn, pidn, 2 * VOLUME * sizeof(int)).wait();
  /*
  DPCT1012:12: Detected kernel execution time measurement pattern and
  generated an initial code for time measurements in SYCL. You can change the
  way time is measured depending on your goals.
  */
  stop_ct1 = std::chrono::steady_clock::now();
  milliseconds = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  printf("Time for cudaMemcpy H2D of lookup tables (ms): %.2f\n", milliseconds);

  // Launch kernels on GPU
  block_size = 128;
  grid_size = ceil(VOLUME / static_cast<float>(block_size));
  /*
  DPCT1012:13: Detected kernel execution time measurement pattern and
  generated an initial code for time measurements in SYCL. You can change the
  way time is measured depending on your goals.
  */
  start_ct1 = std::chrono::steady_clock::now();
  /*
  DPCT1049:15: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  stop = q_ct1.parallel_for<class my_mulpauli_kernel>(sycl::nd_range<1>(sycl::range<1>(grid_size)
                                                      * sycl::range<1>(block_size), sycl::range<1>(block_size)),
                                                      [=](sycl::nd_item<1> item_ct1)
                                                      {
                                                        mulpauli_kernel(VOLUME, mu, d_s_soa, d_r_soa,
                                                                        d_m_soa, item_ct1);
                                                      });
  /*
  DPCT1012:14: Detected kernel execution time measurement pattern and
  generated an initial code for time measurements in SYCL. You can change the
  way time is measured depending on your goals.
  */
  stop.wait();
  stop_ct1 = std::chrono::steady_clock::now();
  milliseconds = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  printf("Time for kernel mul_pauli (ms): %.2f\n", milliseconds);

  block_size = 128;
  grid_size = ceil((VOLUME / 2.0) / static_cast<float>(block_size));
  /*
  DPCT1012:16: Detected kernel execution time measurement pattern and
  generated an initial code for time measurements in SYCL. You can change the
  way time is measured depending on your goals.
  */
  start_ct1 = std::chrono::steady_clock::now();
  /*
  DPCT1049:18: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  stop = q_ct1.parallel_for<class my_doe_kernel>(sycl::nd_range<1>(sycl::range<1>(grid_size)
                                                 * sycl::range<1>(block_size), sycl::range<1>(block_size)),
                                                 [=](sycl::nd_item<1> item_ct1)
                                                 {
                                                  doe_kernel(VOLUME, d_s_soa, d_r_soa, d_u_soa, d_piup, d_pidn,
                                                             coe, gamma_f, one_over_gammaf, item_ct1);
                                                 });
  /*
  DPCT1012:17: Detected kernel execution time measurement pattern and
  generated an initial code for time measurements in SYCL. You can change the
  way time is measured depending on your goals.
  */
  stop.wait();
  stop_ct1 = std::chrono::steady_clock::now();
  milliseconds = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  printf("Time for kernel doe (ms): %.2f\n", milliseconds);

  block_size = 128;
  grid_size = ceil((VOLUME / 2.0) / static_cast<float>(block_size));
  /*
  DPCT1012:19: Detected kernel execution time measurement pattern and
  generated an initial code for time measurements in SYCL. You can change the
  way time is measured depending on your goals.
  */
  start_ct1 = std::chrono::steady_clock::now();
  /*
  DPCT1049:21: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  stop = q_ct1.parallel_for<class my_deo_kernel>(sycl::nd_range<1>(sycl::range<1>(grid_size) *
                                                 sycl::range<1>(block_size), sycl::range<1>(block_size)),
                                                 [=](sycl::nd_item<1> item_ct1)
                                                 { deo_kernel(VOLUME, d_s_soa, d_r_soa, d_u_soa, d_piup,
                                                              d_pidn, ceo, one_over_gammaf, item_ct1);
                                                 });
  /*
  DPCT1012:20: Detected kernel execution time measurement pattern and
  generated an initial code for time measurements in SYCL. You can change the
  way time is measured depending on your goals.
  */
  stop.wait();
  stop_ct1 = std::chrono::steady_clock::now();
  milliseconds = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  printf("Time for kernel deo (ms): %.2f\n", milliseconds);

  // Convert from SoA to AoS in GPU
  spinor *d_r_aos;
  d_r_aos = sycl::malloc_device<spinor>(VOLUME, q_ct1);
  block_size = 128;
  grid_size = ceil(VOLUME / static_cast<float>(block_size));
  /*
  DPCT1012:22: Detected kernel execution time measurement pattern and
  generated an initial code for time measurements in SYCL. You can change the
  way time is measured depending on your goals.
  */
  start_ct1 = std::chrono::steady_clock::now();
  /*
  DPCT1049:24: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  stop = q_ct1.parallel_for<class spinor_SoA2AoS_kernel>(sycl::nd_range<1>(sycl::range<1>(grid_size)
                                                         * sycl::range<1>(block_size), sycl::range<1>(block_size)),
                                                         [=](sycl::nd_item<1> item_ct1)
                                                         { spinor_SoA2AoS(VOLUME, d_r_aos, d_r_soa, item_ct1); });
  /*
  DPCT1012:23: Detected kernel execution time measurement pattern and
  generated an initial code for time measurements in SYCL. You can change the
  way time is measured depending on your goals.
  */
  stop.wait();
  stop_ct1 = std::chrono::steady_clock::now();
  milliseconds = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  printf("Time for SoA to AoS (GPU) (ms): %.2f\n", milliseconds);

  // Copy result back to the host
  /*
  DPCT1012:25: Detected kernel execution time measurement pattern and
  generated an initial code for time measurements in SYCL. You can change the
  way time is measured depending on your goals.
  */
  start_ct1 = std::chrono::steady_clock::now();
  q_ct1.memcpy(r, d_r_aos, VOLUME * sizeof(spinor)).wait();
  /*
  DPCT1012:26: Detected kernel execution time measurement pattern and
  generated an initial code for time measurements in SYCL. You can change the
  way time is measured depending on your goals.
  */
  stop_ct1 = std::chrono::steady_clock::now();
  milliseconds = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  printf("Time for cudaMemcpy D2H (ms): %.2f\n", milliseconds);

  // Free GPU memory
  destroy_pauli_soa(d_m_soa, q_ct1);
  destroy_su3_soa(d_u_soa, q_ct1);
  destroy_spinor_soa(d_s_soa, q_ct1);
  destroy_spinor_soa(d_r_soa, q_ct1);

  sycl::free(d_piup, q_ct1);
  sycl::free(d_pidn, q_ct1);
  sycl::free(d_r_aos, q_ct1);

}
