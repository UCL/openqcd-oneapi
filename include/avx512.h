
/*******************************************************************************
 *
 * File avx512.h
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Macros for operating on SU(3) vectors and matrices using Intel intrinsic
 * operations for AVX512 compatible processors
 *
 *******************************************************************************/

#ifndef AVX512_H
#define AVX512_H

#ifndef SSE2_H
#include "sse2.h"
#endif

#ifdef EMULATEAVX
#include "emmintrin.h"
#else
#include "immintrin.h"
#endif

/* Macros for single precision floating point numbers */

/* Load the upper of lower part of a 128 bit vector */
#define _avx512_load_64_dn(a, m) a = _mm_loadl_pi(a, (__m64 *)(m));
#define _avx512_load_64_up(a, m) a = _mm_loadh_pi(a, (__m64 *)(m));

/* Store the upper of lower part of a 128 bit vector */
#define _avx512_store_64_dn(a, m) _mm_storel_pi((__m64 *)(m), a);
#define _avx512_store_64_up(a, m) _mm_storeh_pi((__m64 *)(m), a);

/* Write two vectors from a 4 wide array  */
#define _avx512_write_2_f(w, v1, v2)                                           \
  {                                                                            \
    _avx512_store_64_dn(w, v1);                                                \
    _avx512_store_64_up(w, v2);                                                \
  }

/* Load 2 complex numbers into 128 bit vector  */
#define _avx512_load_2_f(w, v1, v2)                                            \
  {                                                                            \
    _avx512_load_64_dn(w, v1);                                                 \
    _avx512_load_64_up(w, v2);                                                 \
  }

/* Load a color from 4 half-spinors */
#define _avx512_load_4_color_f(r1, s1, s2, s3, s4)                             \
  {                                                                            \
    __m128 t128;                                                               \
    t128 = _mm_load_ps(s1);                                                    \
    _avx512_load_64_up(t128, s1 + 6);                                          \
    r1 = _mm512_castps128_ps512(t128);                                         \
    t128 = _mm_load_ps(s2);                                                    \
    _avx512_load_64_up(t128, s2 + 6);                                          \
    r1 = _mm512_insertf32x4(r1, t128, 1);                                      \
    t128 = _mm_load_ps(s3);                                                    \
    _avx512_load_64_up(t128, s3 + 6);                                          \
    r1 = _mm512_insertf32x4(r1, t128, 2);                                      \
    t128 = _mm_load_ps(s4);                                                    \
    _avx512_load_64_up(t128, s4 + 6);                                          \
    r1 = _mm512_insertf32x4(r1, t128, 3);                                      \
  }

/* Load a color from 4 half-spinors with dirac indeces reversed for s3 and s4 */
#define _avx512_load_4_color_f_reverse_up(r1, s1, s2, s3, s4)                  \
  {                                                                            \
    __m128 t128;                                                               \
    t128 = _mm_load_ps(s1);                                                    \
    _avx512_load_64_up(t128, s1 + 6);                                          \
    r1 = _mm512_castps128_ps512(t128);                                         \
    t128 = _mm_load_ps(s2);                                                    \
    _avx512_load_64_up(t128, s2 + 6);                                          \
    r1 = _mm512_insertf32x4(r1, t128, 1);                                      \
    t128 = _mm_load_ps(s3 + 6);                                                \
    _avx512_load_64_up(t128, s3);                                              \
    r1 = _mm512_insertf32x4(r1, t128, 2);                                      \
    t128 = _mm_load_ps(s4 + 6);                                                \
    _avx512_load_64_up(t128, s4);                                              \
    r1 = _mm512_insertf32x4(r1, t128, 3);                                      \
  }

/* Load a color from 4 half-spinors with dirac indeces reversed for s1 and s2 */
#define _avx512_load_4_color_f_reverse_dn(r1, s1, s2, s3, s4)                  \
  {                                                                            \
    __m128 t128;                                                               \
    t128 = _mm_load_ps(s1 + 6);                                                \
    _avx512_load_64_up(t128, s1);                                              \
    r1 = _mm512_castps128_ps512(t128);                                         \
    t128 = _mm_load_ps(s2 + 6);                                                \
    _avx512_load_64_up(t128, s2);                                              \
    r1 = _mm512_insertf32x4(r1, t128, 1);                                      \
    t128 = _mm_load_ps(s3);                                                    \
    _avx512_load_64_up(t128, s3 + 6);                                          \
    r1 = _mm512_insertf32x4(r1, t128, 2);                                      \
    t128 = _mm_load_ps(s4);                                                    \
    _avx512_load_64_up(t128, s4 + 6);                                          \
    r1 = _mm512_insertf32x4(r1, t128, 3);                                      \
  }

/* Load 4 half-spinors and organize colorwise into vectors r1, r2 and r3 */
#define _avx512_load_4_halfspinor_f(r1, r2, r3, s1, s2, s3, s4)                \
  {                                                                            \
    _avx512_load_4_color_f(r1, s1, s2, s3, s4);                                \
    _avx512_load_4_color_f(r2, s1 + 2, s2 + 2, s3 + 2, s4 + 2);                \
    _avx512_load_4_color_f(r3, s1 + 4, s2 + 4, s3 + 4, s4 + 4);                \
  }

/* Load 4 half-spinors reversing the second two spinor indeces and
 * organize colorwise into vectors r1, r2 and r3 */
#define _avx512_load_4_halfspinor_f_reverse_up(r1, r2, r3, s1, s2, s3, s4)     \
  {                                                                            \
    _avx512_load_4_color_f_reverse_up(r1, s1, s2, s3, s4);                     \
    _avx512_load_4_color_f_reverse_up(r2, s1 + 2, s2 + 2, s3 + 2, s4 + 2);     \
    _avx512_load_4_color_f_reverse_up(r3, s1 + 4, s2 + 4, s3 + 4, s4 + 4);     \
  }

/* Load 4 half-spinors reversing first two the spinor indeces and
 * organize colorwise into vectors r1, r2 and r3 */
#define _avx512_load_4_halfspinor_f_reverse_dn(r1, r2, r3, s1, s2, s3, s4)     \
  {                                                                            \
    _avx512_load_4_color_f_reverse_dn(r1, s1, s2, s3, s4);                     \
    _avx512_load_4_color_f_reverse_dn(r2, s1 + 2, s2 + 2, s3 + 2, s4 + 2);     \
    _avx512_load_4_color_f_reverse_dn(r3, s1 + 4, s2 + 4, s3 + 4, s4 + 4);     \
  }

/* Store a color to 4 half-spinors */
#define _avx512_write_4_color(r1, s1, s2, s3, s4)                              \
  {                                                                            \
    __m128 t128;                                                               \
    t128 = _mm512_castps512_ps128(r1);                                         \
    _avx512_store_64_dn(t128, s1);                                             \
    _avx512_store_64_up(t128, s1 + 6);                                         \
    t128 = _mm512_extractf32x4_ps(r1, 1);                                      \
    _avx512_store_64_dn(t128, s2);                                             \
    _avx512_store_64_up(t128, s2 + 6);                                         \
    t128 = _mm512_extractf32x4_ps(r1, 2);                                      \
    _avx512_store_64_dn(t128, s3);                                             \
    _avx512_store_64_up(t128, s3 + 6);                                         \
    t128 = _mm512_extractf32x4_ps(r1, 3);                                      \
    _avx512_store_64_dn(t128, s4);                                             \
    _avx512_store_64_up(t128, s4 + 6);                                         \
  }

/* Store a color to 4 half-spinors reversing the first two Dirac indeces */
#define _avx512_write_4_color_reverse_up(r1, s1, s2, s3, s4)                   \
  {                                                                            \
    __m128 t128;                                                               \
    t128 = _mm512_castps512_ps128(r1);                                         \
    _avx512_store_64_dn(t128, s1);                                             \
    _avx512_store_64_up(t128, s1 + 6);                                         \
    t128 = _mm512_extractf32x4_ps(r1, 1);                                      \
    _avx512_store_64_dn(t128, s2);                                             \
    _avx512_store_64_up(t128, s2 + 6);                                         \
    t128 = _mm512_extractf32x4_ps(r1, 2);                                      \
    _avx512_store_64_dn(t128, s3 + 6);                                         \
    _avx512_store_64_up(t128, s3);                                             \
    t128 = _mm512_extractf32x4_ps(r1, 3);                                      \
    _avx512_store_64_dn(t128, s4 + 6);                                         \
    _avx512_store_64_up(t128, s4);                                             \
  }

/* Store a color to 4 half-spinors reversing the second two Dirac indeces */
#define _avx512_write_4_color_reverse_dn(r1, s1, s2, s3, s4)                   \
  {                                                                            \
    __m128 t128;                                                               \
    t128 = _mm512_castps512_ps128(r1);                                         \
    _avx512_store_64_dn(t128, s1 + 6);                                         \
    _avx512_store_64_up(t128, s1);                                             \
    t128 = _mm512_extractf32x4_ps(r1, 1);                                      \
    _avx512_store_64_dn(t128, s2 + 6);                                         \
    _avx512_store_64_up(t128, s2);                                             \
    t128 = _mm512_extractf32x4_ps(r1, 2);                                      \
    _avx512_store_64_dn(t128, s3);                                             \
    _avx512_store_64_up(t128, s3 + 6);                                         \
    t128 = _mm512_extractf32x4_ps(r1, 3);                                      \
    _avx512_store_64_dn(t128, s4);                                             \
    _avx512_store_64_up(t128, s4 + 6);                                         \
  }

/* Store 4 half-spinors from color vectors */
#define _avx512_write_4_halfspinor_f(r1, r2, r3, s1, s2, s3, s4)               \
  {                                                                            \
    _avx512_write_4_color(r1, s1, s2, s3, s4);                                 \
    _avx512_write_4_color(r2, s1 + 2, s2 + 2, s3 + 2, s4 + 2);                 \
    _avx512_write_4_color(r3, s1 + 4, s2 + 4, s3 + 4, s4 + 4);                 \
  }

/* Store 4 half-spinors from color vectors reversing the first two Dirac indeces
 */
#define _avx512_write_4_halfspinor_f_reverse_up(r1, r2, r3, s1, s2, s3, s4)    \
  {                                                                            \
    _avx512_write_4_color_reverse_up(r1, s1, s2, s3, s4);                      \
    _avx512_write_4_color_reverse_up(r2, s1 + 2, s2 + 2, s3 + 2, s4 + 2);      \
    _avx512_write_4_color_reverse_up(r3, s1 + 4, s2 + 4, s3 + 4, s4 + 4);      \
  }

/* Store 4 half-spinors from color vectors reversing the second two Dirac
 * indeces */
#define _avx512_write_4_halfspinor_f_reverse_dn(r1, r2, r3, s1, s2, s3, s4)    \
  {                                                                            \
    _avx512_write_4_color_reverse_dn(r1, s1, s2, s3, s4);                      \
    _avx512_write_4_color_reverse_dn(r2, s1 + 2, s2 + 2, s3 + 2, s4 + 2);      \
    _avx512_write_4_color_reverse_dn(r3, s1 + 4, s2 + 4, s3 + 4, s4 + 4);      \
  }

/* Load 4 complex numbers, each broadcasted to 2 neighbouring memory locations
 */
#define _avx512_load_4_2_f(s512, fp1, fp2, fp3, fp4)                           \
  {                                                                            \
    __m128 t128;                                                               \
    t128 = _mm_load_ps(fp1);                                                   \
    t128 = _mm_permute_ps(t128, 0b01000100);                                   \
    s512 = _mm512_castps128_ps512(t128);                                       \
    t128 = _mm_load_ps(fp2);                                                   \
    t128 = _mm_permute_ps(t128, 0b01000100);                                   \
    s512 = _mm512_insertf32x4(s512, t128, 1);                                  \
    t128 = _mm_load_ps(fp3);                                                   \
    t128 = _mm_permute_ps(t128, 0b01000100);                                   \
    s512 = _mm512_insertf32x4(s512, t128, 2);                                  \
    t128 = _mm_load_ps(fp4);                                                   \
    t128 = _mm_permute_ps(t128, 0b01000100);                                   \
    s512 = _mm512_insertf32x4(s512, t128, 3);                                  \
  }

/* Load 8 numbers, each broadcasted to 2 neighbouring memory locations  */
#define _avx512_load_8_2(u512, fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8)         \
  {                                                                            \
    __m128 t1, t2;                                                             \
    t1 = _mm_load_ps1(fp1);                                                    \
    t2 = _mm_load_ps1(fp2);                                                    \
    t1 = _mm_blend_ps(t1, t2, 0b1100);                                         \
    u512 = _mm512_castps128_ps512(t1);                                         \
    t1 = _mm_load_ps1(fp3);                                                    \
    t2 = _mm_load_ps1(fp4);                                                    \
    t1 = _mm_blend_ps(t1, t2, 0b1100);                                         \
    u512 = _mm512_insertf32x4(u512, t1, 1);                                    \
    t1 = _mm_load_ps1(fp5);                                                    \
    t2 = _mm_load_ps1(fp6);                                                    \
    t1 = _mm_blend_ps(t1, t2, 0b1100);                                         \
    u512 = _mm512_insertf32x4(u512, t1, 2);                                    \
    t1 = _mm_load_ps1(fp7);                                                    \
    t2 = _mm_load_ps1(fp8);                                                    \
    t1 = _mm_blend_ps(t1, t2, 0b1100);                                         \
    u512 = _mm512_insertf32x4(u512, t1, 3);                                    \
  }

/* Combine Dirac spinors to half-spinors in deo and doe.
 */
static const int avx512_permute_im_1[] = {0, 1, 2,  3,  4,  5,  6,  7,
                                          9, 8, 11, 10, 13, 12, 15, 14};
#define _avx512_dirac_combine_f_1(a, b)                                        \
  {                                                                            \
    __m512i indexes = _mm512_load_epi32(avx512_permute_im_1);                  \
    __m512 c = _mm512_permutexvar_ps(indexes, b);                              \
    a = _mm512_mask_add_ps(a, 0b0101101000001111, a, c);                       \
    a = _mm512_mask_sub_ps(a, 0b1010010111110000, a, c);                       \
  }

#define _avx512_dirac_combine_f_2(a, b)                                        \
  {                                                                            \
    __m512i indexes = _mm512_load_epi32(avx512_permute_im_1);                  \
    __m512 c = _mm512_permutexvar_ps(indexes, b);                              \
    a = _mm512_mask_add_ps(a, 0b1001011011000011, a, c);                       \
    a = _mm512_mask_sub_ps(a, 0b0110100100111100, a, c);                       \
  }

#define _avx512_dirac_combine_f_3(a, b)                                        \
  {                                                                            \
    __m512i indexes = _mm512_load_epi32(avx512_permute_im_1);                  \
    __m512 c = _mm512_permutexvar_ps(indexes, b);                              \
    a = _mm512_mask_add_ps(a, 0b1010010100001111, a, c);                       \
    a = _mm512_mask_sub_ps(a, 0b0101101011110000, a, c);                       \
  }

#define _avx512_dirac_combine_f_4(a, b)                                        \
  {                                                                            \
    __m512i indexes = _mm512_load_epi32(avx512_permute_im_1);                  \
    __m512 c = _mm512_permutexvar_ps(indexes, b);                              \
    a = _mm512_mask_add_ps(a, 0b0110100111000011, a, c);                       \
    a = _mm512_mask_sub_ps(a, 0b1001011000111100, a, c);                       \
  }

/* Load and broadcast 4 floats, each broadcasted into 4 consecutive memory slots
 */
#define _avx512_load_4_f(d1, d2, d3, d4, c)                                    \
  {                                                                            \
    __m128 t128 = _mm_load_ps1(&d1);                                           \
    c = _mm512_castps128_ps512(t128);                                          \
    t128 = _mm_load_ps1(&d2);                                                  \
    c = _mm512_insertf32x4(c, t128, 1);                                        \
    t128 = _mm_load_ps1(&d3);                                                  \
    c = _mm512_insertf32x4(c, t128, 2);                                        \
    t128 = _mm_load_ps1(&d4);                                                  \
    c = _mm512_insertf32x4(c, t128, 3);                                        \
  }

/* Multiply the 4 half-dirac vectors in a 512 bit array with different numbers
 */
#define _avx512_16_d_mul(d1, d2, d3, d4, b, a)                                 \
  {                                                                            \
    __m512 c;                                                                  \
    _avx512_load_4_f(d1, d2, d3, d4, c);                                       \
    b = _mm512_mul_ps(a, c);                                                   \
  }

/* Multiply the 4 half-dirac vectors in a 512 bit array with different numbers
   and accumulate
 */
#define _avx512_16_d_mul_add(d1, d2, d3, d4, b, a)                             \
  {                                                                            \
    __m512 c;                                                                  \
    _avx512_load_4_f(d1, d2, d3, d4, c);                                       \
    b = _mm512_fmadd_ps(a, c, b);                                              \
  }
/* Multiply the 4 half-dirac vectors in a 512 bit array with different imaginary
   numbers and accumulate
 */
#define _avx512_16_d_mul_im_add(d1, d2, d3, d4, b, a)                          \
  {                                                                            \
    __m512 c;                                                                  \
    _avx512_load_4_f(d1, d2, d3, d4, c);                                       \
    c = _mm512_mul_ps(a, c);                                                   \
    b = _mm512_mask_add_ps(b, 0b0101101001011010, b, c);                       \
    b = _mm512_mask_sub_ps(b, 0b1010010110100101, b, c);                       \
  }

/* Multiply 4 vectors with a su(3) matrices, taking the inverse of every
 * second matrix
 */
static const int avx512_permute_im[] = {1, 0, 3,  2,  5,  4,  7,  6,
                                        9, 8, 11, 10, 13, 12, 15, 14};
#define avx512_su3_mixed_multiply_8(u1, um1, u2, um2, b1, b2, b3, a1, a2, a3)  \
  {                                                                            \
    _avx512_16_d_mul((u1).c11.re, (um1).c11.re, (u2).c11.re, (um2).c11.re, b1, \
                     a1);                                                      \
    _avx512_16_d_mul_add((u1).c12.re, (um1).c21.re, (u2).c12.re, (um2).c21.re, \
                         b1, a2);                                              \
    _avx512_16_d_mul((u1).c21.re, (um1).c12.re, (u2).c21.re, (um2).c12.re, b2, \
                     a1);                                                      \
    _avx512_16_d_mul_add((u1).c22.re, (um1).c22.re, (u2).c22.re, (um2).c22.re, \
                         b2, a2);                                              \
    _avx512_16_d_mul((u1).c31.re, (um1).c13.re, (u2).c31.re, (um2).c13.re, b3, \
                     a1);                                                      \
    _avx512_16_d_mul_add((u1).c32.re, (um1).c23.re, (u2).c32.re, (um2).c23.re, \
                         b3, a2);                                              \
                                                                               \
    __m512i permute_index = _mm512_load_epi32(avx512_permute_im);              \
    __m512 t1 = _mm512_permutexvar_ps(permute_index, a1);                      \
    _avx512_16_d_mul_im_add((u1).c11.im, (um1).c11.im, (u2).c11.im,            \
                            (um2).c11.im, b1, t1);                             \
    _avx512_16_d_mul_add((u1).c13.re, (um1).c31.re, (u2).c13.re, (um2).c31.re, \
                         b1, a3);                                              \
    _avx512_16_d_mul_im_add((u1).c21.im, (um1).c12.im, (u2).c21.im,            \
                            (um2).c12.im, b2, t1);                             \
    _avx512_16_d_mul_add((u1).c23.re, (um1).c32.re, (u2).c23.re, (um2).c32.re, \
                         b2, a3);                                              \
    _avx512_16_d_mul_im_add((u1).c31.im, (um1).c13.im, (u2).c31.im,            \
                            (um2).c13.im, b3, t1);                             \
    _avx512_16_d_mul_add((u1).c33.re, (um1).c33.re, (u2).c33.re, (um2).c33.re, \
                         b3, a3);                                              \
                                                                               \
    __m512 t2 = _mm512_permutexvar_ps(permute_index, a2);                      \
    __m512 t3 = _mm512_permutexvar_ps(permute_index, a3);                      \
    _avx512_16_d_mul_im_add((u1).c12.im, (um1).c21.im, (u2).c12.im,            \
                            (um2).c21.im, b1, t2);                             \
    _avx512_16_d_mul_im_add((u1).c13.im, (um1).c31.im, (u2).c13.im,            \
                            (um2).c31.im, b1, t3);                             \
    _avx512_16_d_mul_im_add((u1).c22.im, (um1).c22.im, (u2).c22.im,            \
                            (um2).c22.im, b2, t2);                             \
    _avx512_16_d_mul_im_add((u1).c23.im, (um1).c32.im, (u2).c23.im,            \
                            (um2).c32.im, b2, t3);                             \
    _avx512_16_d_mul_im_add((u1).c32.im, (um1).c23.im, (u2).c32.im,            \
                            (um2).c23.im, b3, t2);                             \
    _avx512_16_d_mul_im_add((u1).c33.im, (um1).c33.im, (u2).c33.im,            \
                            (um2).c33.im, b3, t3);                             \
  }

/*  Accumulate elements of Dirac vectors into a Weyl vector in deo and doe */
#define _avx512_to_weyl_f_1(w, b, gamma_f)                                     \
  {                                                                            \
    register __m128 t2 = _mm512_extractf32x4_ps((b), 1);                       \
    register __m128 t1 = _mm512_castps512_ps128((b));                          \
    register __m128 gamma = _mm_load_ps1(&gamma_f);                            \
    t1 = _mm_add_ps(t1, t2);                                                   \
    (w) = _mm_mul_ps(t1, gamma);                                               \
    t2 = _mm512_extractf32x4_ps((b), 3);                                       \
    t1 = _mm512_extractf32x4_ps((b), 2);                                       \
    t1 = _mm_add_ps(t1, t2);                                                   \
    (w) = _mm_add_ps((w), t1);                                                 \
  }

#define _avx512_to_weyl_f_2(w, b, gamma_f)                                     \
  {                                                                            \
    register __m128 t2 = _mm512_extractf32x4_ps((b), 1);                       \
    register __m128 t1 = _mm512_castps512_ps128((b));                          \
    register __m128 gamma = _mm_load_ps1(&gamma_f);                            \
    t1 = _mm_sub_ps(t1, t2);                                                   \
    (w) = _mm_mul_ps(t1, gamma);                                               \
    t2 = _mm512_extractf32x4_ps((b), 3);                                       \
    t1 = _mm512_extractf32x4_ps((b), 2);                                       \
    t1 = _mm_sub_ps(t2, t1);                                                   \
    t1 = _mm_permute_ps(t1, 0b00011011);                                       \
    (w) = _mm_addsub_ps((w), t1);                                              \
  }

#define _avx512_to_weyl_f_3(w, b)                                              \
  {                                                                            \
    register __m128 t1 = _mm512_castps512_ps128((b));                          \
    register __m128 t2 = _mm512_extractf32x4_ps((b), 1);                       \
    t1 = _mm_add_ps(t1, t2);                                                   \
    (w) = _mm_add_ps((w), t1);                                                 \
    t2 = _mm512_extractf32x4_ps((b), 3);                                       \
    t1 = _mm512_extractf32x4_ps((b), 2);                                       \
    t1 = _mm_add_ps(t1, t2);                                                   \
    (w) = _mm_add_ps((w), t1);                                                 \
  }

#define _avx512_to_weyl_f_4(w, b)                                              \
  {                                                                            \
    register __m128 t1 = _mm512_castps512_ps128((b));                          \
    register __m128 t2 = _mm512_extractf32x4_ps((b), 1);                       \
    t1 = _mm_sub_ps(t2, t1);                                                   \
    t1 = _mm_permute_ps(t1, 0b11011000);                                       \
    t2 = _mm_permute_ps((w), 0b01110010);                                      \
    t1 = _mm_addsub_ps(t2, t1);                                                \
    (w) = _mm_permute_ps(t1, 0b10001101);                                      \
    t2 = _mm512_extractf32x4_ps((b), 3);                                       \
    t1 = _mm512_extractf32x4_ps((b), 2);                                       \
    t1 = _mm_sub_ps(t2, t1);                                                   \
    t1 = _mm_permute_ps(t1, 0b11100001);                                       \
    t2 = _mm_permute_ps((w), 0b10110100);                                      \
    t1 = _mm_addsub_ps(t2, t1);                                                \
    (w) = _mm_permute_ps(t1, 0b10110100);                                      \
  }

/* Expand a Weyl vector into a Dirac vector in deo and doe */
#define _avx512_to_dirac_f_1(a1, w1, w2)                                       \
  {                                                                            \
    register __m128 t1, t2;                                                    \
    t1 = _mm_add_ps(w1, w2);                                                   \
    t2 = _mm_sub_ps(w1, w2);                                                   \
    a1 = _mm512_castps128_ps512(t1);                                           \
    a1 = _mm512_insertf32x4(a1, t2, 1);                                        \
  }

#define _avx512_to_dirac_f_2(a1, w1, w2)                                       \
  {                                                                            \
    register __m128 t1, t2;                                                    \
    t2 = _mm_permute_ps(w2, 0b00011011);                                       \
    t1 = _mm_addsub_ps(w1, t2);                                                \
    a1 = _mm512_insertf32x4(a1, t1, 2);                                        \
    t2 = _mm_permute_ps(t2, 0b10110001);                                       \
    t1 = _mm_permute_ps(w1, 0b10110001);                                       \
    t2 = _mm_addsub_ps(t1, t2);                                                \
    t2 = _mm_permute_ps(t2, 0b10110001);                                       \
    a1 = _mm512_insertf32x4(a1, t2, 3);                                        \
  }

#define _avx512_to_dirac_f_3(a1, w1, w2)                                       \
  {                                                                            \
    register __m128 t1, t2;                                                    \
    t2 = _mm_permute_ps(w2, 0b11011000);                                       \
    t1 = _mm_permute_ps(w1, 0b01110010);                                       \
    t1 = _mm_addsub_ps(t1, t2);                                                \
    t1 = _mm_permute_ps(t1, 0b10001101);                                       \
    a1 = _mm512_castps128_ps512(t1);                                           \
                                                                               \
    t2 = _mm_permute_ps(t2, 0b10110001);                                       \
    t1 = _mm_permute_ps(w1, 0b11011000);                                       \
    t2 = _mm_addsub_ps(t1, t2);                                                \
    t2 = _mm_permute_ps(t2, 0b11011000);                                       \
    a1 = _mm512_insertf32x4(a1, t2, 1);                                        \
  }

#define _avx512_to_dirac_f_4(a1, w1, w2)                                       \
  {                                                                            \
    register __m128 t1, t2, t3;                                                \
    t2 = _mm_permute_ps(w2, 0b11100001);                                       \
    t1 = _mm_permute_ps(w1, 0b10110100);                                       \
    t3 = _mm_addsub_ps(t1, t2);                                                \
    t3 = _mm_permute_ps(t3, 0b10110100);                                       \
    a1 = _mm512_insertf32x4(a1, t3, 2);                                        \
                                                                               \
    t2 = _mm_permute_ps(t2, 0b10110001);                                       \
    t1 = _mm_permute_ps(t1, 0b10110001);                                       \
    t3 = _mm_addsub_ps(t1, t2);                                                \
    t3 = _mm_permute_ps(t3, 0b11100001);                                       \
    a1 = _mm512_insertf32x4(a1, t3, 3);                                        \
  }

/* Macros for double precision numbers */

/* Load 2 half-spinors and organize colorwise into vectors s1, s2 and s3 */
#define _avx512_load_2_halfspinor_d(s1, s2, s3, sp, sm)                        \
  {                                                                            \
    register __m256d _t1, _t2, _t3, _t4, _t5, _t6, up, dn;                     \
    register __m512d t512 = _mm512_load_pd(sm);                                \
    _t1 = _mm512_castpd512_pd256(t512);                                        \
    _t2 = _mm512_extractf64x4_pd(t512, 1);                                     \
    _t3 = _mm256_load_pd(sm + 8);                                              \
    t512 = _mm512_load_pd(sp);                                                 \
    _t4 = _mm512_castpd512_pd256(t512);                                        \
    _t5 = _mm512_extractf64x4_pd(t512, 1);                                     \
    _t6 = _mm256_load_pd(sp + 8);                                              \
                                                                               \
    up = _mm256_blend_pd(_t1, _t2, 0b1100);                                    \
    dn = _mm256_blend_pd(_t4, _t5, 0b1100);                                    \
    s1 = _mm512_castpd256_pd512(dn);                                           \
    s1 = _mm512_insertf64x4(s1, up, 1);                                        \
                                                                               \
    up = _mm256_permute2f128_pd(_t3, _t1, 0b00000011);                         \
    dn = _mm256_permute2f128_pd(_t6, _t4, 0b00000011);                         \
    s2 = _mm512_castpd256_pd512(dn);                                           \
    s2 = _mm512_insertf64x4(s2, up, 1);                                        \
                                                                               \
    up = _mm256_blend_pd(_t2, _t3, 0b1100);                                    \
    dn = _mm256_blend_pd(_t5, _t6, 0b1100);                                    \
    s3 = _mm512_castpd256_pd512(dn);                                           \
    s3 = _mm512_insertf64x4(s3, up, 1);                                        \
  }

/* Load 2 half-spinors reversing the spinor indeces and
 * organize colorwise into vectors s1, s2 and s3 */
#define _avx512_load_2_halfspinor_d_reverse(s1, s2, s3, sp, sm)                \
  {                                                                            \
    register __m256d _t1, _t2, _t3, _t4, _t5, _t6, up, dn;                     \
    register __m512d t512 = _mm512_load_pd(sm);                                \
    _t1 = _mm512_castpd512_pd256(t512);                                        \
    _t2 = _mm512_extractf64x4_pd(t512, 1);                                     \
    _t3 = _mm256_load_pd(sm + 8);                                              \
    t512 = _mm512_load_pd(sp);                                                 \
    _t4 = _mm512_castpd512_pd256(t512);                                        \
    _t5 = _mm512_extractf64x4_pd(t512, 1);                                     \
    _t6 = _mm256_load_pd(sp + 8);                                              \
                                                                               \
    up = _mm256_permute2f128_pd(_t1, _t2, 0b00000011);                         \
    dn = _mm256_permute2f128_pd(_t4, _t5, 0b00000011);                         \
    s1 = _mm512_castpd256_pd512(dn);                                           \
    s1 = _mm512_insertf64x4(s1, up, 1);                                        \
                                                                               \
    up = _mm256_blend_pd(_t3, _t1, 0b1100);                                    \
    dn = _mm256_blend_pd(_t6, _t4, 0b1100);                                    \
    s2 = _mm512_castpd256_pd512(dn);                                           \
    s2 = _mm512_insertf64x4(s2, up, 1);                                        \
                                                                               \
    up = _mm256_permute2f128_pd(_t2, _t3, 0b00000011);                         \
    dn = _mm256_permute2f128_pd(_t5, _t6, 0b00000011);                         \
    s3 = _mm512_castpd256_pd512(dn);                                           \
    s3 = _mm512_insertf64x4(s3, up, 1);                                        \
  }

/* Write 2 half-spinors from three color vectors */
#define _avx512_store_2_halfspinor_d(s1, s2, s3, sp, sm)                       \
  {                                                                            \
    register __m256d t1 = _mm512_castpd512_pd256(s1);                          \
    register __m256d t2 = _mm512_castpd512_pd256(s2);                          \
    register __m256d t3 = _mm512_castpd512_pd256(s3);                          \
    register __m256d l = _mm256_permute2f128_pd(t1, t2, 0b00100000);           \
    _mm256_store_pd(sp, l);                                                    \
    l = _mm256_permute2f128_pd(t3, t1, 0b00110000);                            \
    _mm256_store_pd(sp + 4, l);                                                \
    l = _mm256_permute2f128_pd(t2, t3, 0b00110001);                            \
    _mm256_store_pd(sp + 8, l);                                                \
                                                                               \
    t1 = _mm512_extractf64x4_pd(s1, 1);                                        \
    t2 = _mm512_extractf64x4_pd(s2, 1);                                        \
    t3 = _mm512_extractf64x4_pd(s3, 1);                                        \
    l = _mm256_permute2f128_pd(t1, t2, 0b00100000);                            \
    _mm256_store_pd(sm, l);                                                    \
    l = _mm256_permute2f128_pd(t3, t1, 0b00110000);                            \
    _mm256_store_pd(sm + 4, l);                                                \
    l = _mm256_permute2f128_pd(t2, t3, 0b00110001);                            \
    _mm256_store_pd(sm + 8, l);                                                \
  }

/* Load four complex numbers. loadu2 comprices two load operations. */
#define _avx512_load_4_d(r, v1, v2, v3, v4)                                    \
  {                                                                            \
    register __m256d up = _mm256_loadu2_m128d(&(v4).re, &(v3).re);             \
    register __m256d dn = _mm256_loadu2_m128d(&(v2).re, &(v1).re);             \
    r = _mm512_castpd256_pd512(dn);                                            \
    r = _mm512_insertf64x4(r, up, 1);                                          \
  }

/* Store four complex numbers */
#define _avx512_store_4_d(r, v1, v2, v3, v4)                                   \
  {                                                                            \
    register __m256d t1 = _mm512_extractf64x4_pd(r, 1);                        \
    _mm256_storeu2_m128d(&(v4).re, &(v3).re, t1);                              \
    t1 = _mm512_castpd512_pd256(r);                                            \
    _mm256_storeu2_m128d(&(v2).re, &(v1).re, t1);                              \
  }

/* Load 4 doubles each broadcasted to 2 neighbouring entries  */
#define _avx512_load_4_2_d(r, d1, d2, d3, d4)                                  \
  {                                                                            \
    register __m128d t128l = _mm_load_pd1(d1);                                 \
    register __m128d t128u = _mm_load_pd1(d2);                                 \
    register __m256d t256l = _mm256_castpd128_pd256(t128l);                    \
    t256l = _mm256_insertf128_pd(t256l, t128u, 1);                             \
    register __m128d t128l2 = _mm_load_pd1(d3);                                \
    register __m128d t128u2 = _mm_load_pd1(d4);                                \
    register __m256d t256u = _mm256_castpd128_pd256(t128l2);                   \
    t256u = _mm256_insertf128_pd(t256u, t128u2, 1);                            \
    r = _mm512_castpd256_pd512(t256l);                                         \
    r = _mm512_insertf64x4(r, t256u, 1);                                       \
  }

/* Load 1 double broadcasted four times and 2 broadcasted twice */
#define _avx512_load_1d2u_d(r, d1, d3, d4)                                     \
  {                                                                            \
    register __m256d t256l = _mm256_broadcast_sd(d1);                          \
    register __m128d t128l2 = _mm_load_pd1(d3);                                \
    register __m128d t128u2 = _mm_load_pd1(d4);                                \
    register __m256d t256u = _mm256_castpd128_pd256(t128l2);                   \
    t256u = _mm256_insertf128_pd(t256u, t128u2, 1);                            \
    r = _mm512_castpd256_pd512(t256l);                                         \
    r = _mm512_insertf64x4(r, t256u, 1);                                       \
  }

/* Load 2 broadcasted twice and 1 double broadcasted four times */
#define _avx512_load_2d1u_d(r, d1, d2, d3)                                     \
  {                                                                            \
    register __m128d t128l = _mm_load_pd1(d1);                                 \
    register __m128d t128u = _mm_load_pd1(d2);                                 \
    register __m256d t256l = _mm256_castpd128_pd256(t128l);                    \
    t256l = _mm256_insertf128_pd(t256l, t128u, 1);                             \
    register __m256d t256u = _mm256_broadcast_sd(d3);                          \
    r = _mm512_castpd256_pd512(t256l);                                         \
    r = _mm512_insertf64x4(r, t256u, 1);                                       \
  }

/* Load 2 doubles each broadcasted 4 times */
#define _avx512_expand_2_dble(r, d1, d2)                                       \
  {                                                                            \
    register __m256d a = _mm256_broadcast_sd(&(d1));                           \
    r = _mm512_castpd256_pd512(a);                                             \
    a = _mm256_broadcast_sd(&(d2));                                            \
    r = _mm512_insertf64x4(r, a, 1);                                           \
  }

/* Multiply the lower and upper halves of a 8-double wide vector with
 * different numbers
 */
#define _avx512_d_mul_dble(d1, d2, b, c)                                       \
  {                                                                            \
    register __m512d t1;                                                       \
    _avx512_expand_2_dble(t1, d1, d2);                                         \
    b = _mm512_mul_pd(t1, c);                                                  \
  }

/* Multiply the lower and upper halves of a 8-double wide vector with
 * different numbers and accumulate
 */
#define _avx512_d_mul_add_dble(d1, d2, b, c)                                   \
  {                                                                            \
    register __m512d t1;                                                       \
    _avx512_expand_2_dble(t1, d1, d2);                                         \
    b = _mm512_fmadd_pd(t1, c, b);                                             \
  }

/* Multiply the lower half with the imaginary number d1 and
 * the upper half by the conjugate on d2
 */
#define _avx512_d_mul_im_add_dble(d1, d2, b, c)                                \
  {                                                                            \
    register __m512d _t1;                                                      \
    _avx512_expand_2_dble(_t1, d1, d2);                                        \
    _t1 = _mm512_mul_pd(_t1, c);                                               \
    b = _mm512_mask_add_pd(b, 0b01011010, b, _t1);                             \
    b = _mm512_mask_sub_pd(b, 0b10100101, b, _t1);                             \
  }

/* Multiply the lower half of the color vectors distributed in c1, c2 and c3
 * by the su3 matrix u and the upper half by the conjugate of um
 * Store in b1, b2 and b3
 */
#define avx512_su3_mul_quad_dble(u, um, b1, b2, b3, c1, c2, c3)                \
  {                                                                            \
    register __m512d t1, t2, t3;                                               \
    _avx512_d_mul_dble((u).c11.re, (um).c11.re, b1, c1);                       \
    _avx512_d_mul_add_dble((u).c12.re, (um).c21.re, b1, c2);                   \
    _avx512_d_mul_dble((u).c21.re, (um).c12.re, b2, c1);                       \
    _avx512_d_mul_add_dble((u).c22.re, (um).c22.re, b2, c2);                   \
    _avx512_d_mul_dble((u).c31.re, (um).c13.re, b3, c1);                       \
    _avx512_d_mul_add_dble((u).c32.re, (um).c23.re, b3, c2);                   \
                                                                               \
    t1 = _mm512_permute_pd(c1, 0b01010101);                                    \
    _avx512_d_mul_im_add_dble((u).c11.im, (um).c11.im, b1, t1);                \
    _avx512_d_mul_add_dble((u).c13.re, (um).c31.re, b1, c3);                   \
    _avx512_d_mul_im_add_dble((u).c21.im, (um).c12.im, b2, t1);                \
    _avx512_d_mul_add_dble((u).c23.re, (um).c32.re, b2, c3);                   \
    _avx512_d_mul_im_add_dble((u).c31.im, (um).c13.im, b3, t1);                \
    _avx512_d_mul_add_dble((u).c33.re, (um).c33.re, b3, c3);                   \
                                                                               \
    t2 = _mm512_permute_pd(c2, 0b01010101);                                    \
    t3 = _mm512_permute_pd(c3, 0b01010101);                                    \
    _avx512_d_mul_im_add_dble((u).c12.im, (um).c21.im, b1, (t2));              \
    _avx512_d_mul_im_add_dble((u).c13.im, (um).c31.im, b1, (t3));              \
    _avx512_d_mul_im_add_dble((u).c22.im, (um).c22.im, b2, (t2));              \
    _avx512_d_mul_im_add_dble((u).c23.im, (um).c32.im, b2, (t3));              \
    _avx512_d_mul_im_add_dble((u).c32.im, (um).c23.im, b3, (t2));              \
    _avx512_d_mul_im_add_dble((u).c33.im, (um).c33.im, b3, (t3));              \
  }

/* Add the upper and lower halves and multiply by a constant
 */
#define _avx512_to_weyl_addmul(w, b, gamma)                                    \
  {                                                                            \
    register __m256d t2 = _mm512_extractf64x4_pd((b), 1);                      \
    register __m256d t1 = _mm512_castpd512_pd256((b));                         \
    t1 = _mm256_add_pd(t1, t2);                                                \
    (w) = _mm256_mul_pd(t1, gamma);                                            \
  }

/* Substract the upper and lower halves and multiply by a constant
 */
#define _avx512_to_weyl_submul(w, b, gamma)                                    \
  {                                                                            \
    register __m256d t2 = _mm512_extractf64x4_pd((b), 1);                      \
    register __m256d t1 = _mm512_castpd512_pd256((b));                         \
    t1 = _mm256_sub_pd(t1, t2);                                                \
    (w) = _mm256_mul_pd(t1, gamma);                                            \
  }

/* Add the upper and lower halves and accumulate
 */
#define _avx512_to_weyl_acc(w, b)                                              \
  {                                                                            \
    register __m256d t2 = _mm512_extractf64x4_pd((b), 1);                      \
    register __m256d t1 = _mm512_castpd512_pd256((b));                         \
    register __m256d t3 = _mm256_add_pd(t1, t2);                               \
    (w) = _mm256_add_pd(t3, (w));                                              \
  }

/* Combine spinor entries into 2 weyl vectors
   stored in high and low entries of a spinor
 */
#define _avx512_to_weyl_1(w, b, gamma)                                         \
  {                                                                            \
    (b) = _mm512_mul_pd(b, gamma);                                             \
    register __m256d bh = _mm512_extractf64x4_pd((b), 1);                      \
    register __m256d bl = _mm512_castpd512_pd256((b));                         \
    register __m256d wl = _mm256_add_pd(bl, bh);                               \
    register __m256d wh = _mm256_sub_pd(bl, bh);                               \
    (w) = _mm512_castpd256_pd512(wl);                                          \
    (w) = _mm512_insertf64x4((w), wh, 1);                                      \
  }

#define _avx512_to_weyl_2b(w, b)                                               \
  {                                                                            \
    register __m256d bh = _mm512_extractf64x4_pd((b), 1);                      \
    register __m256d bl = _mm512_castpd512_pd256((b));                         \
    register __m256d tl = _mm256_add_pd(bl, bh);                               \
    register __m256d th = _mm256_sub_pd(bh, bl);                               \
    th = _mm256_permute4x64_pd(th, 0b00011011);                                \
                                                                               \
    register __m256d wh = _mm512_extractf64x4_pd((w), 1);                      \
    register __m256d wl = _mm512_castpd512_pd256((w));                         \
    wl = _mm256_add_pd(wl, tl);                                                \
    wh = _mm256_addsub_pd(wh, th);                                             \
    w = _mm512_castpd256_pd512(wl);                                            \
    w = _mm512_insertf64x4(w, wh, 1);                                          \
  }

#define _avx512_to_weyl_2(w, b)                                                \
  {                                                                            \
    register __m256d bh = _mm512_extractf64x4_pd((b), 1);                      \
    register __m256d bl = _mm512_castpd512_pd256((b));                         \
    register __m256d tl = _mm256_add_pd(bl, bh);                               \
    register __m256d th = _mm256_sub_pd(bh, bl);                               \
    th = _mm256_permute4x64_pd(th, 0b00011011);                                \
                                                                               \
    register __m512d t = _mm512_castpd256_pd512(tl);                           \
    t = _mm512_insertf64x4(t, th, 1);                                          \
    w = _mm512_mask_add_pd(w, 0b10101111, w, t);                               \
    w = _mm512_mask_sub_pd(w, 0b01010000, w, t);                               \
  }

#define _avx512_to_weyl_3(w, b)                                                \
  {                                                                            \
    register __m256d bh = _mm512_extractf64x4_pd((b), 1);                      \
    register __m256d bl = _mm512_castpd512_pd256((b));                         \
    register __m256d tl = _mm256_add_pd(bl, bh);                               \
    register __m256d th = _mm256_sub_pd(bh, bl);                               \
    th = _mm256_permute4x64_pd(th, 0b01001110);                                \
                                                                               \
    register __m512d t = _mm512_castpd256_pd512(tl);                           \
    t = _mm512_insertf64x4(t, th, 1);                                          \
    w = _mm512_mask_add_pd(w, 0b00111111, w, t);                               \
    w = _mm512_mask_sub_pd(w, 0b11000000, w, t);                               \
  }

#define _avx512_to_weyl_4(w, b)                                                \
  {                                                                            \
    register __m256d bh = _mm512_extractf64x4_pd((b), 1);                      \
    register __m256d bl = _mm512_castpd512_pd256((b));                         \
    register __m256d tl = _mm256_add_pd(bl, bh);                               \
    register __m256d th = _mm256_sub_pd(bh, bl);                               \
    th = _mm256_permute_pd(th, 0b0101);                                        \
                                                                               \
    register __m512d t = _mm512_castpd256_pd512(tl);                           \
    t = _mm512_insertf64x4(t, th, 1);                                          \
    w = _mm512_mask_add_pd(w, 0b01101111, w, t);                               \
    w = _mm512_mask_sub_pd(w, 0b10010000, w, t);                               \
  }

/* Create a full Dirac vector by adding and subtracting the indeces of
 * a weyl vector */
#define _avx512_expand_weyl(a, w)                                              \
  {                                                                            \
    register __m256d wh = _mm512_extractf64x4_pd((w), 1);                      \
    register __m256d wl = _mm512_castpd512_pd256((w));                         \
    register __m256d tl = _mm256_add_pd(wl, wh);                               \
    register __m256d th = _mm256_sub_pd(wl, wh);                               \
    a = _mm512_castpd256_pd512(tl);                                            \
    a = _mm512_insertf64x4(a, th, 1);                                          \
  }

#define _avx512_expand_weyl_2(a, w)                                            \
  {                                                                            \
    register __m256d wh = _mm512_extractf64x4_pd((w), 1);                      \
    wh = _mm256_permute4x64_pd(wh, 0b00011011);                                \
    register __m512d t1 = _mm512_broadcast_f64x4(wh);                          \
    register __m256d wl = _mm512_castpd512_pd256((w));                         \
    a = _mm512_broadcast_f64x4(wl);                                            \
    a = _mm512_mask_add_pd(a, 0b01011010, a, t1);                              \
    a = _mm512_mask_sub_pd(a, 0b10100101, a, t1);                              \
  }

#define _avx512_expand_weyl_3(a, w)                                            \
  {                                                                            \
    register __m256d wh = _mm512_extractf64x4_pd((w), 1);                      \
    wh = _mm256_permute4x64_pd(wh, 0b01001110);                                \
    register __m512d t1 = _mm512_broadcast_f64x4(wh);                          \
    register __m256d wl = _mm512_castpd512_pd256((w));                         \
    a = _mm512_broadcast_f64x4(wl);                                            \
    a = _mm512_mask_add_pd(a, 0b11000011, a, t1);                              \
    a = _mm512_mask_sub_pd(a, 0b00111100, a, t1);                              \
  }

#define _avx512_expand_weyl_4(a, w)                                            \
  {                                                                            \
    register __m256d wh = _mm512_extractf64x4_pd((w), 1);                      \
    wh = _mm256_permute4x64_pd(wh, 0b10110001);                                \
    register __m512d t1 = _mm512_broadcast_f64x4(wh);                          \
    register __m256d wl = _mm512_castpd512_pd256((w));                         \
    a = _mm512_broadcast_f64x4(wl);                                            \
    a = _mm512_mask_add_pd(a, 0b10010110, a, t1);                              \
    a = _mm512_mask_sub_pd(a, 0b01101001, a, t1);                              \
  }

/* Add a vector to a spinor */
#define _avx512_add_to_spinors(b, v1, v2, v3, v4)                              \
  {                                                                            \
    register __m512d a;                                                        \
    _avx512_load_4_d(a, v1, v2, v3, v4);                                       \
    a = _mm512_add_pd(a, b);                                                   \
    _avx512_store_4_d(a, v1, v2, v3, v4);                                      \
  }

#define _avx512_add_to_spinors_2(b, v1, v2, v3, v4)                            \
  {                                                                            \
    register __m512d a;                                                        \
    _avx512_load_4_d(a, v1, v2, v3, v4);                                       \
    a = _mm512_mask_add_pd(a, 0b00001111, a, b);                               \
    a = _mm512_mask_sub_pd(a, 0b11110000, a, b);                               \
    _avx512_store_4_d(a, v1, v2, v3, v4);                                      \
  }

#define _avx512_add_to_spinors_3(b, v1, v2, v3, v4)                            \
  {                                                                            \
    register __m512d a;                                                        \
    _avx512_load_4_d(a, v1, v2, v3, v4);                                       \
    register __m512d t1 = _mm512_permutex_pd(b, 0b00011011);                   \
    a = _mm512_mask_add_pd(a, 0b10100101, a, t1);                              \
    a = _mm512_mask_sub_pd(a, 0b01011010, a, t1);                              \
    _avx512_store_4_d(a, v1, v2, v3, v4);                                      \
  }

#define _avx512_add_to_spinors_4(b, v1, v2, v3, v4)                            \
  {                                                                            \
    register __m512d a;                                                        \
    _avx512_load_4_d(a, v1, v2, v3, v4);                                       \
    a = _mm512_mask_add_pd(a, 0b11000011, a, b);                               \
    a = _mm512_mask_sub_pd(a, 0b00111100, a, b);                               \
    _avx512_store_4_d(a, v1, v2, v3, v4);                                      \
  }

#define _avx512_add_to_spinors_5(b, v1, v2, v3, v4)                            \
  {                                                                            \
    register __m512d a;                                                        \
    _avx512_load_4_d(a, v1, v2, v3, v4);                                       \
    register __m512d t1 = _mm512_permute_pd(b, 0b01010101);                    \
    a = _mm512_mask_add_pd(a, 0b01101001, a, t1);                              \
    a = _mm512_mask_sub_pd(a, 0b10010110, a, t1);                              \
    _avx512_store_4_d(a, v1, v2, v3, v4);                                      \
  }

#endif
