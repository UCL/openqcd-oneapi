#ifndef QPX_H
#define QPX_H
/*******************************************************************************
*
* File qpx.h
*
* Copyright (C) 2013 Dalibor Djukanovic
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Type definitions and macros for fast manipulation  of
* SU(3) matrices, SU(3) vectors and Dirac spinors exploiting the Quad FPU
* unit of BlueGene/Q
*
*******************************************************************************/

#ifdef PDToolkit
/* Needed for parsing with TAU */
typedef double vector4double[4];
#endif

static vector4double qpx_r1, qpx_r2, qpx_r3, qpx_r4, qpx_r5, qpx_r6, qpx_r7,
    qpx_r8, qpx_r9;
static vector4double qpx_r10, qpx_r11, qpx_r12, qpx_r13, qpx_r14, qpx_r15,
    qpx_r16, qpx_r17;
static vector4double vec_i = (vector4double){0, 1, 0, 1};
static vector4double vec_i_s = (vector4double){0, 1, 0, -1};
static vector4double sign0 = (vector4double){1., 1., 1., -1.};
static vector4double sign1 = (vector4double){1., 1., -1., -1.};
static vector4double sign2 = (vector4double){-1., -1., 1., 1.};

/* Operands for qvfperm QRT QRA QRB QRC
   QRC[msw]        double  QRC[12:14]  QRT
   0x4000 0000   = 2.0     000 = 0     QRA0
   0x4002 0000   = 2.25    001 = 1     QRA1
   0x4004 0000   = 2.50    010 = 2     QRA2
   0x4006 0000   = 2.75    011 = 3     QRA3
   0x4008 0000   = 3.00    100 = 4     QRB0
   0x400a 0000   = 3.25    101 = 5     QRB1
   0x400c 0000   = 3.50    110 = 6     QRB2
   0x400e 0000   = 3.75    111 = 7     QRB3
*/
static vector4double perm0011 = {2.000000, 2.000000, 2.250000,
                                 2.250000}; /* A0 A0 A1 A1 */
static vector4double perm2233 = {2.500000, 2.500000, 2.750000,
                                 2.750000}; /* A2 A2 A3 A3 */
static vector4double perm1 = {2.000000, 2.250000, 3.000000,
                              3.250000}; /* A0 A1 B0 B1 */
static vector4double perm2 = {2.500000, 2.750000, 3.500000,
                              3.750000}; /* A2 A3 B2 B3 */
static vector4double perml1 = {2.000000, 3.00000, 2.250000,
                               3.000000}; /* A0 B0 A1 B0 */
static vector4double perml2 = {2.500000, 3.00000, 2.750000,
                               3.000000}; /* A2 B0 A3 B0 */
static vector4double perm12 = {2.000000, 2.250000, 3.500000,
                               3.750000}; /* A0 A1 B2 B3 */
static vector4double perm21 = {3.500000, 3.750000, 2.000000,
                               2.250000}; /* B2 B3 A0 A1 */

/* Prefetch */

#define _qpx_prefetch_su3_dp(addr)                                             \
  __dcbt(((char *)((unsigned long int)(addr))));                               \
  __dcbt(((char *)((unsigned long int)(addr))) + 128);

#define _qpx_prefetch_spinor_dp(addr)                                          \
  __dcbt(((char *)((unsigned long int)(addr))));                               \
  __dcbt(((char *)((unsigned long int)(addr))) + 128);

#define _qpx_prefetch_su3_sp(addr)                                             \
  __dcbt(((char *)((unsigned long int)(addr))));

#define _qpx_prefetch_spinor_sp(addr)                                          \
  __dcbt(((char *)((unsigned long int)(addr))));

/* Load and Store

   Asssume 32 Byte alignment for double precision structures
   (spinor_dble, weyl_dble, and su3_dble) and 16 Byte alignment
   for single-precision structures (spinor, weyl, su3)

   Use vec_lda and vec_sta to raise exception (SIG 7)
   in case of incorrect alignment (if environment variable
   BG_MAXALIGNEXP is set to a small value, e.g. 1)
*/

/* Load first Weyl spinor = components (c1, c2) of Dirac spinor:
   psi11 <- c1.c1.re  c1.c1.im   c1.c2.re  c1.c2.im
   psi11 <- c1.c3.re  c1.c3.im   c2.c1.re  c2.c1.im
   psi11 <- c2.c2.re  c2.c2.im   c2.c3.re  c2.c3.im
*/
#define _qpx_load_w1(r, ps)                                                    \
  r##1 = vec_lda(0, &(((ps)->c1).c1.re));                                      \
  r##2 = vec_lda(0, &(((ps)->c1).c3.re));                                      \
  r##3 = vec_lda(0, &(((ps)->c2).c2.re));

/* Load second Weyl spinor = components (c3, c4) of Dirac spinor:
   psi11 <- c3.c1.re  c3.c1.im   c3.c2.re  c3.c2.im
   psi11 <- c3.c3.re  c3.c3.im   c4.c1.re  c4.c1.im
   psi11 <- c4.c2.re  c4.c2.im   c4.c3.re  c4.c3.im
*/
#define _qpx_load_w2(r, ps)                                                    \
  r##1 = vec_lda(0, &(((ps)->c3).c1.re));                                      \
  r##2 = vec_lda(0, &(((ps)->c3).c3.re));                                      \
  r##3 = vec_lda(0, &(((ps)->c4).c2.re));

#define _qpx_store_w1(r, ps)                                                   \
  vec_sta(r##1, 0, &(((ps)->c1).c1.re));                                       \
  vec_sta(r##2, 0, &(((ps)->c1).c3.re));                                       \
  vec_sta(r##3, 0, &(((ps)->c2).c2.re));

#define _qpx_store_w2(r, ps)                                                   \
  vec_sta(r##1, 0, &(((ps)->c3).c1.re));                                       \
  vec_sta(r##2, 0, &(((ps)->c3).c3.re));                                       \
  vec_sta(r##3, 0, &(((ps)->c4).c2.re));

/* Permutation for Dirac Operators
  res1 =  ( v2.2   v2.3   v3.0   v3.1 )
  res2 =  ( v3.2   v2.3   v1.0   v1.1 )
  res3 =  ( v1.2   v1.3   v2.0   v2.1 )
*/
#define _qpx_vec_x(res, v)                                                     \
  res##1 = vec_sldw(v##2, v##3, 2);                                            \
  res##2 = vec_sldw(v##3, v##1, 2);                                            \
  res##3 = vec_sldw(v##1, v##2, 2);

/********************** Math functions ********************/

/******************* res = va + vb ***********************/
#define _qpx_vec_add(res, va, vb)                                              \
  res##1 = vec_add(va##1, vb##1);                                              \
  res##2 = vec_add(va##2, vb##2);                                              \
  res##3 = vec_add(va##3, vb##3);

/*********************************************************
    res1 = va1 + vb1
    res2 = va2 + ( +vb2.0 +vb2.1 -vb2.2 -vb2.3 )
    res3 = va3 - vb3

    If the operands are
         va1 = ( psi_1 psi_1 )   vb1 = ( psi_4 psi_4 )
         va2 = ( psi_1 psi_2 )   vb2 = ( psi_4 psi_3 )
         va3 = ( psi_2 psi_2 )   vb3 = ( psi_3 psi_3 )
    then
         res1 = ( phi_1 phi_1 )
         res2 = ( phi_1 phi_2 )
         res3 = ( phi_2 phi_2 )
    where
         phi_1 = psi_1 + psi_4
         phi_2 = psi_2 - psi_3
    is the spinor combination for mu=+2 of eq. (A.12) of doc/dirac.pdf
*/
#define _qpx_vec_add_n(res, va, vb)                                            \
  res##1 = vec_add(va##1, vb##1);                                              \
  res##2 = vec_madd(vb##2, sign1, va##2);                                      \
  res##3 = vec_sub(va##3, vb##3);

/******************* res = va - vb ***********************/
#define _qpx_vec_sub(res, va, vb)                                              \
  res##1 = vec_sub(va##1, vb##1);                                              \
  res##2 = vec_sub(va##2, vb##2);                                              \
  res##3 = vec_sub(va##3, vb##3);

/*********************************************************
    res1 = va1 - vb1
    res2 = va2 + ( -vb2.0 -vb2.1 +vb2.2 +vb2.3 )
    res3 = va3 + vb3

    If the operands are
         va1 = ( psi_1 psi_1 )   vb1 = ( psi_4 psi_4 )
         va2 = ( psi_1 psi_2 )   vb2 = ( psi_4 psi_3 )
         va3 = ( psi_2 psi_2 )   vb3 = ( psi_3 psi_3 )
    then
         res1 = ( phi_1 phi_1 )
         res2 = ( phi_1 phi_2 )
         res3 = ( phi_2 phi_2 )
    where
         phi_1 = psi_1 - psi_4
         phi_2 = psi_2 + psi_3
    is the spinor combination for mu=-2 of eq. (A.13) of doc/dirac.pdf
*/
#define _qpx_vec_sub_n(res, va, vb)                                            \
  res##1 = vec_sub(va##1, vb##1);                                              \
  res##2 = vec_madd(sign2, vb##2, va##2);                                      \
  res##3 = vec_add(va##3, vb##3);

/******************* res = va - i vb **********************/
#define _qpx_vec_i_sub(res, va, vb)                                            \
  res##1 = vec_xxcpnmadd(vb##1, vec_i, va##1);                                 \
  res##2 = vec_xxcpnmadd(vb##2, vec_i, va##2);                                 \
  res##3 = vec_xxcpnmadd(vb##3, vec_i, va##3);

/*********************************************************
    res1 = va1 - i vb1
    res2 = va2 + ( -i vb2.0, -i vb2.1, +i vb2.2, +i vb2.3 )
    res3 = va3 + i vb3
*/
#define _qpx_vec_i_sub_n(res, va, vb)                                          \
  res##1 = vec_xxcpnmadd(vb##1, vec_i, va##1);                                 \
  res##2 = vec_xxcpnmadd(vb##2, vec_i_s, va##2);                               \
  res##3 = vec_xxnpmadd(vb##3, vec_i, va##3);

/******************* res = va + i vb **********************/
#define _qpx_vec_i_add(res, va, vb)                                            \
  res##1 = vec_xxnpmadd(vb##1, vec_i, va##1);                                  \
  res##2 = vec_xxnpmadd(vb##2, vec_i, va##2);                                  \
  res##3 = vec_xxnpmadd(vb##3, vec_i, va##3);

/*********************************************************
    res1 = va1 + i vb1
    res2 = va2 + ( +i vb2.0, +i vb2.1, -i vb2.2, -i vb2.3 )
    res3 = va3 - i vb3
*/
#define _qpx_vec_i_add_n(res, va, vb)                                          \
  res##1 = vec_xxnpmadd(vb##1, vec_i, va##1);                                  \
  res##2 = vec_xxnpmadd(vb##2, vec_i_s, va##2);                                \
  res##3 = vec_xxcpnmadd(vb##3, vec_i, va##3);

#define _qpx_su3_mul(res, u, psi)                                              \
  qpx_r1 = vec_ld2(0, &((u).c11.re));                                          \
  qpx_r2 = vec_ld2(0, &((u).c21.re));                                          \
  qpx_r3 = vec_ld2(0, &((u).c31.re));                                          \
  qpx_r4 = vec_ld2(0, &((u).c12.re));                                          \
  qpx_r5 = vec_ld2(0, &((u).c22.re));                                          \
  qpx_r6 = vec_ld2(0, &((u).c32.re));                                          \
  qpx_r7 = vec_ld2(0, &((u).c13.re));                                          \
  qpx_r8 = vec_ld2(0, &((u).c23.re));                                          \
  qpx_r9 = vec_ld2(0, &((u).c33.re));                                          \
  qpx_r10 = vec_perm(psi##1, psi##2, perm12);                                  \
  qpx_r11 = vec_sldw(psi##1, psi##3, 2);                                       \
  qpx_r12 = vec_perm(psi##2, psi##3, perm12);                                  \
  qpx_r13 = vec_xxnpmadd(qpx_r10, qpx_r1, vec_xmul(qpx_r1, qpx_r10));          \
  qpx_r14 =                                                                    \
      vec_xxnpmadd(qpx_r11, qpx_r4, vec_xmadd(qpx_r4, qpx_r11, qpx_r13));      \
  qpx_r15 =                                                                    \
      vec_xxnpmadd(qpx_r12, qpx_r7, vec_xmadd(qpx_r7, qpx_r12, qpx_r14));      \
  qpx_r13 = vec_xxnpmadd(qpx_r10, qpx_r2, vec_xmul(qpx_r2, qpx_r10));          \
  qpx_r14 =                                                                    \
      vec_xxnpmadd(qpx_r11, qpx_r5, vec_xmadd(qpx_r5, qpx_r11, qpx_r13));      \
  qpx_r16 =                                                                    \
      vec_xxnpmadd(qpx_r12, qpx_r8, vec_xmadd(qpx_r8, qpx_r12, qpx_r14));      \
  qpx_r13 = vec_xxnpmadd(qpx_r10, qpx_r3, vec_xmul(qpx_r3, qpx_r10));          \
  qpx_r14 =                                                                    \
      vec_xxnpmadd(qpx_r11, qpx_r6, vec_xmadd(qpx_r6, qpx_r11, qpx_r13));      \
  qpx_r17 =                                                                    \
      vec_xxnpmadd(qpx_r12, qpx_r9, vec_xmadd(qpx_r9, qpx_r12, qpx_r14));      \
  res##1 = vec_perm(qpx_r15, qpx_r16, perm1);                                  \
  res##2 = vec_perm(qpx_r17, qpx_r15, perm12);                                 \
  res##3 = vec_perm(qpx_r16, qpx_r17, perm2);

#define _qpx_su3_inv_mul(res, u, psi)                                          \
  qpx_r1 = vec_ld2(0, &((u).c11.re));                                          \
  qpx_r2 = vec_ld2(0, &((u).c12.re));                                          \
  qpx_r3 = vec_ld2(0, &((u).c13.re));                                          \
  qpx_r4 = vec_ld2(0, &((u).c21.re));                                          \
  qpx_r5 = vec_ld2(0, &((u).c22.re));                                          \
  qpx_r6 = vec_ld2(0, &((u).c23.re));                                          \
  qpx_r7 = vec_ld2(0, &((u).c31.re));                                          \
  qpx_r8 = vec_ld2(0, &((u).c32.re));                                          \
  qpx_r9 = vec_ld2(0, &((u).c33.re));                                          \
  qpx_r10 = vec_perm(psi##1, psi##2, perm12);                                  \
  qpx_r11 = vec_sldw(psi##1, psi##3, 2);                                       \
  qpx_r12 = vec_perm(psi##2, psi##3, perm12);                                  \
  qpx_r13 = vec_xxcpnmadd(qpx_r10, qpx_r1, vec_xmul(qpx_r1, qpx_r10));         \
  qpx_r14 =                                                                    \
      vec_xxcpnmadd(qpx_r11, qpx_r4, vec_xmadd(qpx_r4, qpx_r11, qpx_r13));     \
  qpx_r15 =                                                                    \
      vec_xxcpnmadd(qpx_r12, qpx_r7, vec_xmadd(qpx_r7, qpx_r12, qpx_r14));     \
  qpx_r13 = vec_xxcpnmadd(qpx_r10, qpx_r2, vec_xmul(qpx_r2, qpx_r10));         \
  qpx_r14 =                                                                    \
      vec_xxcpnmadd(qpx_r11, qpx_r5, vec_xmadd(qpx_r5, qpx_r11, qpx_r13));     \
  qpx_r16 =                                                                    \
      vec_xxcpnmadd(qpx_r12, qpx_r8, vec_xmadd(qpx_r8, qpx_r12, qpx_r14));     \
  qpx_r13 = vec_xxcpnmadd(qpx_r10, qpx_r3, vec_xmul(qpx_r3, qpx_r10));         \
  qpx_r14 =                                                                    \
      vec_xxcpnmadd(qpx_r11, qpx_r6, vec_xmadd(qpx_r6, qpx_r11, qpx_r13));     \
  qpx_r17 =                                                                    \
      vec_xxcpnmadd(qpx_r12, qpx_r9, vec_xmadd(qpx_r9, qpx_r12, qpx_r14));     \
  res##1 = vec_perm(qpx_r15, qpx_r16, perm1);                                  \
  res##2 = vec_perm(qpx_r17, qpx_r15, perm12);                                 \
  res##3 = vec_perm(qpx_r16, qpx_r17, perm2);

#define _qpx_vec_i_add_assign(res, va)                                         \
  res##1 = vec_xxnpmadd(va##1, vec_i, res##1);                                 \
  res##2 = vec_xxnpmadd(va##2, vec_i, res##2);                                 \
  res##3 = vec_xxnpmadd(va##3, vec_i, res##3);

#define _qpx_vec_add_assign(va, vb)                                            \
  va##1 = vec_add(va##1, vb##1);                                               \
  va##2 = vec_add(va##2, vb##2);                                               \
  va##3 = vec_add(va##3, vb##3);

#define _qpx_vec_sub_assign(va, vb)                                            \
  va##1 = vec_sub(va##1, vb##1);                                               \
  va##2 = vec_sub(va##2, vb##2);                                               \
  va##3 = vec_sub(va##3, vb##3);

#define _qpx_vec_i_sub_assign(res, va)                                         \
  res##1 = vec_xxcpnmadd(va##1, vec_i, res##1);                                \
  res##2 = vec_xxcpnmadd(va##2, vec_i, res##2);                                \
  res##3 = vec_xxcpnmadd(va##3, vec_i, res##3);

#define _qpx_vec_prod(a, b, res)                                               \
  res##1 = vec_xxcpnmadd(b##1, a##1, vec_xmadd(a##1, b##1, res##1));           \
  res##2 = vec_xxcpnmadd(b##2, a##2, vec_xmadd(a##2, b##2, res##2));           \
  res##3 = vec_xxcpnmadd(b##3, a##3, vec_xmadd(a##3, b##3, res##3));

#endif
