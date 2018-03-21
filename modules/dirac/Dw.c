
/*****************************************************************************
 *
 * File Dw.c
 *
 * Copyright (C) 2005, 2011-2013, 2016 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Application of the O(a)-improved Wilson-Dirac operator D (single-
 * precision programs).
 *
 * The externally accessible functions are
 *
 *   void Dw(float mu,spinor *s,spinor *r)
 *     Depending on whether the twisted-mass flag is set or not, this
 *     program applies D+i*mu*gamma_5*1e or D+i*mu*gamma_5 to the field
 *     s and assigns the result to the field r.
 *
 *   void Dwee(float mu,spinor *s,spinor *r)
 *     Applies D_ee+i*mu*gamma_5 to the field s on the even points of the
 *     lattice and assigns the result to the field r.
 *
 *   void Dwoo(float mu,spinor *s,spinor *r)
 *     Depending on whether the twisted-mass flag is set or not, this
 *     program applies D_oo or D_oo+i*mu*gamma_5 to the field s on the
 *     odd points of the lattice and assigns the result to the field r.
 *
 *   void Dwoe(spinor *s,spinor *r)
 *     Applies D_oe to the field s and assigns the result to the field r.
 *
 *   void Dweo(spinor *s,spinor *r)
 *     Applies D_eo to the field s and *subtracts* the result from the
 *     field r.
 *
 *   void Dwhat(float mu,spinor *s,spinor *r)
 *     Applies Dhat+i*mu*gamma_5 to the field s and assigns the result to
 *     the field r.
 *
 * The following programs operate on the fields in the n'th block b of the
 * specified block grid:
 *
 *   void Dw_blk(blk_grid_t grid,int n,float mu,int k,int l)
 *     Depending on whether the twisted-mass flag is set or not, this
 *     program applies D+i*mu*gamma_5*1e or D+i*mu*gamma_5 to the field
 *     b.s[k] and assigns the result to the field b.s[l].
 *
 *   void Dwee_blk(blk_grid_t grid,int n,float mu,int k,int l)
 *     Applies D_ee+i*mu*gamma_5 to the field b.s[k] on the even points and
 *     assigns the result to the field b.s[l].
 *
 *   void Dwoo_blk(blk_grid_t grid,int n,float mu,int k,int l)
 *     Depending on whether the twisted-mass flag is set or not, this
 *     program applies D_oo or D_oo+i*mu*gamma_5 to the field b.s[k] on
 *     the odd points and assigns the result to the field b.s[l].
 *
 *   void Dwoe_blk(blk_grid_t grid,int n,int k,int l)
 *     Applies D_oe to the field b.s[k] and assigns the result to the field
 *     b.s[l].
 *
 *   void Dweo_blk(blk_grid_t grid,int n,int k,int l)
 *     Applies D_eo to the field b.s[k] and *subtracts* the result from the
 *     field b.s[l].
 *
 *   void Dwhat_blk(blk_grid_t grid,int n,float mu,int k,int l)
 *     Applies Dhat+i*mu*gamma_5 to the field b.s[k] and assigns the result
 *     to the field b.s[l].
 *
 * Notes:
 *
 * The notation and normalization conventions are specified in the notes
 * "Implementation of the lattice Dirac operator" (file doc/dirac.pdf).
 *
 * In all these programs, it is assumed that the SW term is in the proper
 * condition and that the global spinor fields have NSPIN elements. The
 * programs check whether the twisted-mass flag (see flags/lat_parms.c) is
 * set and turn off the twisted-mass term on the odd lattice sites if it is.
 * The input and output fields may not coincide in the case of the programs
 * Dw(), Dwhat(), Dw_blk() and Dwhat_blk().
 *
 * When the input and output fields are different, the input field is not
 * changed except possibly at the points at global time 0 and NPROC0*L0-1,
 * where both fields are set to zero if so required by the chosen boundary
 * conditions. Depending on the operator considered, the fields are zeroed
 * only on the even or odd points at these times.
 *
 * The programs Dw(),..,Dwhat() perform global operations and must be called
 * simultaneously on all processes.
 *
 * CONST_CORRECTNESS:
 *   The functions here are the main culprints why the entire codebase cannot be
 *   made to satisfy const correctness. All the functions should have the
 *   signature:
 *    
 *   Dw...(float mu, spinor const *s, spinor *r)
 *   
 *   However, due to the fact that these routines manipulate the boundary of the
 *   field s this cannot be the case. The boundary storage is fairly ingrained
 *   into the computation, so it will be a bit of work to decouple these. Any
 *   future attempt at making the rest of the code const correct will have to
 *   start here.
 *
 *****************************************************************************/

#define DW_C

#include "dirac.h"
#include "global.h"
#include "lattice.h"
#include "mpi.h"
#include "sflds.h"
#include "sw_term.h"
#include "uflds.h"

#define N0 (NPROC0 * L0)

typedef union
{
  spinor s;
  weyl w[2];
} spin_t;

static float coe, ceo;
static float gamma_f, one_over_gammaf;
static const spinor s0 = {{{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}},
                          {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}},
                          {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}},
                          {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}}};
static spin_t rs ALIGNED32;

static void set_aniso_hopping_coeffs(void)
{
  ani_params_t ani;
  ani = ani_parms();

  gamma_f = (float)(ani.xi / ani.nu);
  one_over_gammaf = (float)(ani.nu / ani.xi);

  coe *= (float)(1.0 / ani.ut_fermion);
  ceo *= (float)(1.0 / ani.ut_fermion);
}

#if (defined AVX512)

#include "avx512.h"
#include "sse.h"

static void doe(int *piup, int *pidn, su3 *u, spinor *pk)
{
  spinor *sp, *sm, *sp2, *sm2;
  su3 *up, *up1, *u1, *up2, *u2;

  /* 512-bit wide stores for the spinor for each color */
  __m512 a1, a2, a3;
  __m512 b1, b2, b3;
  __m128 w1, w2, w3, w4, w5, w6;

  /******************************* direction 0,1
   * *********************************/

  sp = pk + (*(piup++));
  sm = pk + (*(pidn++));
  sp2 = pk + (*(piup++));
  sm2 = pk + (*(pidn++));

  _avx512_load_4_halfspinor_f(a1, a2, a3, &(*sp).c1.c1.re, &(*sm).c1.c1.re,
                              &(*sp2).c1.c1.re, &(*sm2).c1.c1.re);
  _avx512_load_4_halfspinor_f_reverse_up(b1, b2, b3, &(*sp).c3.c1.re,
                                         &(*sm).c3.c1.re, &(*sp2).c3.c1.re,
                                         &(*sm2).c3.c1.re);

  sp = pk + (*(piup++));
  _prefetch_spinor(sp);
  sm = pk + (*(pidn++));
  _prefetch_spinor(sm);
  sp2 = pk + (*(piup));
  _prefetch_spinor(sp2);
  sm2 = pk + (*(pidn));
  _prefetch_spinor(sm2);

  up1 = u;
  u1 = u + 1;
  up2 = u + 2;
  u2 = u + 3;
  u = u2;
  _avx512_dirac_combine_f_1(a1, b1);
  _avx512_dirac_combine_f_1(a2, b2);
  _avx512_dirac_combine_f_1(a3, b3);

  avx512_su3_mixed_multiply_8(*up1, *u1, *up2, *u2, b1, b2, b3, a1, a2, a3);

  _avx512_to_weyl_f_1(w1, b1, gamma_f);
  _avx512_to_weyl_f_1(w2, b2, gamma_f);
  _avx512_to_weyl_f_1(w3, b3, gamma_f);
  _avx512_to_weyl_f_2(w4, b1, gamma_f);
  _avx512_to_weyl_f_2(w5, b2, gamma_f);
  _avx512_to_weyl_f_2(w6, b3, gamma_f);

  /******************************* direction 2,3
   **********************************/

  _avx512_load_4_halfspinor_f(a1, a2, a3, &(*sp).c1.c1.re, &(*sm).c1.c1.re,
                              &(*sp2).c1.c1.re, &(*sm2).c1.c1.re);
  _avx512_load_4_halfspinor_f_reverse_dn(b1, b2, b3, &(*sp).c3.c1.re,
                                         &(*sm).c3.c1.re, &(*sp2).c3.c1.re,
                                         &(*sm2).c3.c1.re);

  _avx512_dirac_combine_f_2(a1, b1);
  _avx512_dirac_combine_f_2(a2, b2);
  _avx512_dirac_combine_f_2(a3, b3);

  up1 = u + 1;
  u1 = u + 2;
  up2 = u + 3;
  u2 = u + 4;
  avx512_su3_mixed_multiply_8(*up1, *u1, *up2, *u2, b1, b2, b3, a1, a2, a3);

  _avx512_to_weyl_f_3(w1, b1);
  _avx512_to_weyl_f_3(w2, b2);
  _avx512_to_weyl_f_3(w3, b3);
  _avx512_to_weyl_f_4(w4, b1);
  _avx512_to_weyl_f_4(w5, b2);
  _avx512_to_weyl_f_4(w6, b3);

  float gc = one_over_gammaf * coe;
  register __m128 gamma = _mm_load_ps1(&gc);
  w1 = _mm_mul_ps(gamma, w1);
  w2 = _mm_mul_ps(gamma, w2);
  w3 = _mm_mul_ps(gamma, w3);
  w4 = _mm_mul_ps(gamma, w4);
  w5 = _mm_mul_ps(gamma, w5);
  w6 = _mm_mul_ps(gamma, w6);

  _avx512_write_2_f(w1, &rs.s.c1.c1.re, &rs.s.c2.c1.re);
  _avx512_write_2_f(w2, &rs.s.c1.c2.re, &rs.s.c2.c2.re);
  _avx512_write_2_f(w3, &rs.s.c1.c3.re, &rs.s.c2.c3.re);
  _avx512_write_2_f(w4, &rs.s.c3.c1.re, &rs.s.c4.c1.re);
  _avx512_write_2_f(w5, &rs.s.c3.c2.re, &rs.s.c4.c2.re);
  _avx512_write_2_f(w6, &rs.s.c3.c3.re, &rs.s.c4.c3.re);
}

static void deo(int *piup, int *pidn, su3 *u, spinor *pl)
{
  spinor *sp, *sm, *sp2, *sm2;
  su3 *up, *up1, *u1, *up2, *u2;

  /* 512-bit wide stores for the spinor for each color */
  __m512 a1, a2, a3;
  __m512 b1, b2, b3;
  __m128 w1, w2, w3, w4, w5, w6;

  /******************************* direction +0
   * *********************************/

  sp = pl + (*(piup++));
  _prefetch_spinor(sp);
  sm = pl + (*(pidn++));
  _prefetch_spinor(sm);
  sp2 = pl + (*(piup++));
  _prefetch_spinor(sp2);
  sm2 = pl + (*(pidn++));
  _prefetch_spinor(sm2);

  _avx512_load_2_f(w1, &rs.s.c1.c1.re, &rs.s.c2.c1.re);
  _avx512_load_2_f(w2, &rs.s.c1.c2.re, &rs.s.c2.c2.re);
  _avx512_load_2_f(w3, &rs.s.c1.c3.re, &rs.s.c2.c3.re);
  _avx512_load_2_f(w4, &rs.s.c3.c1.re, &rs.s.c4.c1.re);
  _avx512_load_2_f(w5, &rs.s.c3.c2.re, &rs.s.c4.c2.re);
  _avx512_load_2_f(w6, &rs.s.c3.c3.re, &rs.s.c4.c3.re);

  register __m128 c128 = _mm_load_ps1(&ceo);
  w1 = _mm_mul_ps(c128, w1);
  w2 = _mm_mul_ps(c128, w2);
  w3 = _mm_mul_ps(c128, w3);
  w4 = _mm_mul_ps(c128, w4);
  w5 = _mm_mul_ps(c128, w5);
  w6 = _mm_mul_ps(c128, w6);

  _avx512_to_dirac_f_1(a1, w1, w4);
  _avx512_to_dirac_f_1(a2, w2, w5);
  _avx512_to_dirac_f_1(a3, w3, w6);

  c128 = _mm_load_ps1(&one_over_gammaf);
  w1 = _mm_mul_ps(c128, w1);
  w2 = _mm_mul_ps(c128, w2);
  w3 = _mm_mul_ps(c128, w3);
  w4 = _mm_mul_ps(c128, w4);
  w5 = _mm_mul_ps(c128, w5);
  w6 = _mm_mul_ps(c128, w6);

  _avx512_to_dirac_f_2(a1, w1, w4);
  _avx512_to_dirac_f_2(a2, w2, w5);
  _avx512_to_dirac_f_2(a3, w3, w6);

  up1 = u;
  u1 = u + 1;
  up2 = u + 2;
  u2 = u + 3;
  u = u2;
  avx512_su3_mixed_multiply_8(*u1, *up1, *u2, *up2, b1, b2, b3, a1, a2, a3);

  _avx512_load_4_halfspinor_f(a1, a2, a3, &(*sm).c1.c1.re, &(*sp).c1.c1.re,
                              &(*sm2).c1.c1.re, &(*sp2).c1.c1.re);
  a1 = _mm512_add_ps(a1, b1);
  a2 = _mm512_add_ps(a2, b2);
  a3 = _mm512_add_ps(a3, b3);
  _avx512_write_4_halfspinor_f(a1, a2, a3, &(*sm).c1.c1.re, &(*sp).c1.c1.re,
                               &(*sm2).c1.c1.re, &(*sp2).c1.c1.re);

  _avx512_load_4_halfspinor_f_reverse_up(a1, a2, a3, &(*sm).c3.c1.re,
                                         &(*sp).c3.c1.re, &(*sm2).c3.c1.re,
                                         &(*sp2).c3.c1.re);
  _avx512_dirac_combine_f_3(a1, b1);
  _avx512_dirac_combine_f_3(a2, b2);
  _avx512_dirac_combine_f_3(a3, b3);
  _avx512_write_4_halfspinor_f_reverse_up(a1, a2, a3, &(*sm).c3.c1.re,
                                          &(*sp).c3.c1.re, &(*sm2).c3.c1.re,
                                          &(*sp2).c3.c1.re);

  /******************************* direction +2
   * *********************************/

  sp = pl + (*(piup++));
  _prefetch_spinor(sp);
  sm = pl + (*(pidn++));
  _prefetch_spinor(sm);
  sp2 = pl + (*(piup++));
  _prefetch_spinor(sp2);
  sm2 = pl + (*(pidn++));
  _prefetch_spinor(sm2);

  _avx512_to_dirac_f_3(a1, w1, w4);
  _avx512_to_dirac_f_3(a2, w2, w5);
  _avx512_to_dirac_f_3(a3, w3, w6);
  _avx512_to_dirac_f_4(a1, w1, w4);
  _avx512_to_dirac_f_4(a2, w2, w5);
  _avx512_to_dirac_f_4(a3, w3, w6);

  up1 = u + 1;
  u1 = u + 2;
  up2 = u + 3;
  u2 = u + 4;
  avx512_su3_mixed_multiply_8(*u1, *up1, *u2, *up2, b1, b2, b3, a1, a2, a3);

  _avx512_load_4_halfspinor_f(a1, a2, a3, &(*sm).c1.c1.re, &(*sp).c1.c1.re,
                              &(*sm2).c1.c1.re, &(*sp2).c1.c1.re);
  a1 = _mm512_add_ps(a1, b1);
  a2 = _mm512_add_ps(a2, b2);
  a3 = _mm512_add_ps(a3, b3);
  _avx512_write_4_halfspinor_f(a1, a2, a3, &(*sm).c1.c1.re, &(*sp).c1.c1.re,
                               &(*sm2).c1.c1.re, &(*sp2).c1.c1.re);

  _avx512_load_4_halfspinor_f_reverse_dn(a1, a2, a3, &(*sm).c3.c1.re,
                                         &(*sp).c3.c1.re, &(*sm2).c3.c1.re,
                                         &(*sp2).c3.c1.re);
  _avx512_dirac_combine_f_4(a1, b1);
  _avx512_dirac_combine_f_4(a2, b2);
  _avx512_dirac_combine_f_4(a3, b3);
  _avx512_write_4_halfspinor_f_reverse_dn(a1, a2, a3, &(*sm).c3.c1.re,
                                          &(*sp).c3.c1.re, &(*sm2).c3.c1.re,
                                          &(*sp2).c3.c1.re);
}

#elif (defined AVX)

#include "avx.h"

#define _load_cst(c)                                                           \
  __asm__ __volatile__("vbroadcastss %0, %%ymm15 \n\t" : : "m"(c) : "xmm15")

#define _mul_cst()                                                             \
  __asm__ __volatile__("vmulps %%ymm15, %%ymm0, %%ymm0 \n\t"                   \
                       "vmulps %%ymm15, %%ymm1, %%ymm1 \n\t"                   \
                       "vmulps %%ymm15, %%ymm2, %%ymm2"                        \
                       :                                                       \
                       :                                                       \
                       : "xmm0", "xmm1", "xmm2")

static void doe(int *piup, int *pidn, su3 *u, spinor *pk)
{
  spinor *sp, *sm;

  /****************************** direction 0 *******************************/

  sp = pk + piup[0];
  sm = pk + pidn[0];

  _avx_spinor_pair_load34(*sp, *sm);

  sp = pk + piup[1];
  sm = pk + pidn[1];
  _prefetch_spinor(sp);
  _prefetch_spinor(sm);

  _avx_spinor_mul_up(_avx_sgn_add);
  _avx_spinor_add();

  _avx_su3_pair_mixed_multiply(u[0], u[1]);

  _avx_spinor_split();
  _avx_spinor_unsplit();
  _avx_spinor_store_up(rs.s);

  _load_cst(gamma_f);
  _avx_spinor_load(rs.s);
  _mul_cst();
  _avx_spinor_store(rs.s);

  /****************************** direction 1 *******************************/

  _avx_spinor_pair_load43(*sp, *sm);

  sp = pk + piup[2];
  sm = pk + pidn[2];
  _prefetch_spinor(sp);
  _prefetch_spinor(sm);

  _avx_spinor_imul_up(_avx_sgn_i_add);
  _avx_spinor_add();

  _avx_su3_pair_mixed_multiply(u[2], u[3]);

  _avx_spinor_split();
  _avx_spinor_load(rs.s);
  _avx_weyl_xch_imul(_sse_sgn24);
  _avx_spinor_unsplit();
  _avx_spinor_add();
  _avx_spinor_store(rs.s);

  /****************************** direction 2 *******************************/

  _avx_spinor_pair_load43(*sp, *sm);

  sp = pk + piup[3];
  sm = pk + pidn[3];
  _prefetch_spinor(sp);
  _prefetch_spinor(sm);

  _avx_spinor_mul_up(_avx_sgn_addsub);
  _avx_spinor_add();

  _avx_su3_pair_mixed_multiply(u[4], u[5]);

  _avx_spinor_split();
  _avx_spinor_load(rs.s);
  _avx_weyl_xch();
  _avx_weyl_mul(_sse_sgn12);
  _avx_spinor_unsplit();
  _avx_spinor_add();
  _avx_spinor_store(rs.s);

  /****************************** direction 3 *******************************/

  _avx_spinor_pair_load34(*sp, *sm);
  _avx_spinor_imul_up(_avx_sgn_i_addsub);
  _avx_spinor_add();

  _avx_su3_pair_mixed_multiply(u[6], u[7]);

  _avx_spinor_split();
  _avx_spinor_load(rs.s);
  _avx_weyl_imul(_sse_sgn23);
  _avx_spinor_unsplit();
  _load_cst(coe);
  _avx_spinor_add();
  _mul_cst();
  _avx_weyl_pair_store12(rs.w[0], rs.w[1]);

  _load_cst(one_over_gammaf);
  _avx_weyl_pair_load12(rs.w[0], rs.w[1]);
  _mul_cst();
  _avx_weyl_pair_store12(rs.w[0], rs.w[1]);

  _avx_zeroupper();
}

static void deo(int *piup, int *pidn, su3 *u, spinor *pl)
{
  spinor *sp, *sm;

  _load_cst(ceo);
  _avx_spinor_load(rs.s);
  _mul_cst();
  _avx_spinor_store(rs.s);

  /****************************** direction 0 *******************************/

  sm = pl + pidn[0];
  sp = pl + piup[0];

  _prefetch_spinor(sm);
  _prefetch_spinor(sp);

  _avx_spinor_load_dup(rs.s);
  _avx_spinor_mul_up(_avx_sgn_add);
  _avx_spinor_add();

  _avx_su3_pair_mixed_multiply(u[1], u[0]);

  _avx_weyl_pair_load12(*sm, *sp);
  _avx_spinor_add();
  _avx_weyl_pair_store12(*sm, *sp);

  _avx_weyl_pair_load34(*sm, *sp);
  _avx_spinor_mul_up(_avx_sgn_add);
  _avx_spinor_add();
  _avx_weyl_pair_store34(*sm, *sp);

  /****************************** direction 1 *******************************/

  _load_cst(one_over_gammaf);
  _avx_spinor_load(rs.s);
  _mul_cst();
  _avx_spinor_store(rs.s);

  sm = pl + pidn[1];
  sp = pl + piup[1];

  _prefetch_spinor(sm);
  _prefetch_spinor(sp);

  _avx_spinor_load_dup(rs.s);
  _avx_spinor_xch_imul_up(_avx_sgn_i_add);
  _avx_spinor_add();

  _avx_su3_pair_mixed_multiply(u[3], u[2]);

  _avx_weyl_pair_load12(*sm, *sp);
  _avx_spinor_add();
  _avx_weyl_pair_store12(*sm, *sp);

  _avx_weyl_pair_load34(*sm, *sp);
  _avx_spinor_xch_imul_up(_avx_sgn_i_add);
  _avx_spinor_sub();
  _avx_weyl_pair_store34(*sm, *sp);

  /****************************** direction 2 *******************************/

  sm = pl + pidn[2];
  sp = pl + piup[2];

  _prefetch_spinor(sm);
  _prefetch_spinor(sp);

  _avx_spinor_load_dup(rs.s);
  _avx_spinor_xch_up();
  _avx_spinor_mul_up(_avx_sgn_addsub);
  _avx_spinor_add();

  _avx_su3_pair_mixed_multiply(u[5], u[4]);

  _avx_weyl_pair_load12(*sm, *sp);
  _avx_spinor_add();
  _avx_weyl_pair_store12(*sm, *sp);

  _avx_weyl_pair_load34(*sm, *sp);
  _avx_spinor_xch_up();
  _avx_spinor_mul_up(_avx_sgn_addsub);
  _avx_spinor_sub();
  _avx_weyl_pair_store34(*sm, *sp);

  /****************************** direction 3 *******************************/

  sm = pl + pidn[3];
  sp = pl + piup[3];

  _prefetch_spinor(sm);
  _prefetch_spinor(sp);

  _avx_spinor_load_dup(rs.s);
  _avx_spinor_imul_up(_avx_sgn_i_addsub);
  _avx_spinor_add();

  _avx_su3_pair_mixed_multiply(u[7], u[6]);

  _avx_weyl_pair_load12(*sm, *sp);
  _avx_spinor_add();
  _avx_weyl_pair_store12(*sm, *sp);

  _avx_weyl_pair_load34(*sm, *sp);
  _avx_spinor_imul_up(_avx_sgn_i_addsub);
  _avx_spinor_sub();
  _avx_weyl_pair_store34(*sm, *sp);

  _avx_zeroupper();
}

#elif (defined x64)
#include "sse2.h"

#define _load_cst(c)                                                           \
  __asm__ __volatile__("movss %0, %%xmm15 \n\t"                                \
                       "shufps $0x0, %%xmm15, %%xmm15"                         \
                       :                                                       \
                       : "m"(c)                                                \
                       : "xmm15")

#define _mul_cst()                                                             \
  __asm__ __volatile__("mulps %%xmm15, %%xmm0 \n\t"                            \
                       "mulps %%xmm15, %%xmm1 \n\t"                            \
                       "mulps %%xmm15, %%xmm2"                                 \
                       :                                                       \
                       :                                                       \
                       : "xmm0", "xmm1", "xmm2")

#define _mul_cst_up()                                                          \
  __asm__ __volatile__("mulps %%xmm15, %%xmm3 \n\t"                            \
                       "mulps %%xmm15, %%xmm4 \n\t"                            \
                       "mulps %%xmm15, %%xmm5"                                 \
                       :                                                       \
                       :                                                       \
                       : "xmm3", "xmm4", "xmm5")

static void doe(int *piup, int *pidn, su3 *u, spinor *pk)
{
  spinor *sp, *sm;

  /***************************** direction +0 *******************************/

  sp = pk + (*(piup++));

  _sse_pair_load((*sp).c1, (*sp).c2);
  _sse_pair_load_up((*sp).c3, (*sp).c4);

  sm = pk + (*(pidn++));
  _prefetch_spinor(sm);

  _sse_vector_add();
  sp = pk + (*(piup++));
  _prefetch_spinor(sp);
  _sse_su3_multiply(*u);

  _sse_weyl_store_up(rs.w[0]);
  _sse_weyl_store_up(rs.w[1]);

  /***************************** direction -0 *******************************/

  _sse_pair_load((*sm).c1, (*sm).c2);
  _sse_pair_load_up((*sm).c3, (*sm).c4);

  u += 2;
  _prefetch_su3_dble(u);
  u -= 1;
  _sse_vector_sub();
  sm = pk + (*(pidn++));
  _prefetch_spinor(sm);
  _sse_su3_inverse_multiply(*u);

  _sse_weyl_load(rs.w[0]);
  _sse_vector_add();
  _sse_weyl_store(rs.w[0]);

  _sse_weyl_load(rs.w[1]);
  _sse_vector_sub();
  _sse_weyl_store(rs.w[1]);

  _load_cst(gamma_f);
  _sse_weyl_load(rs.w[0]);
  _mul_cst();
  _sse_weyl_store(rs.w[0]);
  _sse_weyl_load(rs.w[1]);
  _mul_cst();
  _sse_weyl_store(rs.w[1]);

  /***************************** direction +1 *******************************/

  _sse_pair_load((*sp).c1, (*sp).c2);
  _sse_pair_load_up((*sp).c4, (*sp).c3);

  _sse_vector_i_add();
  sp = pk + (*(piup++));
  _prefetch_spinor(sp);
  u += 1;
  _sse_su3_multiply(*u);

  _sse_weyl_load(rs.w[0]);
  _sse_vector_add();
  _sse_weyl_store(rs.w[0]);

  _sse_weyl_load(rs.w[1]);
  _sse_vector_xch_i_sub();
  _sse_weyl_store(rs.w[1]);

  /***************************** direction -1 *******************************/

  _sse_pair_load((*sm).c1, (*sm).c2);
  _sse_pair_load_up((*sm).c4, (*sm).c3);

  u += 2;
  _prefetch_su3_dble(u);
  u -= 1;
  _sse_vector_i_sub();
  sm = pk + (*(pidn++));
  _prefetch_spinor(sm);
  _sse_su3_inverse_multiply(*u);

  _sse_weyl_load(rs.w[0]);
  _sse_vector_add();
  _sse_weyl_store(rs.w[0]);

  _sse_weyl_load(rs.w[1]);
  _sse_vector_xch_i_add();
  _sse_weyl_store(rs.w[1]);

  /***************************** direction +2 *******************************/

  _sse_pair_load((*sp).c1, (*sp).c2);
  _sse_pair_load_up((*sp).c4, (*sp).c3);

  _sse_vector_addsub();

  u += 1;
  _sse_su3_multiply(*u);
  sp = pk + (*(piup));
  _prefetch_spinor(sp);
  _sse_weyl_load(rs.w[0]);
  _sse_vector_add();
  _sse_weyl_store(rs.w[0]);

  _sse_weyl_load(rs.w[1]);
  _sse_vector_xch();
  _sse_vector_subadd();
  _sse_weyl_store(rs.w[1]);

  /***************************** direction -2 *******************************/

  _sse_pair_load((*sm).c1, (*sm).c2);
  _sse_pair_load_up((*sm).c4, (*sm).c3);

  u += 2;
  _prefetch_su3_dble(u);
  u -= 1;
  _sse_vector_subadd();
  sm = pk + (*(pidn));
  _prefetch_spinor(sm);
  _sse_su3_inverse_multiply(*u);

  _sse_weyl_load(rs.w[0]);
  _sse_vector_add();
  _sse_weyl_store(rs.w[0]);

  _sse_weyl_load(rs.w[1]);
  _sse_vector_xch();
  _sse_vector_addsub();
  _sse_weyl_store(rs.w[1]);

  /***************************** direction +3 *******************************/

  _sse_pair_load((*sp).c1, (*sp).c2);
  _sse_pair_load_up((*sp).c3, (*sp).c4);

  _sse_vector_i_addsub();
  u += 1;
  _sse_su3_multiply(*u);

  _sse_weyl_load(rs.w[0]);
  _sse_vector_add();
  _sse_weyl_store(rs.w[0]);

  _sse_weyl_load(rs.w[1]);
  _sse_vector_i_subadd();
  _sse_weyl_store(rs.w[1]);

  /***************************** direction -3 *******************************/

  _sse_pair_load((*sm).c1, (*sm).c2);
  _sse_pair_load_up((*sm).c3, (*sm).c4);

  u += 2;
  _prefetch_su3_dble(u);
  u -= 1;
  _sse_vector_i_subadd();
  _sse_su3_inverse_multiply(*u);

  _load_cst(coe);
  _sse_weyl_load(rs.w[0]);
  _sse_vector_add();
  _mul_cst();
  _sse_pair_store(rs.s.c1, rs.s.c2);

  _sse_weyl_load(rs.w[1]);
  _sse_vector_i_addsub();
  _mul_cst();
  _sse_pair_store(rs.s.c3, rs.s.c4);

  _load_cst(one_over_gammaf);
  _sse_pair_load(rs.s.c1, rs.s.c2);
  _mul_cst();
  _sse_pair_store(rs.s.c1, rs.s.c2);
  _sse_pair_load(rs.s.c3, rs.s.c4);
  _mul_cst();
  _sse_pair_store(rs.s.c3, rs.s.c4);
}

static void deo(int *piup, int *pidn, su3 *u, spinor *pl)
{
  spinor *sp, *sm;

  /***************************** direction +0 *******************************/

  sp = pl + (*(piup++));
  _prefetch_spinor(sp);

  _load_cst(ceo);
  _sse_pair_load(rs.s.c1, rs.s.c2);
  _sse_pair_load_up(rs.s.c3, rs.s.c4);
  _mul_cst();
  _mul_cst_up();
  _sse_weyl_store(rs.w[0]);
  _sse_weyl_store_up(rs.w[1]);

  sm = pl + (*(pidn++));
  _prefetch_spinor(sm);
  _sse_vector_sub();
  _sse_su3_inverse_multiply(*u);

  _sse_pair_load((*sp).c1, (*sp).c2);
  _sse_vector_add();
  _sse_pair_store((*sp).c1, (*sp).c2);

  _sse_pair_load((*sp).c3, (*sp).c4);
  _sse_vector_sub();
  _sse_pair_store((*sp).c3, (*sp).c4);

  /***************************** direction -0 *******************************/

  _sse_weyl_load(rs.w[0]);
  _sse_weyl_load_up(rs.w[1]);

  sp = pl + (*(piup++));
  _prefetch_spinor(sp);
  _sse_vector_add();
  u += 1;
  _sse_su3_multiply(*u);

  _sse_pair_load((*sm).c1, (*sm).c2);
  _sse_vector_add();
  _sse_pair_store((*sm).c1, (*sm).c2);

  _sse_pair_load((*sm).c3, (*sm).c4);
  _sse_vector_add();
  _sse_pair_store((*sm).c3, (*sm).c4);

  /***************************** direction +1 *******************************/

  _load_cst(one_over_gammaf);
  _sse_weyl_load(rs.w[0]);
  _sse_weyl_load_up(rs.w[1]);
  _mul_cst();
  _mul_cst_up();
  _sse_weyl_store(rs.w[0]);
  _sse_weyl_store_up(rs.w[1]);

  _sse_weyl_load(rs.w[0]);
  _sse_weyl_load_up(rs.w[1]);

  sm = pl + (*(pidn++));
  _prefetch_spinor(sm);
  _sse_vector_xch_i_sub();
  u += 1;
  _sse_su3_inverse_multiply(*u);

  _sse_pair_load((*sp).c1, (*sp).c2);
  _sse_vector_add();
  _sse_pair_store((*sp).c1, (*sp).c2);

  _sse_pair_load((*sp).c3, (*sp).c4);
  _sse_vector_xch_i_add();
  _sse_pair_store((*sp).c3, (*sp).c4);

  /***************************** direction -1 *******************************/

  _sse_weyl_load(rs.w[0]);
  _sse_weyl_load_up(rs.w[1]);

  sp = pl + (*(piup++));
  _prefetch_spinor(sp);
  _sse_vector_xch_i_add();
  u += 1;
  _sse_su3_multiply(*u);

  _sse_pair_load((*sm).c1, (*sm).c2);
  _sse_vector_add();
  _sse_pair_store((*sm).c1, (*sm).c2);

  _sse_pair_load((*sm).c3, (*sm).c4);
  _sse_vector_xch_i_sub();
  _sse_pair_store((*sm).c3, (*sm).c4);

  /***************************** direction +2 *******************************/

  _sse_weyl_load(rs.w[0]);
  _sse_weyl_load_up(rs.w[1]);

  sm = pl + (*(pidn++));
  _prefetch_spinor(sm);
  _sse_vector_xch();
  _sse_vector_subadd();
  u += 1;
  _sse_su3_inverse_multiply(*u);

  _sse_pair_load((*sp).c1, (*sp).c2);
  _sse_vector_add();
  _sse_pair_store((*sp).c1, (*sp).c2);

  _sse_pair_load((*sp).c3, (*sp).c4);
  _sse_vector_xch();
  _sse_vector_addsub();
  _sse_pair_store((*sp).c3, (*sp).c4);

  /***************************** direction -2 *******************************/

  _sse_weyl_load(rs.w[0]);
  _sse_weyl_load_up(rs.w[1]);

  sp = pl + (*(piup));
  _prefetch_spinor(sp);
  _sse_vector_xch();
  _sse_vector_addsub();
  u += 1;
  _sse_su3_multiply(*u);

  _sse_pair_load((*sm).c1, (*sm).c2);
  _sse_vector_add();
  _sse_pair_store((*sm).c1, (*sm).c2);

  _sse_pair_load((*sm).c3, (*sm).c4);
  _sse_vector_xch();
  _sse_vector_subadd();
  _sse_pair_store((*sm).c3, (*sm).c4);

  /***************************** direction +3 *******************************/

  _sse_weyl_load(rs.w[0]);
  _sse_weyl_load_up(rs.w[1]);

  sm = pl + (*(pidn));
  _prefetch_spinor(sm);
  _sse_vector_i_subadd();
  u += 1;
  _sse_su3_inverse_multiply(*u);

  _sse_pair_load((*sp).c1, (*sp).c2);
  _sse_vector_add();
  _sse_pair_store((*sp).c1, (*sp).c2);

  _sse_pair_load((*sp).c3, (*sp).c4);
  _sse_vector_i_addsub();
  _sse_pair_store((*sp).c3, (*sp).c4);

  /***************************** direction -3 *******************************/

  _sse_weyl_load(rs.w[0]);
  _sse_weyl_load_up(rs.w[1]);

  _sse_vector_i_addsub();
  u += 1;
  _sse_su3_multiply(*u);

  _sse_pair_load((*sm).c1, (*sm).c2);
  _sse_vector_add();
  _sse_pair_store((*sm).c1, (*sm).c2);

  _sse_pair_load((*sm).c3, (*sm).c4);
  _sse_vector_i_subadd();
  _sse_pair_store((*sm).c3, (*sm).c4);
}

#elif (defined QPX)

#include "qpx.h"

static vector4double dim11, dim12, dim13, dim21, dim22, dim23;
static vector4double v11, v12, v13, v21, v22, v23;
static vector4double *im1[3] = {&dim11, &dim12, &dim13};
static vector4double *im2[3] = {&dim21, &dim22, &dim23};

static void doe(int *piup, int *pidn, su3 *u, spinor *pk)
{
  spinor *sp, *sm;
  su3_vector psi, chi;
  vector4double psi11, psi12, chi11, chi12, psi13, chi13, psi21, psi22, psi23,
      chi21, chi22, chi23, vcoe, vmul;
  int myrank;

  /***************************** direction +0 *******************************/

  sp = pk + (*(piup++));
  _qpx_load_w1(psi1, sp);
  _qpx_load_w2(psi2, sp);
  sm = pk + (*(pidn++));
  _qpx_prefetch_spinor_sp(sm);
  _qpx_vec_add(psi1, psi1, psi2);
  _qpx_su3_mul(dim1, *u, psi1);
  u += 1;
  _qpx_prefetch_su3_sp(u);
  dim21 = dim11;
  dim22 = dim12;
  dim23 = dim13;

  /***************************** direction -0 *******************************/

  _qpx_load_w1(psi1, sm);
  _qpx_load_w2(psi2, sm);
  sp = pk + (*(piup++));
  _qpx_prefetch_spinor_sp(sp);
  _qpx_vec_sub(psi1, psi1, psi2);
  _qpx_su3_inv_mul(chi1, *u, psi1);
  u += 1;
  _qpx_prefetch_su3_sp(u);
  _qpx_vec_add_assign(dim1, chi1);
  _qpx_vec_sub_assign(dim2, chi1);

  vmul = vec_splats(gamma_f);
  dim11 = vec_mul(vmul, dim11);
  dim12 = vec_mul(vmul, dim12);
  dim13 = vec_mul(vmul, dim13);
  dim21 = vec_mul(vmul, dim21);
  dim22 = vec_mul(vmul, dim22);
  dim23 = vec_mul(vmul, dim23);

  /***************************** direction +1 *******************************/

  _qpx_load_w1(psi1, sp);
  _qpx_load_w2(psi2, sp);
  sm = pk + (*(pidn++));
  _qpx_prefetch_spinor_sp(sm);
  _qpx_vec_x(chi1, psi2);
  _qpx_vec_i_add(psi1, psi1, chi1);
  _qpx_su3_mul(chi1, *u, psi1);
  u += 1;
  _qpx_prefetch_su3_sp(u);
  _qpx_vec_add_assign(dim1, chi1);
  _qpx_vec_x(chi2, chi1);
  _qpx_vec_i_sub(dim2, dim2, chi2);

  /***************************** direction -1 *******************************/

  _qpx_load_w1(psi1, sm);
  _qpx_load_w2(psi2, sm);
  sp = pk + (*(piup++));
  _qpx_prefetch_spinor_sp(sp);
  _qpx_vec_x(chi1, psi2);
  _qpx_vec_i_sub(psi1, psi1, chi1);
  _qpx_su3_inv_mul(chi1, *u, psi1);
  u += 1;
  _qpx_prefetch_su3_sp(u);
  _qpx_vec_add_assign(dim1, chi1);
  _qpx_vec_x(chi2, chi1);
  _qpx_vec_i_add(dim2, dim2, chi2);

  /***************************** direction +2 *******************************/

  _qpx_load_w1(psi1, sp);
  _qpx_load_w2(psi2, sp);
  sm = pk + (*(pidn++));
  _qpx_prefetch_spinor_sp(sm);
  _qpx_vec_x(chi1, psi2);
  _qpx_vec_add_n(psi1, psi1, chi1);
  _qpx_su3_mul(chi1, *u, psi1);
  u += 1;
  _qpx_prefetch_su3_sp(u);
  _qpx_vec_add_assign(dim1, chi1);
  _qpx_vec_x(chi2, chi1);
  _qpx_vec_sub_n(dim2, dim2, chi2);

  /***************************** direction -2 *******************************/

  _qpx_load_w1(psi1, sm);
  _qpx_load_w2(psi2, sm);
  sp = pk + (*(piup));
  _qpx_prefetch_spinor_sp(sp);
  _qpx_vec_x(chi1, psi2);
  _qpx_vec_sub_n(psi1, psi1, chi1);
  _qpx_su3_inv_mul(chi1, *u, psi1);
  u += 1;
  _qpx_prefetch_su3_sp(u);
  _qpx_vec_add_assign(dim1, chi1);
  _qpx_vec_x(chi2, chi1);
  _qpx_vec_add_n(dim2, dim2, chi2);

  /***************************** direction +3 *******************************/

  _qpx_load_w1(psi1, sp);
  _qpx_load_w2(psi2, sp);
  sm = pk + (*(pidn));
  _qpx_prefetch_spinor_sp(sm);
  _qpx_vec_i_add_n(psi1, psi1, psi2);
  _qpx_su3_mul(chi1, *u, psi1);
  u += 1;
  _qpx_prefetch_su3_sp(u);
  _qpx_vec_add(dim1, dim1, chi1);
  _qpx_vec_i_sub_n(dim2, dim2, chi1);

  /***************************** direction -3 *******************************/

  _qpx_load_w1(psi1, sm);
  _qpx_load_w2(psi2, sm);
  _qpx_vec_i_sub_n(psi1, psi1, psi2);
  _qpx_su3_inv_mul(chi1, *u, psi1);
  _qpx_vec_add(dim1, dim1, chi1);
  _qpx_vec_i_add_n(dim2, dim2, chi1);

  vcoe = vec_splats(coe);
  dim11 = vec_mul(vcoe, dim11);
  dim12 = vec_mul(vcoe, dim12);
  dim13 = vec_mul(vcoe, dim13);
  dim21 = vec_mul(vcoe, dim21);
  dim22 = vec_mul(vcoe, dim22);
  dim23 = vec_mul(vcoe, dim23);

  vmul = vec_splats(one_over_gammaf);
  dim11 = vec_mul(vmul, dim11);
  dim12 = vec_mul(vmul, dim12);
  dim13 = vec_mul(vmul, dim13);
  dim21 = vec_mul(vmul, dim21);
  dim22 = vec_mul(vmul, dim22);
  dim23 = vec_mul(vmul, dim23);
}

static void deo(int *piup, int *pidn, su3 *u, spinor *pl)
{
  spinor *sp, *sm;
  su3_vector psi, chi;
  vector4double psi11, psi12, chi11, chi12, psi13, chi13, chi21, chi22, chi23,
      vceo, vmul;

  vceo = vec_splats(ceo);
  dim11 = vec_mul(vceo, dim11);
  dim12 = vec_mul(vceo, dim12);
  dim13 = vec_mul(vceo, dim13);
  dim21 = vec_mul(vceo, dim21);
  dim22 = vec_mul(vceo, dim22);
  dim23 = vec_mul(vceo, dim23);

  /***************************** direction +0 *******************************/

  sp = pl + (*(piup++));
  _qpx_vec_sub(psi1, dim1, dim2);
  sm = pl + (*(pidn++));
  _qpx_prefetch_spinor_sp(sm);
  _qpx_su3_inv_mul(chi1, *u, psi1);
  u += 1;
  _qpx_prefetch_su3_sp(u);
  _qpx_load_w1(v1, sp);
  _qpx_load_w2(v2, sp);
  _qpx_vec_add(v1, v1, chi1);
  _qpx_store_w1(v1, sp);
  _qpx_vec_sub(v2, v2, chi1);
  _qpx_store_w2(v2, sp);

  /***************************** direction -0 *******************************/

  _qpx_vec_add(psi1, dim1, dim2);
  sp = pl + (*(piup++));
  _qpx_prefetch_spinor_sp(sp);
  _qpx_su3_mul(chi1, *u, psi1);
  u += 1;
  _qpx_prefetch_su3_sp(u);
  _qpx_load_w1(v1, sm);
  _qpx_load_w2(v2, sm);
  _qpx_vec_add(v1, v1, chi1);
  _qpx_store_w1(v1, sm);
  _qpx_vec_add(v2, v2, chi1);
  _qpx_store_w2(v2, sm);

  /***************************** direction +1 *******************************/

  vmul = vec_splats(one_over_gammaf);
  dim11 = vec_mul(vmul, dim11);
  dim12 = vec_mul(vmul, dim12);
  dim13 = vec_mul(vmul, dim13);
  dim21 = vec_mul(vmul, dim21);
  dim22 = vec_mul(vmul, dim22);
  dim23 = vec_mul(vmul, dim23);

  _qpx_vec_x(v1, dim2);
  _qpx_vec_i_sub(psi1, dim1, v1);
  sm = pl + (*(pidn++));
  _qpx_prefetch_spinor_sp(sm);
  _qpx_su3_inv_mul(chi1, *u, psi1);
  u += 1;
  _qpx_prefetch_su3_sp(u);
  _qpx_load_w1(v1, sp);
  _qpx_load_w2(v2, sp);
  _qpx_vec_add(v1, v1, chi1);
  _qpx_store_w1(v1, sp);
  _qpx_vec_x(chi2, chi1);
  _qpx_vec_i_add(psi1, v2, chi2);
  _qpx_store_w2(psi1, sp);

  /***************************** direction -1 *******************************/

  _qpx_vec_x(v1, dim2);
  _qpx_vec_i_add(psi1, dim1, v1);
  sp = pl + (*(piup++));
  _qpx_prefetch_spinor_sp(sp);
  _qpx_su3_mul(chi1, *u, psi1);
  u += 1;
  _qpx_prefetch_su3_sp(u);
  _qpx_load_w1(v1, sm);
  _qpx_load_w2(v2, sm);
  _qpx_vec_add(v1, v1, chi1);
  _qpx_store_w1(v1, sm);
  _qpx_vec_x(chi2, chi1);
  _qpx_vec_i_sub(psi1, v2, chi2);
  _qpx_store_w2(psi1, sm);

  /***************************** direction +2 *******************************/

  _qpx_vec_x(v1, dim2);
  _qpx_vec_sub_n(psi1, dim1, v1);
  sm = pl + (*(pidn++));
  _qpx_prefetch_spinor_sp(sm);
  _qpx_su3_inv_mul(chi1, *u, psi1);
  u += 1;
  _qpx_prefetch_su3_sp(u);
  _qpx_load_w1(v1, sp);
  _qpx_load_w2(v2, sp);
  _qpx_vec_add(v1, v1, chi1);
  _qpx_store_w1(v1, sp);
  _qpx_vec_x(chi2, chi1);
  _qpx_vec_add_n(psi1, v2, chi2);
  _qpx_store_w2(psi1, sp);

  /***************************** direction -2 *******************************/

  _qpx_vec_x(v1, dim2);
  _qpx_vec_add_n(psi1, dim1, v1);
  sp = pl + (*(piup));
  _qpx_prefetch_spinor_sp(sp);
  _qpx_su3_mul(chi1, *u, psi1);
  u += 1;
  _qpx_prefetch_su3_sp(u);
  _qpx_load_w1(v1, sm);
  _qpx_load_w2(v2, sm);
  _qpx_vec_add(v1, v1, chi1);
  _qpx_store_w1(v1, sm);
  _qpx_vec_x(chi2, chi1);
  _qpx_vec_sub_n(psi1, v2, chi2);
  _qpx_store_w2(psi1, sm);

  /***************************** direction +3 *******************************/

  _qpx_vec_i_sub_n(psi1, dim1, dim2);
  sm = pl + (*(pidn));
  _qpx_prefetch_spinor_sp(sm);
  _qpx_su3_inv_mul(chi1, *u, psi1);
  u += 1;
  _qpx_prefetch_su3_sp(u);
  _qpx_load_w1(v1, sp);
  _qpx_load_w2(v2, sp);
  _qpx_vec_add(v1, v1, chi1);
  _qpx_store_w1(v1, sp);
  _qpx_vec_i_add_n(psi1, v2, chi1);
  _qpx_store_w2(psi1, sp);

  /***************************** direction -3 *******************************/

  _qpx_vec_i_add_n(psi1, dim1, dim2);
  _qpx_su3_mul(chi1, *u, psi1);

  _qpx_load_w1(v1, sm);
  _qpx_load_w2(v2, sm);
  _qpx_vec_add(v1, v1, chi1);
  _qpx_store_w1(v1, sm);
  _qpx_vec_i_sub_n(psi1, v2, chi1);
  _qpx_store_w2(psi1, sm);
}

#else

#define _vector_mul_assign(r, c)                                               \
  (r).c1.re *= (c);                                                            \
  (r).c1.im *= (c);                                                            \
  (r).c2.re *= (c);                                                            \
  (r).c2.im *= (c);                                                            \
  (r).c3.re *= (c);                                                            \
  (r).c3.im *= (c)

static void doe(int *piup, int *pidn, su3 *u, spinor *pk)
{
  spinor *sp, *sm;
  su3_vector psi, chi;

  /***************************** direction +0 *******************************/

  sp = pk + (*(piup++));

  _vector_add(psi, (*sp).c1, (*sp).c3);
  _su3_multiply(rs.s.c1, *u, psi);
  rs.s.c3 = rs.s.c1;

  _vector_add(psi, (*sp).c2, (*sp).c4);
  _su3_multiply(rs.s.c2, *u, psi);
  rs.s.c4 = rs.s.c2;

  /***************************** direction -0 *******************************/

  sm = pk + (*(pidn++));
  u += 1;

  _vector_sub(psi, (*sm).c1, (*sm).c3);
  _su3_inverse_multiply(chi, *u, psi);
  _vector_add_assign(rs.s.c1, chi);
  _vector_sub_assign(rs.s.c3, chi);

  _vector_sub(psi, (*sm).c2, (*sm).c4);
  _su3_inverse_multiply(chi, *u, psi);
  _vector_add_assign(rs.s.c2, chi);
  _vector_sub_assign(rs.s.c4, chi);

  _vector_mul_assign(rs.s.c1, gamma_f);
  _vector_mul_assign(rs.s.c2, gamma_f);
  _vector_mul_assign(rs.s.c3, gamma_f);
  _vector_mul_assign(rs.s.c4, gamma_f);

  /***************************** direction +1 *******************************/

  sp = pk + (*(piup++));
  u += 1;

  _vector_i_add(psi, (*sp).c1, (*sp).c4);
  _su3_multiply(chi, *u, psi);
  _vector_add_assign(rs.s.c1, chi);
  _vector_i_sub_assign(rs.s.c4, chi);

  _vector_i_add(psi, (*sp).c2, (*sp).c3);
  _su3_multiply(chi, *u, psi);
  _vector_add_assign(rs.s.c2, chi);
  _vector_i_sub_assign(rs.s.c3, chi);

  /***************************** direction -1 *******************************/

  sm = pk + (*(pidn++));
  u += 1;

  _vector_i_sub(psi, (*sm).c1, (*sm).c4);
  _su3_inverse_multiply(chi, *u, psi);
  _vector_add_assign(rs.s.c1, chi);
  _vector_i_add_assign(rs.s.c4, chi);

  _vector_i_sub(psi, (*sm).c2, (*sm).c3);
  _su3_inverse_multiply(chi, *u, psi);
  _vector_add_assign(rs.s.c2, chi);
  _vector_i_add_assign(rs.s.c3, chi);

  /***************************** direction +2 *******************************/

  sp = pk + (*(piup++));
  u += 1;

  _vector_add(psi, (*sp).c1, (*sp).c4);
  _su3_multiply(chi, *u, psi);
  _vector_add_assign(rs.s.c1, chi);
  _vector_add_assign(rs.s.c4, chi);

  _vector_sub(psi, (*sp).c2, (*sp).c3);
  _su3_multiply(chi, *u, psi);
  _vector_add_assign(rs.s.c2, chi);
  _vector_sub_assign(rs.s.c3, chi);

  /***************************** direction -2 *******************************/

  sm = pk + (*(pidn++));
  u += 1;

  _vector_sub(psi, (*sm).c1, (*sm).c4);
  _su3_inverse_multiply(chi, *u, psi);
  _vector_add_assign(rs.s.c1, chi);
  _vector_sub_assign(rs.s.c4, chi);

  _vector_add(psi, (*sm).c2, (*sm).c3);
  _su3_inverse_multiply(chi, *u, psi);
  _vector_add_assign(rs.s.c2, chi);
  _vector_add_assign(rs.s.c3, chi);

  /***************************** direction +3 *******************************/

  sp = pk + (*(piup));
  u += 1;

  _vector_i_add(psi, (*sp).c1, (*sp).c3);
  _su3_multiply(chi, *u, psi);
  _vector_add_assign(rs.s.c1, chi);
  _vector_i_sub_assign(rs.s.c3, chi);

  _vector_i_sub(psi, (*sp).c2, (*sp).c4);
  _su3_multiply(chi, *u, psi);
  _vector_add_assign(rs.s.c2, chi);
  _vector_i_add_assign(rs.s.c4, chi);

  /***************************** direction -3 *******************************/

  sm = pk + (*(pidn));
  u += 1;

  _vector_i_sub(psi, (*sm).c1, (*sm).c3);
  _su3_inverse_multiply(chi, *u, psi);
  _vector_add_assign(rs.s.c1, chi);
  _vector_i_add_assign(rs.s.c3, chi);

  _vector_i_add(psi, (*sm).c2, (*sm).c4);
  _su3_inverse_multiply(chi, *u, psi);
  _vector_add_assign(rs.s.c2, chi);
  _vector_i_sub_assign(rs.s.c4, chi);

  _vector_mul_assign(rs.s.c1, coe);
  _vector_mul_assign(rs.s.c2, coe);
  _vector_mul_assign(rs.s.c3, coe);
  _vector_mul_assign(rs.s.c4, coe);

  _vector_mul_assign(rs.s.c1, one_over_gammaf);
  _vector_mul_assign(rs.s.c2, one_over_gammaf);
  _vector_mul_assign(rs.s.c3, one_over_gammaf);
  _vector_mul_assign(rs.s.c4, one_over_gammaf);
}

static void deo(int *piup, int *pidn, su3 *u, spinor *pl)
{
  spinor *sp, *sm;
  su3_vector psi, chi;

  _vector_mul_assign(rs.s.c1, ceo);
  _vector_mul_assign(rs.s.c2, ceo);
  _vector_mul_assign(rs.s.c3, ceo);
  _vector_mul_assign(rs.s.c4, ceo);

  /***************************** direction +0 *******************************/

  sp = pl + (*(piup++));

  _vector_sub(psi, rs.s.c1, rs.s.c3);
  _su3_inverse_multiply(chi, *u, psi);
  _vector_add_assign((*sp).c1, chi);
  _vector_sub_assign((*sp).c3, chi);

  _vector_sub(psi, rs.s.c2, rs.s.c4);
  _su3_inverse_multiply(chi, *u, psi);
  _vector_add_assign((*sp).c2, chi);
  _vector_sub_assign((*sp).c4, chi);

  /***************************** direction -0 *******************************/

  sm = pl + (*(pidn++));
  u += 1;

  _vector_add(psi, rs.s.c1, rs.s.c3);
  _su3_multiply(chi, *u, psi);
  _vector_add_assign((*sm).c1, chi);
  _vector_add_assign((*sm).c3, chi);

  _vector_add(psi, rs.s.c2, rs.s.c4);
  _su3_multiply(chi, *u, psi);
  _vector_add_assign((*sm).c2, chi);
  _vector_add_assign((*sm).c4, chi);

  /***************************** direction +1 *******************************/

  _vector_mul_assign(rs.s.c1, one_over_gammaf);
  _vector_mul_assign(rs.s.c2, one_over_gammaf);
  _vector_mul_assign(rs.s.c3, one_over_gammaf);
  _vector_mul_assign(rs.s.c4, one_over_gammaf);

  sp = pl + (*(piup++));
  u += 1;

  _vector_i_sub(psi, rs.s.c1, rs.s.c4);
  _su3_inverse_multiply(chi, *u, psi);
  _vector_add_assign((*sp).c1, chi);
  _vector_i_add_assign((*sp).c4, chi);

  _vector_i_sub(psi, rs.s.c2, rs.s.c3);
  _su3_inverse_multiply(chi, *u, psi);
  _vector_add_assign((*sp).c2, chi);
  _vector_i_add_assign((*sp).c3, chi);

  /***************************** direction -1 *******************************/

  sm = pl + (*(pidn++));
  u += 1;

  _vector_i_add(psi, rs.s.c1, rs.s.c4);
  _su3_multiply(chi, *u, psi);
  _vector_add_assign((*sm).c1, chi);
  _vector_i_sub_assign((*sm).c4, chi);

  _vector_i_add(psi, rs.s.c2, rs.s.c3);
  _su3_multiply(chi, *u, psi);
  _vector_add_assign((*sm).c2, chi);
  _vector_i_sub_assign((*sm).c3, chi);

  /***************************** direction +2 *******************************/

  sp = pl + (*(piup++));
  u += 1;

  _vector_sub(psi, rs.s.c1, rs.s.c4);
  _su3_inverse_multiply(chi, *u, psi);
  _vector_add_assign((*sp).c1, chi);
  _vector_sub_assign((*sp).c4, chi);

  _vector_add(psi, rs.s.c2, rs.s.c3);
  _su3_inverse_multiply(chi, *u, psi);
  _vector_add_assign((*sp).c2, chi);
  _vector_add_assign((*sp).c3, chi);

  /***************************** direction -2 *******************************/

  sm = pl + (*(pidn++));
  u += 1;

  _vector_add(psi, rs.s.c1, rs.s.c4);
  _su3_multiply(chi, *u, psi);
  _vector_add_assign((*sm).c1, chi);
  _vector_add_assign((*sm).c4, chi);

  _vector_sub(psi, rs.s.c2, rs.s.c3);
  _su3_multiply(chi, *u, psi);
  _vector_add_assign((*sm).c2, chi);
  _vector_sub_assign((*sm).c3, chi);

  /***************************** direction +3 *******************************/

  sp = pl + (*(piup));
  u += 1;

  _vector_i_sub(psi, rs.s.c1, rs.s.c3);
  _su3_inverse_multiply(chi, *u, psi);
  _vector_add_assign((*sp).c1, chi);
  _vector_i_add_assign((*sp).c3, chi);

  _vector_i_add(psi, rs.s.c2, rs.s.c4);
  _su3_inverse_multiply(chi, *u, psi);
  _vector_add_assign((*sp).c2, chi);
  _vector_i_sub_assign((*sp).c4, chi);

  /***************************** direction -3 *******************************/

  sm = pl + (*(pidn));
  u += 1;

  _vector_i_add(psi, rs.s.c1, rs.s.c3);
  _su3_multiply(chi, *u, psi);
  _vector_add_assign((*sm).c1, chi);
  _vector_i_sub_assign((*sm).c3, chi);

  _vector_i_sub(psi, rs.s.c2, rs.s.c4);
  _su3_multiply(chi, *u, psi);
  _vector_add_assign((*sm).c2, chi);
  _vector_i_add_assign((*sm).c4, chi);
}

#endif

void Dw(float mu, spinor *s, spinor *r)
{
#ifdef dirac_counters
  Dw_counter++;
#endif
  int bc, ix, t;
  int *piup, *pidn;
#if (defined QPX)
  vector4double chid11, chid12, chid13, chid21, chid22, chid23;
#endif
  su3 *u, *um;
  pauli *m;
  spin_t *so, *ro;
  tm_parms_t tm;

  cps_int_bnd(0x1, s);
  m = swfld();
  apply_sw(VOLUME / 2, mu, m, s, r);
  set_s2zero(BNDRY / 2, r + VOLUME);

  tm = tm_parms();
  if (tm.eoflg == 1)
    mu = 0.0f;

  coe = -0.5f;
  ceo = -0.5f;
  set_aniso_hopping_coeffs();

  bc = bc_type();
  piup = iup[VOLUME / 2];
  pidn = idn[VOLUME / 2];

  so = (spin_t *)(s + (VOLUME / 2));
  ro = (spin_t *)(r + (VOLUME / 2));
  m += VOLUME;
  u = ufld();
  um = u + 4 * VOLUME;

  if (((cpr[0] == 0) && (bc != 3)) || ((cpr[0] == (NPROC0 - 1)) && (bc == 0))) {
    ix = VOLUME / 2;

    for (; u < um; u += 8) {
      t = global_time(ix);
      ix += 1;

      if ((t > 0) && ((t < (N0 - 1)) || (bc != 0))) {
        doe(piup, pidn, u, s);

#if (defined QPX)
        mul_pauli(mu, m, (*so).w, (*ro).w);
        mul_pauli(-mu, m + 1, (*so).w + 1, (*ro).w + 1);
        _qpx_load_w1(chid1, &((*ro).s));
        _qpx_load_w2(chid2, &((*ro).s));
        _qpx_vec_add_assign(chid1, dim1);
        _qpx_vec_add_assign(chid2, dim2);
        _qpx_store_w1(chid1, &((*ro).s));
        _qpx_store_w2(chid2, &((*ro).s));
        _qpx_load_w1(dim1, &((*so).s));
        _qpx_load_w2(dim2, &((*so).s));
#else
        mul_pauli2(mu, m, &((*so).s), &((*ro).s));
        _vector_add_assign((*ro).s.c1, rs.s.c1);
        _vector_add_assign((*ro).s.c2, rs.s.c2);
        _vector_add_assign((*ro).s.c3, rs.s.c3);
        _vector_add_assign((*ro).s.c4, rs.s.c4);
        rs = (*so);
#endif

        deo(piup, pidn, u, r);
      } else {
        (*so).s = s0;
        (*ro).s = s0;
      }

      piup += 4;
      pidn += 4;
      so += 1;
      ro += 1;
      m += 2;
    }
  } else {
    for (; u < um; u += 8) {
      doe(piup, pidn, u, s);

      mul_pauli2(mu, m, &((*so).s), &((*ro).s));

#if (defined QPX)
      _qpx_load_w1(chid1, &((*ro).s));
      _qpx_load_w2(chid2, &((*ro).s));
      _qpx_vec_add_assign(chid1, dim1);
      _qpx_vec_add_assign(chid2, dim2);
      _qpx_store_w1(chid1, &((*ro).s));
      _qpx_store_w2(chid2, &((*ro).s));
      _qpx_load_w1(dim1, &((*so).s));
      _qpx_load_w2(dim2, &((*so).s));
#else
      _vector_add_assign((*ro).s.c1, rs.s.c1);
      _vector_add_assign((*ro).s.c2, rs.s.c2);
      _vector_add_assign((*ro).s.c3, rs.s.c3);
      _vector_add_assign((*ro).s.c4, rs.s.c4);
      rs = (*so);
#endif
      deo(piup, pidn, u, r);

      piup += 4;
      pidn += 4;
      so += 1;
      ro += 1;
      m += 2;
    }
  }

  cps_ext_bnd(0x1, r);
}

void Dwee(float mu, spinor *s, spinor *r)
{
#ifdef dirac_counters
  Dwee_counter++;
#endif
  int bc, ix, t;
  pauli *m, *mm;
  spin_t *se, *re;

  bc = bc_type();
  m = swfld();
  mm = m + VOLUME;
  se = (spin_t *)(s);
  re = (spin_t *)(r);

  if (((cpr[0] == 0) && (bc != 3)) || ((cpr[0] == (NPROC0 - 1)) && (bc == 0))) {
    ix = 0;

    for (; m < mm; m += 2) {
      t = global_time(ix);
      ix += 1;

      if ((t > 0) && ((t < (N0 - 1)) || (bc != 0)))
        mul_pauli2(mu, m, &((*se).s), &((*re).s));
      else {
        (*se).s = s0;
        (*re).s = s0;
      }

      se += 1;
      re += 1;
    }
  } else {
    for (; m < mm; m += 2) {
      mul_pauli2(mu, m, &((*se).s), &((*re).s));

      se += 1;
      re += 1;
    }
  }
}

void Dwoo(float mu, spinor *s, spinor *r)
{
#ifdef dirac_counters
  Dwoo_counter++;
#endif
  int bc, ix, t;
  pauli *m, *mm;
  spin_t *so, *ro;
  tm_parms_t tm;

  bc = bc_type();
  m = swfld() + VOLUME;
  mm = m + VOLUME;
  so = (spin_t *)(s + (VOLUME / 2));
  ro = (spin_t *)(r + (VOLUME / 2));
  tm = tm_parms();
  if (tm.eoflg == 1)
    mu = 0.0f;

  if (((cpr[0] == 0) && (bc != 3)) || ((cpr[0] == (NPROC0 - 1)) && (bc == 0))) {
    ix = VOLUME / 2;

    for (; m < mm; m += 2) {
      t = global_time(ix);
      ix += 1;

      if ((t > 0) && ((t < (N0 - 1)) || (bc != 0)))
        mul_pauli2(mu, m, &((*so).s), &((*ro).s));
      else {
        (*so).s = s0;
        (*ro).s = s0;
      }

      so += 1;
      ro += 1;
    }
  } else {
    for (; m < mm; m += 2) {
      mul_pauli2(mu, m, &((*so).s), &((*ro).s));

      so += 1;
      ro += 1;
    }
  }
}

void Dwoe(spinor *s, spinor *r)
{
#ifdef dirac_counters
  Dwoe_counter++;
#endif
  int bc, ix, t;
  int *piup, *pidn;
  su3 *u, *um;
  spin_t *ro;

  cps_int_bnd(0x1, s);

  coe = -0.5f;
  set_aniso_hopping_coeffs();

  bc = bc_type();
  piup = iup[VOLUME / 2];
  pidn = idn[VOLUME / 2];

  ro = (spin_t *)(r + (VOLUME / 2));
  u = ufld();
  um = u + 4 * VOLUME;

  if (((cpr[0] == 0) && (bc != 3)) || ((cpr[0] == (NPROC0 - 1)) && (bc == 0))) {
    ix = VOLUME / 2;

    for (; u < um; u += 8) {
      t = global_time(ix);
      ix += 1;

      if ((t > 0) && ((t < (N0 - 1)) || (bc != 0))) {
        doe(piup, pidn, u, s);
#if (defined QPX)
        _qpx_store_w1(dim1, &((*ro).s));
        _qpx_store_w2(dim2, &((*ro).s));
#else
        (*ro) = rs;
#endif
      } else
        (*ro).s = s0;

      piup += 4;
      pidn += 4;
      ro += 1;
    }
  } else {
    for (; u < um; u += 8) {
      doe(piup, pidn, u, s);
#if (defined QPX)
      _qpx_store_w1(dim1, &((*ro).s));
      _qpx_store_w2(dim2, &((*ro).s));
#else
      (*ro) = rs;
#endif
      piup += 4;
      pidn += 4;
      ro += 1;
    }
  }
}

void Dweo(spinor *s, spinor *r)
{
#ifdef dirac_counters
  Dweo_counter++;
#endif
  int bc, ix, t;
  int *piup, *pidn;
  su3 *u, *um;
  spin_t *so;

  set_s2zero(BNDRY / 2, r + VOLUME);

  ceo = 0.5f;
  set_aniso_hopping_coeffs();

  bc = bc_type();
  piup = iup[VOLUME / 2];
  pidn = idn[VOLUME / 2];

  so = (spin_t *)(s + (VOLUME / 2));
  u = ufld();
  um = u + 4 * VOLUME;

  if (((cpr[0] == 0) && (bc != 3)) || ((cpr[0] == (NPROC0 - 1)) && (bc == 0))) {
    ix = VOLUME / 2;

    for (; u < um; u += 8) {
      t = global_time(ix);
      ix += 1;

      if ((t > 0) && ((t < (N0 - 1)) || (bc != 0))) {
#if (defined QPX)
        _qpx_load_w1(dim1, &((*so).s));
        _qpx_load_w2(dim2, &((*so).s));
#else
        rs = (*so);
#endif
        deo(piup, pidn, u, r);
      } else
        (*so).s = s0;

      piup += 4;
      pidn += 4;
      so += 1;
    }
  } else {
    for (; u < um; u += 8) {
#if (defined QPX)
      _qpx_load_w1(dim1, &((*so).s));
      _qpx_load_w2(dim2, &((*so).s));
#else
      rs = (*so);
#endif
      deo(piup, pidn, u, r);

      piup += 4;
      pidn += 4;
      so += 1;
    }
  }

  cps_ext_bnd(0x1, r);
}

void Dwhat(float mu, spinor *s, spinor *r)
{
#ifdef dirac_counters
  Dwhat_counter++;
#endif
  int bc, ix, t;
  int *piup, *pidn;
  su3 *u, *um;
  pauli *m;

  cps_int_bnd(0x1, s);
  m = swfld();
  apply_sw(VOLUME / 2, mu, m, s, r);
  set_s2zero(BNDRY / 2, r + VOLUME);

  coe = -0.5f;
  ceo = 0.5f;
  set_aniso_hopping_coeffs();

  bc = bc_type();
  piup = iup[VOLUME / 2];
  pidn = idn[VOLUME / 2];

  m += VOLUME;
  u = ufld();
  um = u + 4 * VOLUME;

  if (((cpr[0] == 0) && (bc != 3)) || ((cpr[0] == (NPROC0 - 1)) && (bc == 0))) {
    ix = VOLUME / 2;

    for (; u < um; u += 8) {
      t = global_time(ix);
      ix += 1;

      if ((t > 0) && ((t < (N0 - 1)) || (bc != 0))) {
        doe(piup, pidn, u, s);

#if (defined QPX)
        mul_pauli_qpx(m, im1, im1);
        mul_pauli_qpx(m + 1, im2, im2);
#else
        mul_pauli2(0.0f, m, &(rs.s), &(rs.s));
#endif

        deo(piup, pidn, u, r);
      }

      piup += 4;
      pidn += 4;
      m += 2;
    }
  } else {
    for (; u < um; u += 8) {
      doe(piup, pidn, u, s);

#if (defined QPX)
      mul_pauli_qpx(m, im1, im1);
      mul_pauli_qpx(m + 1, im2, im2);
#else
      mul_pauli2(0.0f, m, &(rs.s), &(rs.s));
#endif

      deo(piup, pidn, u, r);

      piup += 4;
      pidn += 4;
      m += 2;
    }
  }

  cps_ext_bnd(0x1, r);
}

void Dw_blk(blk_grid_t grid, int n, float mu, int k, int l)
{
#ifdef dirac_counters
  Dw_blk_counter++;
#endif
  int nb, isw, vol, volh, ibu, ibd;
  int *piup, *pidn, *ibp, *ibm;
#if (defined QPX)
  vector4double chi11, chi12, chi13, chi21, chi22, chi23;
#endif
  su3 *u, *um;
  pauli *m;
  spinor *s, *r;
  spin_t *so, *ro;
  block_t *b;
  tm_parms_t tm;

  b = blk_list(grid, &nb, &isw);

  if ((n < 0) || (n >= nb)) {
    error_loc(1, 1, "Dw_blk [Dw.c]",
              "Block grid is not allocated or block number out of range");
    return;
  }

  if ((k < 0) || (l < 0) || (k == l) || (k >= (*b).ns) || (l >= (*b).ns) ||
      ((*b).u == NULL)) {
    error_loc(1, 1, "Dw_blk [Dw.c]",
              "Attempt to access unallocated memory space");
    return;
  }

  b += n;
  vol = (*b).vol;
  volh = vol / 2;
  s = (*b).s[k];
  r = (*b).s[l];
  so = (spin_t *)(s + volh);
  ro = (spin_t *)(r + volh);

  s[vol] = s0;
  r[vol] = s0;
  m = (*b).sw;
  apply_sw(volh, mu, m, s, r);
  tm = tm_parms();
  if (tm.eoflg == 1)
    mu = 0.0f;

  coe = -0.5f;
  ceo = -0.5f;
  set_aniso_hopping_coeffs();

  piup = (*b).iup[volh];
  pidn = (*b).idn[volh];
  m += vol;
  u = (*b).u;
  um = u + 4 * vol;

  if ((*b).nbp) {
    ibp = (*b).ibp;
    ibm = ibp + (*b).nbp / 2;

    for (; ibp < ibm; ibp++)
      s[*ibp] = s0;

    ibu = ((cpr[0] == (NPROC0 - 1)) && (((*b).bo[0] + (*b).bs[0]) == L0) &&
           (bc_type() == 0));
    ibd = ((cpr[0] == 0) && ((*b).bo[0] == 0));

    for (; u < um; u += 8) {
      if (((pidn[0] < vol) || (!ibd)) && ((piup[0] < vol) || (!ibu))) {
        doe(piup, pidn, u, s);

#if (defined QPX)
        mul_pauli(mu, m, (*so).w, (*ro).w);
        mul_pauli(-mu, m + 1, (*so).w + 1, (*ro).w + 1);
        _qpx_load_w1(chi1, &((*ro).s));
        _qpx_load_w2(chi2, &((*ro).s));
        _qpx_vec_add_assign(chi1, dim1);
        _qpx_vec_add_assign(chi2, dim2);
        _qpx_store_w1(chi1, &((*ro).s));
        _qpx_store_w2(chi2, &((*ro).s));
        _qpx_load_w1(dim1, &((*so).s));
        _qpx_load_w2(dim2, &((*so).s));
#else
        mul_pauli2(mu, m, &((*so).s), &((*ro).s));
        _vector_add_assign((*ro).s.c1, rs.s.c1);
        _vector_add_assign((*ro).s.c2, rs.s.c2);
        _vector_add_assign((*ro).s.c3, rs.s.c3);
        _vector_add_assign((*ro).s.c4, rs.s.c4);
        rs = (*so);
#endif

        deo(piup, pidn, u, r);
      } else {
        (*so).s = s0;
        (*ro).s = s0;
      }

      piup += 4;
      pidn += 4;
      ro += 1;
      so += 1;
      m += 2;
    }

    ibp = (*b).ibp;
    ibm = ibp + (*b).nbp / 2;

    for (; ibp < ibm; ibp++)
      r[*ibp] = s0;
  } else {
    for (; u < um; u += 8) {
      doe(piup, pidn, u, s);

#if (defined QPX)
      mul_pauli(mu, m, (*so).w, (*ro).w);
      mul_pauli(-mu, m + 1, (*so).w + 1, (*ro).w + 1);
      _qpx_load_w1(chi1, &((*ro).s));
      _qpx_load_w2(chi2, &((*ro).s));
      _qpx_vec_add_assign(chi1, dim1);
      _qpx_vec_add_assign(chi2, dim2);
      _qpx_store_w1(chi1, &((*ro).s));
      _qpx_store_w2(chi2, &((*ro).s));
      _qpx_load_w1(dim1, &((*so).s));
      _qpx_load_w2(dim2, &((*so).s));
#else
      mul_pauli2(mu, m, &((*so).s), &((*ro).s));
      _vector_add_assign((*ro).s.c1, rs.s.c1);
      _vector_add_assign((*ro).s.c2, rs.s.c2);
      _vector_add_assign((*ro).s.c3, rs.s.c3);
      _vector_add_assign((*ro).s.c4, rs.s.c4);
      rs = (*so);
#endif

      deo(piup, pidn, u, r);

      piup += 4;
      pidn += 4;
      ro += 1;
      so += 1;
      m += 2;
    }
  }
}

void Dwee_blk(blk_grid_t grid, int n, float mu, int k, int l)
{
#ifdef dirac_counters
  Dwee_blk_counter++;
#endif
  int nb, isw, vol, ibu, ibd;
  int *piup, *pidn;
  pauli *m, *mm;
  spin_t *se, *re;
  block_t *b;

  b = blk_list(grid, &nb, &isw);

  if ((n < 0) || (n >= nb)) {
    error_loc(1, 1, "Dwee_blk [Dw.c]",
              "Block grid is not allocated or block number out of range");
    return;
  }

  if ((k < 0) || (l < 0) || (k >= (*b).ns) || (l >= (*b).ns) ||
      ((*b).u == NULL)) {
    error_loc(1, 1, "Dwee_blk [Dw.c]",
              "Attempt to access unallocated memory space");
    return;
  }

  b += n;
  vol = (*b).vol;
  se = (spin_t *)((*b).s[k]);
  re = (spin_t *)((*b).s[l]);
  m = (*b).sw;
  mm = m + vol;

  if ((*b).nbp) {
    piup = (*b).iup[0];
    pidn = (*b).idn[0];

    ibu = ((cpr[0] == (NPROC0 - 1)) && (((*b).bo[0] + (*b).bs[0]) == L0) &&
           (bc_type() == 0));
    ibd = ((cpr[0] == 0) && ((*b).bo[0] == 0));

    for (; m < mm; m += 2) {
      if (((pidn[0] < vol) || (!ibd)) && ((piup[0] < vol) || (!ibu)))
        mul_pauli2(mu, m, &((*se).s), &((*re).s));
      else {
        (*se).s = s0;
        (*re).s = s0;
      }

      piup += 4;
      pidn += 4;
      se += 1;
      re += 1;
    }
  } else {
    for (; m < mm; m += 2) {
      mul_pauli2(mu, m, &((*se).s), &((*re).s));

      se += 1;
      re += 1;
    }
  }
}

void Dwoo_blk(blk_grid_t grid, int n, float mu, int k, int l)
{
#ifdef dirac_counters
  Dwoo_blk_counter++;
#endif
  int nb, isw, vol, volh, ibu, ibd;
  int *piup, *pidn;
  pauli *m, *mm;
  spin_t *so, *ro;
  block_t *b;
  tm_parms_t tm;

  b = blk_list(grid, &nb, &isw);

  if ((n < 0) || (n >= nb)) {
    error_loc(1, 1, "Dwoo_blk [Dw.c]",
              "Block grid is not allocated or block number out of range");
    return;
  }

  if ((k < 0) || (l < 0) || (k >= (*b).ns) || (l >= (*b).ns) ||
      ((*b).u == NULL)) {
    error_loc(1, 1, "Dwoo_blk [Dw.c]",
              "Attempt to access unallocated memory space");
    return;
  }

  b += n;
  vol = (*b).vol;
  volh = vol / 2;
  so = (spin_t *)((*b).s[k] + volh);
  ro = (spin_t *)((*b).s[l] + volh);
  tm = tm_parms();
  if (tm.eoflg == 1)
    mu = 0.0f;

  m = (*b).sw + vol;
  mm = m + vol;

  if ((*b).nbp) {
    piup = (*b).iup[volh];
    pidn = (*b).idn[volh];

    ibu = ((cpr[0] == (NPROC0 - 1)) && (((*b).bo[0] + (*b).bs[0]) == L0) &&
           (bc_type() == 0));
    ibd = ((cpr[0] == 0) && ((*b).bo[0] == 0));

    for (; m < mm; m += 2) {
      if (((pidn[0] < vol) || (!ibd)) && ((piup[0] < vol) || (!ibu)))
        mul_pauli2(mu, m, &((*so).s), &((*ro).s));
      else {
        (*so).s = s0;
        (*ro).s = s0;
      }

      piup += 4;
      pidn += 4;
      so += 1;
      ro += 1;
    }
  } else {
    for (; m < mm; m += 2) {
      mul_pauli2(mu, m, &((*so).s), &((*ro).s));

      so += 1;
      ro += 1;
    }
  }
}

void Dwoe_blk(blk_grid_t grid, int n, int k, int l)
{
#ifdef dirac_counters
  Dwoe_blk_counter++;
#endif
  int nb, isw, vol, volh, ibu, ibd;
  int *piup, *pidn, *ibp, *ibm;
#if (defined QPX)
  vector4double chi11, chi12, chi13, chi21, chi22, chi23;
#endif
  su3 *u, *um;
  spinor *s;
  spin_t *ro;
  block_t *b;

  b = blk_list(grid, &nb, &isw);

  if ((n < 0) || (n >= nb)) {
    error_loc(1, 1, "Dwoe_blk [Dw.c]",
              "Block grid is not allocated or block number out of range");
    return;
  }

  if ((k < 0) || (l < 0) || (k >= (*b).ns) || (l >= (*b).ns) ||
      ((*b).u == NULL)) {
    error_loc(1, 1, "Dwoe_blk [Dw.c]",
              "Attempt to access unallocated memory space");
    return;
  }

  b += n;
  vol = (*b).vol;
  volh = vol / 2;
  s = (*b).s[k];
  ro = (spin_t *)((*b).s[l] + volh);
  s[vol] = s0;

  coe = -0.5f;
  set_aniso_hopping_coeffs();

  piup = (*b).iup[volh];
  pidn = (*b).idn[volh];
  u = (*b).u;
  um = u + 4 * vol;

  if ((*b).nbp) {
    ibp = (*b).ibp;
    ibm = ibp + (*b).nbp / 2;

    for (; ibp < ibm; ibp++)
      s[*ibp] = s0;

    ibu = ((cpr[0] == (NPROC0 - 1)) && (((*b).bo[0] + (*b).bs[0]) == L0) &&
           (bc_type() == 0));
    ibd = ((cpr[0] == 0) && ((*b).bo[0] == 0));

    for (; u < um; u += 8) {
      if (((pidn[0] < vol) || (!ibd)) && ((piup[0] < vol) || (!ibu))) {
        doe(piup, pidn, u, s);
#if (defined QPX)
        _qpx_store_w1(dim1, &((*ro).s));
        _qpx_store_w2(dim2, &((*ro).s));
#else
        (*ro) = rs;
#endif
      } else
        (*ro).s = s0;

      piup += 4;
      pidn += 4;
      ro += 1;
    }
  } else {
    for (; u < um; u += 8) {
      doe(piup, pidn, u, s);
#if (defined QPX)
      _qpx_store_w1(dim1, &((*ro).s));
      _qpx_store_w2(dim2, &((*ro).s));
#else
      (*ro) = rs;
#endif
      piup += 4;
      pidn += 4;
      ro += 1;
    }
  }
}

void Dweo_blk(blk_grid_t grid, int n, int k, int l)
{
#ifdef dirac_counters
  Dweo_blk_counter++;
#endif
  int nb, isw, vol, volh, ibu, ibd;
  int *piup, *pidn, *ibp, *ibm;
  su3 *u, *um;
  spinor *r;
  spin_t *so;
  block_t *b;

  b = blk_list(grid, &nb, &isw);

  if ((n < 0) || (n >= nb)) {
    error_loc(1, 1, "Dweo_blk [Dw.c]",
              "Block grid is not allocated or block number out of range");
    return;
  }

  if ((k < 0) || (l < 0) || (k >= (*b).ns) || (l >= (*b).ns) ||
      ((*b).u == NULL)) {
    error_loc(1, 1, "Dweo_blk [Dw.c]",
              "Attempt to access unallocated memory space");
    return;
  }

  b += n;
  vol = (*b).vol;
  volh = vol / 2;
  so = (spin_t *)((*b).s[k] + volh);
  r = (*b).s[l];
  r[vol] = s0;

  ceo = 0.5f;
  set_aniso_hopping_coeffs();

  piup = (*b).iup[volh];
  pidn = (*b).idn[volh];
  u = (*b).u;
  um = u + 4 * vol;

  if ((*b).nbp) {
    ibu = ((cpr[0] == (NPROC0 - 1)) && (((*b).bo[0] + (*b).bs[0]) == L0) &&
           (bc_type() == 0));
    ibd = ((cpr[0] == 0) && ((*b).bo[0] == 0));

    for (; u < um; u += 8) {
      if (((pidn[0] < vol) || (!ibd)) && ((piup[0] < vol) || (!ibu))) {
#if (defined QPX)
        _qpx_load_w1(dim1, &((*so).s));
        _qpx_load_w2(dim2, &((*so).s));
#else
        rs = (*so);
#endif
        deo(piup, pidn, u, r);
      } else
        (*so).s = s0;

      piup += 4;
      pidn += 4;
      so += 1;
    }

    ibp = (*b).ibp;
    ibm = ibp + (*b).nbp / 2;

    for (; ibp < ibm; ibp++)
      r[*ibp] = s0;
  } else {
    for (; u < um; u += 8) {
#if (defined QPX)
      _qpx_load_w1(dim1, &((*so).s));
      _qpx_load_w2(dim2, &((*so).s));
#else
      rs = (*so);
#endif
      deo(piup, pidn, u, r);

      piup += 4;
      pidn += 4;
      so += 1;
    }
  }
}

void Dwhat_blk(blk_grid_t grid, int n, float mu, int k, int l)
{
#ifdef dirac_counters
  Dwhat_blk_counter++;
#endif
  int nb, isw, vol, volh, ibu, ibd;
  int *piup, *pidn, *ibp, *ibm;
  su3 *u, *um;
  pauli *m;
  spinor *s, *r;
  block_t *b;

  b = blk_list(grid, &nb, &isw);

  if ((n < 0) || (n >= nb)) {
    error_loc(1, 1, "Dwhat_blk [Dw.c]",
              "Block grid is not allocated or block number out of range");
    return;
  }

  if ((k < 0) || (l < 0) || (k == l) || (k >= (*b).ns) || (l >= (*b).ns) ||
      ((*b).u == NULL)) {
    error_loc(1, 1, "Dweo_blk [Dw.c]",
              "Attempt to access unallocated memory space");
    return;
  }

  b += n;
  vol = (*b).vol;
  volh = vol / 2;
  s = (*b).s[k];
  r = (*b).s[l];

  s[vol] = s0;
  r[vol] = s0;
  m = (*b).sw;
  apply_sw(volh, mu, m, s, r);

  coe = -0.5f;
  ceo = 0.5f;
  set_aniso_hopping_coeffs();

  piup = (*b).iup[volh];
  pidn = (*b).idn[volh];
  m += vol;
  u = (*b).u;
  um = u + 4 * vol;

  if ((*b).nbp) {
    ibp = (*b).ibp;
    ibm = ibp + (*b).nbp / 2;

    for (; ibp < ibm; ibp++)
      s[*ibp] = s0;

    ibu = ((cpr[0] == (NPROC0 - 1)) && (((*b).bo[0] + (*b).bs[0]) == L0) &&
           (bc_type() == 0));
    ibd = ((cpr[0] == 0) && ((*b).bo[0] == 0));

    for (; u < um; u += 8) {
      if (((pidn[0] < vol) || (!ibd)) && ((piup[0] < vol) || (!ibu))) {
        doe(piup, pidn, u, s);

#if (defined QPX)
        mul_pauli_qpx(m, im1, im1);
        mul_pauli_qpx(m + 1, im2, im2);
#else
        mul_pauli2(0.0f, m, &(rs.s), &(rs.s));
#endif

        deo(piup, pidn, u, r);
      }

      piup += 4;
      pidn += 4;
      m += 2;
    }

    ibp = (*b).ibp;
    ibm = ibp + (*b).nbp / 2;

    for (; ibp < ibm; ibp++)
      r[*ibp] = s0;
  } else {
    for (; u < um; u += 8) {
      doe(piup, pidn, u, s);

#if (defined QPX)
      mul_pauli_qpx(m, im1, im1);
      mul_pauli_qpx(m + 1, im2, im2);
#else
      mul_pauli2(0.0f, m, &(rs.s), &(rs.s));
#endif

      deo(piup, pidn, u, r);

      piup += 4;
      pidn += 4;
      m += 2;
    }
  }
}
