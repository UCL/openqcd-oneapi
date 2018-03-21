
/*******************************************************************************
 *
 * File su3fcts.h
 *
 * Copyright (C) 2010, 2011, 2016 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef SU3FCTS_H
#define SU3FCTS_H

#include "su3.h"

typedef struct
{
  double t, d;
  complex_dble p[3] ALIGNED16;
} ch_drv0_t;

typedef struct
{
  double t, d;
  complex_dble p[3] ALIGNED16;
  complex_dble pt[3] ALIGNED16;
  complex_dble pd[3] ALIGNED16;
} ch_drv1_t;

typedef struct
{
  double t, d;
  complex_dble p[3] ALIGNED16;
  complex_dble pt[3] ALIGNED16;
  complex_dble pd[3] ALIGNED16;
  complex_dble ptt[3] ALIGNED16;
  complex_dble ptd[3] ALIGNED16;
  complex_dble pdd[3] ALIGNED16;
} ch_drv2_t;

/* CHEXP_C */
extern void ch2mat(complex_dble const *p, su3_alg_dble const *X, su3_dble *u);
extern void chexp_drv0(su3_alg_dble const *X, ch_drv0_t *s);
extern void chexp_drv1(su3_alg_dble const *X, ch_drv1_t *s);
extern void chexp_drv2(su3_alg_dble const *X, ch_drv2_t *s);
extern void expXsu3(double eps, su3_alg_dble const *X, su3_dble *u);
extern void expXsu3_w_factors(double eps, su3_alg_dble const *X, su3_dble *u,
                              ch_drv0_t *s_in);

/* CM3X3_C */
extern void cm3x3_zero(int vol, su3_dble *u);
extern void cm3x3_unity(int vol, su3_dble *u);
extern void cm3x3_assign(int vol, su3_dble const *u, su3_dble *v);
extern void cm3x3_swap(int vol, su3_dble *u, su3_dble *v);
extern void cm3x3_dagger(su3_dble const *u, su3_dble *v);
extern void cm3x3_tr(su3_dble const *u, su3_dble const *v, complex_dble *tr);
extern void cm3x3_retr(su3_dble const *u, su3_dble const *v, double *tr);
extern void cm3x3_imtr(su3_dble const *u, su3_dble const *v, double *tr);
extern void cm3x3_add(su3_dble const *u, su3_dble *v);
extern void cm3x3_mul_add(su3_dble const *u, su3_dble const *v, su3_dble *w);
extern void cm3x3_mulr(double const *r, su3_dble const *u, su3_dble *v);
extern void cm3x3_mulr_add(double const *r, su3_dble const *u, su3_dble *v);
extern void cm3x3_mulc(complex_dble const *c, su3_dble const *u, su3_dble *v);
extern void cm3x3_mulc_add(complex_dble const *c, su3_dble const *u,
                           su3_dble *v);
extern void cm3x3_lc1(complex_dble const *c, su3_dble const *u, su3_dble *v);
extern void cm3x3_lc2(complex_dble const *c, su3_dble const *u, su3_dble *v);

/* RANDOM_SU3_C */
#ifdef SITERANDOM
extern void random_su3(su3 *u, int ix);
extern void random_su3_dble(su3_dble *u, int ix);
#else
extern void random_su3(su3 *u);
extern void random_su3_dble(su3_dble *u);
#endif

/* SU3REN_C */
extern void project_to_su3(su3 *u);
extern void project_to_su3_dble(su3_dble *u);

/* SU3PROD_C */
extern void su3xsu3(su3_dble const *u, su3_dble const *v, su3_dble *w);
extern void su3dagxsu3(su3_dble const *u, su3_dble const *v, su3_dble *w);
extern void su3xsu3dag(su3_dble const *u, su3_dble const *v, su3_dble *w);
extern void su3dagxsu3dag(su3_dble const *u, su3_dble const *v, su3_dble *w);
extern void su3xu3alg(su3_dble const *u, u3_alg_dble const *X, su3_dble *v);
extern void su3dagxu3alg(su3_dble const *u, u3_alg_dble const *X, su3_dble *v);
extern void u3algxsu3(u3_alg_dble const *X, su3_dble const *u, su3_dble *v);
extern void u3algxsu3dag(u3_alg_dble const *X, su3_dble const *u, su3_dble *v);
extern double prod2su3alg(su3_dble const *u, su3_dble const *v,
                          su3_alg_dble *X);
extern void prod2u3alg(su3_dble const *u, su3_dble const *v, u3_alg_dble *X);
extern void rotate_su3alg(su3_dble const *u, su3_alg_dble *X);
extern void su3xsu3alg(su3_dble const *u, su3_alg_dble const *X, su3_dble *v);
extern void su3algxsu3(su3_alg_dble const *X, su3_dble const *u, su3_dble *v);
extern void su3dagxsu3alg(su3_dble const *u, su3_alg_dble const *X,
                          su3_dble *v);
extern void su3algxsu3dag(su3_alg_dble const *X, su3_dble const *u,
                          su3_dble *v);
extern void su3algxsu3_tr(su3_alg_dble const *X, su3_dble const *u,
                          complex_dble *tr);

#endif
