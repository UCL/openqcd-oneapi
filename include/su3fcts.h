
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
  openqcd__complex_dble p[3] ALIGNED16;
} openqcd_su3fcts__ch_drv0_t;

typedef struct
{
  double t, d;
  openqcd__complex_dble p[3] ALIGNED16;
  openqcd__complex_dble pt[3] ALIGNED16;
  openqcd__complex_dble pd[3] ALIGNED16;
} openqcd_su3fcts__ch_drv1_t;

typedef struct
{
  double t, d;
  openqcd__complex_dble p[3] ALIGNED16;
  openqcd__complex_dble pt[3] ALIGNED16;
  openqcd__complex_dble pd[3] ALIGNED16;
  openqcd__complex_dble ptt[3] ALIGNED16;
  openqcd__complex_dble ptd[3] ALIGNED16;
  openqcd__complex_dble pdd[3] ALIGNED16;
} openqcd_su3fcts__ch_drv2_t;

/* CHEXP_C */
extern void openqcd_su3fcts__ch2mat(openqcd__complex_dble const *p,
                                    openqcd__su3_alg_dble const *X,
                                    openqcd__su3_dble *u);
extern void openqcd_su3fcts__chexp_drv0(openqcd__su3_alg_dble const *X,
                                        openqcd_su3fcts__ch_drv0_t *s);
extern void openqcd_su3fcts__chexp_drv1(openqcd__su3_alg_dble const *X,
                                        openqcd_su3fcts__ch_drv1_t *s);
extern void openqcd_su3fcts__chexp_drv2(openqcd__su3_alg_dble const *X,
                                        openqcd_su3fcts__ch_drv2_t *s);
extern void openqcd_su3fcts__expXsu3(double eps, openqcd__su3_alg_dble const *X,
                                     openqcd__su3_dble *u);
extern void
openqcd_su3fcts__expXsu3_w_factors(double eps, openqcd__su3_alg_dble const *X,
                                   openqcd__su3_dble *u,
                                   openqcd_su3fcts__ch_drv0_t *s_in);

/* CM3X3_C */
extern void openqcd_su3fcts__cm3x3_zero(int vol, openqcd__su3_dble *u);
extern void openqcd_su3fcts__cm3x3_unity(int vol, openqcd__su3_dble *u);
extern void openqcd_su3fcts__cm3x3_assign(int vol, openqcd__su3_dble const *u,
                                          openqcd__su3_dble *v);
extern void openqcd_su3fcts__cm3x3_swap(int vol, openqcd__su3_dble *u,
                                        openqcd__su3_dble *v);
extern void openqcd_su3fcts__cm3x3_dagger(openqcd__su3_dble const *u,
                                          openqcd__su3_dble *v);
extern void openqcd_su3fcts__cm3x3_tr(openqcd__su3_dble const *u,
                                      openqcd__su3_dble const *v,
                                      openqcd__complex_dble *tr);
extern void openqcd_su3fcts__cm3x3_retr(openqcd__su3_dble const *u,
                                        openqcd__su3_dble const *v, double *tr);
extern void openqcd_su3fcts__cm3x3_imtr(openqcd__su3_dble const *u,
                                        openqcd__su3_dble const *v, double *tr);
extern void openqcd_su3fcts__cm3x3_add(openqcd__su3_dble const *u,
                                       openqcd__su3_dble *v);
extern void openqcd_su3fcts__cm3x3_mul_add(openqcd__su3_dble const *u,
                                           openqcd__su3_dble const *v,
                                           openqcd__su3_dble *w);
extern void openqcd_su3fcts__cm3x3_mulr(double const *r,
                                        openqcd__su3_dble const *u,
                                        openqcd__su3_dble *v);
extern void openqcd_su3fcts__cm3x3_mulr_add(double const *r,
                                            openqcd__su3_dble const *u,
                                            openqcd__su3_dble *v);
extern void openqcd_su3fcts__cm3x3_mulc(openqcd__complex_dble const *c,
                                        openqcd__su3_dble const *u,
                                        openqcd__su3_dble *v);
extern void openqcd_su3fcts__cm3x3_mulc_add(openqcd__complex_dble const *c,
                                            openqcd__su3_dble const *u,
                                            openqcd__su3_dble *v);
extern void openqcd_su3fcts__cm3x3_lc1(openqcd__complex_dble const *c,
                                       openqcd__su3_dble const *u,
                                       openqcd__su3_dble *v);
extern void openqcd_su3fcts__cm3x3_lc2(openqcd__complex_dble const *c,
                                       openqcd__su3_dble const *u,
                                       openqcd__su3_dble *v);

/* RANDOM_SU3_C */
#ifdef SITERANDOM
extern void openqcd_su3fcts__random_su3(openqcd__su3 *u, int ix);
extern void openqcd_su3fcts__random_su3_dble(openqcd__su3_dble *u, int ix);
#else
extern void openqcd_su3fcts__random_su3(openqcd__su3 *u);
extern void openqcd_su3fcts__random_su3_dble(openqcd__su3_dble *u);
#endif

/* SU3REN_C */
extern void openqcd_su3fcts__project_to_su3(openqcd__su3 *u);
extern void openqcd_su3fcts__project_to_su3_dble(openqcd__su3_dble *u);

/* SU3PROD_C */
extern void openqcd_su3fcts__su3xsu3(openqcd__su3_dble const *u,
                                     openqcd__su3_dble const *v,
                                     openqcd__su3_dble *w);
extern void openqcd_su3fcts__su3dagxsu3(openqcd__su3_dble const *u,
                                        openqcd__su3_dble const *v,
                                        openqcd__su3_dble *w);
extern void openqcd_su3fcts__su3xsu3dag(openqcd__su3_dble const *u,
                                        openqcd__su3_dble const *v,
                                        openqcd__su3_dble *w);
extern void openqcd_su3fcts__su3dagxsu3dag(openqcd__su3_dble const *u,
                                           openqcd__su3_dble const *v,
                                           openqcd__su3_dble *w);
extern void openqcd_su3fcts__su3xu3alg(openqcd__su3_dble const *u,
                                       openqcd__u3_alg_dble const *X,
                                       openqcd__su3_dble *v);
extern void openqcd_su3fcts__su3dagxu3alg(openqcd__su3_dble const *u,
                                          openqcd__u3_alg_dble const *X,
                                          openqcd__su3_dble *v);
extern void openqcd_su3fcts__u3algxsu3(openqcd__u3_alg_dble const *X,
                                       openqcd__su3_dble const *u,
                                       openqcd__su3_dble *v);
extern void openqcd_su3fcts__u3algxsu3dag(openqcd__u3_alg_dble const *X,
                                          openqcd__su3_dble const *u,
                                          openqcd__su3_dble *v);
extern double openqcd_su3fcts__prod2su3alg(openqcd__su3_dble const *u,
                                           openqcd__su3_dble const *v,
                                           openqcd__su3_alg_dble *X);
extern void openqcd_su3fcts__prod2u3alg(openqcd__su3_dble const *u,
                                        openqcd__su3_dble const *v,
                                        openqcd__u3_alg_dble *X);
extern void openqcd_su3fcts__rotate_su3alg(openqcd__su3_dble const *u,
                                           openqcd__su3_alg_dble *X);
extern void openqcd_su3fcts__su3xsu3alg(openqcd__su3_dble const *u,
                                        openqcd__su3_alg_dble const *X,
                                        openqcd__su3_dble *v);
extern void openqcd_su3fcts__su3algxsu3(openqcd__su3_alg_dble const *X,
                                        openqcd__su3_dble const *u,
                                        openqcd__su3_dble *v);
extern void openqcd_su3fcts__su3dagxsu3alg(openqcd__su3_dble const *u,
                                           openqcd__su3_alg_dble const *X,
                                           openqcd__su3_dble *v);
extern void openqcd_su3fcts__su3algxsu3dag(openqcd__su3_alg_dble const *X,
                                           openqcd__su3_dble const *u,
                                           openqcd__su3_dble *v);
extern void openqcd_su3fcts__su3algxsu3_tr(openqcd__su3_alg_dble const *X,
                                           openqcd__su3_dble const *u,
                                           openqcd__complex_dble *tr);

#if defined(OPENQCD_INTERNAL)
#define ch_drv0_t openqcd_su3fcts__ch_drv0_t
#define ch_drv1_t openqcd_su3fcts__ch_drv1_t
#define ch_drv2_t openqcd_su3fcts__ch_drv2_t

/* CHEXP_C */
#define ch2mat(...) openqcd_su3fcts__ch2mat(__VA_ARGS__)
#define chexp_drv0(...) openqcd_su3fcts__chexp_drv0(__VA_ARGS__)
#define chexp_drv1(...) openqcd_su3fcts__chexp_drv1(__VA_ARGS__)
#define chexp_drv2(...) openqcd_su3fcts__chexp_drv2(__VA_ARGS__)
#define expXsu3(...) openqcd_su3fcts__expXsu3(__VA_ARGS__)
#define expXsu3_w_factors(...) openqcd_su3fcts__expXsu3_w_factors(__VA_ARGS__)

/* CM3X3_C */
#define cm3x3_zero(...) openqcd_su3fcts__cm3x3_zero(__VA_ARGS__)
#define cm3x3_unity(...) openqcd_su3fcts__cm3x3_unity(__VA_ARGS__)
#define cm3x3_assign(...) openqcd_su3fcts__cm3x3_assign(__VA_ARGS__)
#define cm3x3_swap(...) openqcd_su3fcts__cm3x3_swap(__VA_ARGS__)
#define cm3x3_dagger(...) openqcd_su3fcts__cm3x3_dagger(__VA_ARGS__)
#define cm3x3_tr(...) openqcd_su3fcts__cm3x3_tr(__VA_ARGS__)
#define cm3x3_retr(...) openqcd_su3fcts__cm3x3_retr(__VA_ARGS__)
#define cm3x3_imtr(...) openqcd_su3fcts__cm3x3_imtr(__VA_ARGS__)
#define cm3x3_add(...) openqcd_su3fcts__cm3x3_add(__VA_ARGS__)
#define cm3x3_mul_add(...) openqcd_su3fcts__cm3x3_mul_add(__VA_ARGS__)
#define cm3x3_mulr(...) openqcd_su3fcts__cm3x3_mulr(__VA_ARGS__)
#define cm3x3_mulr_add(...) openqcd_su3fcts__cm3x3_mulr_add(__VA_ARGS__)
#define cm3x3_mulc(...) openqcd_su3fcts__cm3x3_mulc(__VA_ARGS__)
#define cm3x3_mulc_add(...) openqcd_su3fcts__cm3x3_mulc_add(__VA_ARGS__)
#define cm3x3_lc1(...) openqcd_su3fcts__cm3x3_lc1(__VA_ARGS__)
#define cm3x3_lc2(...) openqcd_su3fcts__cm3x3_lc2(__VA_ARGS__)

/* RANDOM_SU3_C */
#define random_su3(...) openqcd_su3fcts__random_su3(__VA_ARGS__)
#define random_su3_dble(...) openqcd_su3fcts__random_su3_dble(__VA_ARGS__)

/* SU3REN_C */
#define project_to_su3(...) openqcd_su3fcts__project_to_su3(__VA_ARGS__)
#define project_to_su3_dble(...)                                               \
  openqcd_su3fcts__project_to_su3_dble(__VA_ARGS__)

/* SU3PROD_C */
#define su3xsu3(...) openqcd_su3fcts__su3xsu3(__VA_ARGS__)
#define su3dagxsu3(...) openqcd_su3fcts__su3dagxsu3(__VA_ARGS__)
#define su3xsu3dag(...) openqcd_su3fcts__su3xsu3dag(__VA_ARGS__)
#define su3dagxsu3dag(...) openqcd_su3fcts__su3dagxsu3dag(__VA_ARGS__)
#define su3xu3alg(...) openqcd_su3fcts__su3xu3alg(__VA_ARGS__)
#define su3dagxu3alg(...) openqcd_su3fcts__su3dagxu3alg(__VA_ARGS__)
#define u3algxsu3(...) openqcd_su3fcts__u3algxsu3(__VA_ARGS__)
#define u3algxsu3dag(...) openqcd_su3fcts__u3algxsu3dag(__VA_ARGS__)
#define prod2su3alg(...) openqcd_su3fcts__prod2su3alg(__VA_ARGS__)
#define prod2u3alg(...) openqcd_su3fcts__prod2u3alg(__VA_ARGS__)
#define rotate_su3alg(...) openqcd_su3fcts__rotate_su3alg(__VA_ARGS__)
#define su3xsu3alg(...) openqcd_su3fcts__su3xsu3alg(__VA_ARGS__)
#define su3algxsu3(...) openqcd_su3fcts__su3algxsu3(__VA_ARGS__)
#define su3dagxsu3alg(...) openqcd_su3fcts__su3dagxsu3alg(__VA_ARGS__)
#define su3algxsu3dag(...) openqcd_su3fcts__su3algxsu3dag(__VA_ARGS__)
#define su3algxsu3_tr(...) openqcd_su3fcts__su3algxsu3_tr(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif
