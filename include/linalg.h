
/*******************************************************************************
 *
 * File linalg.h
 *
 * Copyright (C) 2011 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef LINALG_H
#define LINALG_H

#include "su3.h"
#include "utils.h"

/* CMATRIX_C */
extern void openqcd_linalg__cmat_vec(int n, openqcd__complex const *a,
                                     openqcd__complex const *v,
                                     openqcd__complex *w);
extern void openqcd_linalg__cmat_vec_assign(int n, openqcd__complex const *a,
                                            openqcd__complex const *v,
                                            openqcd__complex *w);
extern void openqcd_linalg__cmat_add(int n, openqcd__complex const *a,
                                     openqcd__complex const *b,
                                     openqcd__complex *c);
extern void openqcd_linalg__cmat_sub(int n, openqcd__complex const *a,
                                     openqcd__complex const *b,
                                     openqcd__complex *c);
extern void openqcd_linalg__cmat_mul(int n, openqcd__complex const *a,
                                     openqcd__complex const *b,
                                     openqcd__complex *c);
extern void openqcd_linalg__cmat_dag(int n, openqcd__complex const *a,
                                     openqcd__complex *b);

/* CMATRIX_DBLE_C */
extern void openqcd_linalg__cmat_vec_dble(int n, openqcd__complex_dble const *a,
                                          openqcd__complex_dble const *v,
                                          openqcd__complex_dble *w);
extern void openqcd_linalg__cmat_vec_assign_dble(int n,
                                                 openqcd__complex_dble const *a,
                                                 openqcd__complex_dble const *v,
                                                 openqcd__complex_dble *w);
extern void openqcd_linalg__cmat_add_dble(int n, openqcd__complex_dble const *a,
                                          openqcd__complex_dble const *b,
                                          openqcd__complex_dble *c);
extern void openqcd_linalg__cmat_sub_dble(int n, openqcd__complex_dble const *a,
                                          openqcd__complex_dble const *b,
                                          openqcd__complex_dble *c);
extern void openqcd_linalg__cmat_mul_dble(int n, openqcd__complex_dble const *a,
                                          openqcd__complex_dble const *b,
                                          openqcd__complex_dble *c);
extern void openqcd_linalg__cmat_dag_dble(int n, openqcd__complex_dble const *a,
                                          openqcd__complex_dble *b);
extern int openqcd_linalg__cmat_inv_dble(int n, openqcd__complex_dble const *a,
                                         openqcd__complex_dble *b, double *k);

/* LIEALG_C */
extern void openqcd_linalg__random_alg(int vol, openqcd__su3_alg_dble *X);
extern double openqcd_linalg__norm_square_alg(int vol, int icom,
                                              openqcd__su3_alg_dble const *X);
extern double openqcd_linalg__scalar_prod_alg(int vol, int icom,
                                              openqcd__su3_alg_dble const *X,
                                              openqcd__su3_alg_dble const *Y);
extern void openqcd_linalg__set_alg2zero(int vol, openqcd__su3_alg_dble *X);
extern void openqcd_linalg__set_ualg2zero(int vol, openqcd__u3_alg_dble *X);
extern void openqcd_linalg__assign_alg2alg(int vol,
                                           openqcd__su3_alg_dble const *X,
                                           openqcd__su3_alg_dble *Y);
extern void openqcd_linalg__swap_alg(int vol, openqcd__su3_alg_dble *X,
                                     openqcd__su3_alg_dble *Y);
extern void openqcd_linalg__add_alg(int vol, openqcd__su3_alg_dble const *X,
                                    openqcd__su3_alg_dble *Y);
extern void openqcd_linalg__muladd_assign_alg(int vol, double r,
                                              openqcd__su3_alg_dble const *X,
                                              openqcd__su3_alg_dble *Y);
extern void openqcd_linalg__project_to_su3alg(openqcd__su3_dble const *u,
                                              openqcd__su3_alg_dble *X);
extern void openqcd_linalg__su3alg_to_cm3x3(openqcd__su3_alg_dble const *X,
                                            openqcd__su3_dble *u);

/* SALG_C */
extern openqcd__complex openqcd_linalg__spinor_prod(int vol, int icom,
                                                    openqcd__spinor const *s,
                                                    openqcd__spinor const *r);
extern float openqcd_linalg__spinor_prod_re(int vol, int icom,
                                            openqcd__spinor const *s,
                                            openqcd__spinor const *r);
extern float openqcd_linalg__norm_square(int vol, int icom,
                                         openqcd__spinor const *s);
extern void openqcd_linalg__mulc_spinor_add(int vol, openqcd__spinor *s,
                                            openqcd__spinor const *r,
                                            openqcd__complex z);
extern void openqcd_linalg__mulr_spinor_add(int vol, openqcd__spinor *s,
                                            openqcd__spinor const *r, float c);
extern void openqcd_linalg__project(int vol, int icom, openqcd__spinor *s,
                                    openqcd__spinor const *r);
extern void openqcd_linalg__scale(int vol, float c, openqcd__spinor *s);
extern float openqcd_linalg__normalize(int vol, int icom, openqcd__spinor *s);
extern void openqcd_linalg__rotate(int vol, int n, openqcd__spinor **ppk,
                                   openqcd__complex const *v);
extern void openqcd_linalg__mulg5(int vol, openqcd__spinor *s);
extern void openqcd_linalg__mulmg5(int vol, openqcd__spinor *s);

/* SALG_DBLE_C */
extern openqcd__complex_dble
openqcd_linalg__spinor_prod_dble(int vol, int icom,
                                 openqcd__spinor_dble const *s,
                                 openqcd__spinor_dble const *r);
extern double
openqcd_linalg__spinor_prod_re_dble(int vol, int icom,
                                    openqcd__spinor_dble const *s,
                                    openqcd__spinor_dble const *r);
extern openqcd__complex_dble
openqcd_linalg__spinor_prod5_dble(int vol, int icom,
                                  openqcd__spinor_dble const *s,
                                  openqcd__spinor_dble const *r);
extern double openqcd_linalg__norm_square_dble(int vol, int icom,
                                               openqcd__spinor_dble const *s);
extern void openqcd_linalg__mulr_spinor_assign_dble(
    int vol, openqcd__spinor_dble *s, openqcd__spinor_dble const *r, double c);
extern void openqcd_linalg__mulc_spinor_add_dble(int vol,
                                                 openqcd__spinor_dble *s,
                                                 openqcd__spinor_dble const *r,
                                                 openqcd__complex_dble z);
extern void openqcd_linalg__mulr_spinor_add_dble(int vol,
                                                 openqcd__spinor_dble *s,
                                                 openqcd__spinor_dble const *r,
                                                 double c);
extern void openqcd_linalg__combine_spinor_dble(int vol,
                                                openqcd__spinor_dble *s,
                                                openqcd__spinor_dble const *r,
                                                double cs, double cr);
extern void openqcd_linalg__project_dble(int vol, int icom,
                                         openqcd__spinor_dble *s,
                                         openqcd__spinor_dble const *r);
extern void openqcd_linalg__scale_dble(int vol, double c,
                                       openqcd__spinor_dble *s);
extern double openqcd_linalg__normalize_dble(int vol, int icom,
                                             openqcd__spinor_dble *s);
extern void openqcd_linalg__rotate_dble(int vol, int n,
                                        openqcd__spinor_dble **ppk,
                                        openqcd__complex_dble const *v);
extern void openqcd_linalg__mulg5_dble(int vol, openqcd__spinor_dble *s);
extern void openqcd_linalg__mulmg5_dble(int vol, openqcd__spinor_dble *s);

/* VALG_C */
extern openqcd__complex openqcd_linalg__vprod(int n, int icom,
                                              openqcd__complex const *v,
                                              openqcd__complex const *w);
extern float openqcd_linalg__vnorm_square(int n, int icom,
                                          openqcd__complex const *v);
extern void openqcd_linalg__mulc_vadd(int n, openqcd__complex *v,
                                      openqcd__complex const *w,
                                      openqcd__complex z);
extern void openqcd_linalg__vproject(int n, int icom, openqcd__complex *v,
                                     openqcd__complex const *w);
extern void openqcd_linalg__vscale(int n, float r, openqcd__complex *v);
extern float openqcd_linalg__vnormalize(int n, int icom, openqcd__complex *v);
extern void openqcd_linalg__vrotate(int n, int nv, openqcd__complex **pv,
                                    openqcd__complex const *a);

/* VALG_DBLE_C */
extern openqcd__complex_dble
openqcd_linalg__vprod_dble(int n, int icom, openqcd__complex_dble const *v,
                           openqcd__complex_dble const *w);
extern double openqcd_linalg__vnorm_square_dble(int n, int icom,
                                                openqcd__complex_dble const *v);
extern void openqcd_linalg__mulc_vadd_dble(int n, openqcd__complex_dble *v,
                                           openqcd__complex_dble const *w,
                                           openqcd__complex_dble z);
extern void openqcd_linalg__vproject_dble(int n, int icom,
                                          openqcd__complex_dble *v,
                                          openqcd__complex_dble const *w);
extern void openqcd_linalg__vscale_dble(int n, double r,
                                        openqcd__complex_dble *v);
extern double openqcd_linalg__vnormalize_dble(int n, int icom,
                                              openqcd__complex_dble *v);
extern void openqcd_linalg__vrotate_dble(int n, int nv,
                                         openqcd__complex_dble **pv,
                                         openqcd__complex_dble const *a);

#if defined(OPENQCD_INTERNAL)
/* CMATRIX_C */
#define cmat_vec(...) openqcd_linalg__cmat_vec(__VA_ARGS__)
#define cmat_vec_assign(...) openqcd_linalg__cmat_vec_assign(__VA_ARGS__)
#define cmat_add(...) openqcd_linalg__cmat_add(__VA_ARGS__)
#define cmat_sub(...) openqcd_linalg__cmat_sub(__VA_ARGS__)
#define cmat_mul(...) openqcd_linalg__cmat_mul(__VA_ARGS__)
#define cmat_dag(...) openqcd_linalg__cmat_dag(__VA_ARGS__)

/* CMATRIX_DBLE_C */
#define cmat_vec_dble(...) openqcd_linalg__cmat_vec_dble(__VA_ARGS__)
#define cmat_vec_assign_dble(...)                                              \
  openqcd_linalg__cmat_vec_assign_dble(__VA_ARGS__)
#define cmat_add_dble(...) openqcd_linalg__cmat_add_dble(__VA_ARGS__)
#define cmat_sub_dble(...) openqcd_linalg__cmat_sub_dble(__VA_ARGS__)
#define cmat_mul_dble(...) openqcd_linalg__cmat_mul_dble(__VA_ARGS__)
#define cmat_dag_dble(...) openqcd_linalg__cmat_dag_dble(__VA_ARGS__)
#define cmat_inv_dble(...) openqcd_linalg__cmat_inv_dble(__VA_ARGS__)

/* LIEALG_C */
#define random_alg(...) openqcd_linalg__random_alg(__VA_ARGS__)
#define norm_square_alg(...) openqcd_linalg__norm_square_alg(__VA_ARGS__)
#define scalar_prod_alg(...) openqcd_linalg__scalar_prod_alg(__VA_ARGS__)
#define set_alg2zero(...) openqcd_linalg__set_alg2zero(__VA_ARGS__)
#define set_ualg2zero(...) openqcd_linalg__set_ualg2zero(__VA_ARGS__)
#define assign_alg2alg(...) openqcd_linalg__assign_alg2alg(__VA_ARGS__)
#define swap_alg(...) openqcd_linalg__swap_alg(__VA_ARGS__)
#define add_alg(...) openqcd_linalg__add_alg(__VA_ARGS__)
#define muladd_assign_alg(...) openqcd_linalg__muladd_assign_alg(__VA_ARGS__)
#define project_to_su3alg(...) openqcd_linalg__project_to_su3alg(__VA_ARGS__)
#define su3alg_to_cm3x3(...) openqcd_linalg__su3alg_to_cm3x3(__VA_ARGS__)

/* SALG_C */
#define spinor_prod(...) openqcd_linalg__spinor_prod(__VA_ARGS__)
#define spinor_prod_re(...) openqcd_linalg__spinor_prod_re(__VA_ARGS__)
#define norm_square(...) openqcd_linalg__norm_square(__VA_ARGS__)
#define mulc_spinor_add(...) openqcd_linalg__mulc_spinor_add(__VA_ARGS__)
#define mulr_spinor_add(...) openqcd_linalg__mulr_spinor_add(__VA_ARGS__)
#define project(...) openqcd_linalg__project(__VA_ARGS__)
#define scale(...) openqcd_linalg__scale(__VA_ARGS__)
#define normalize(...) openqcd_linalg__normalize(__VA_ARGS__)
#define rotate(...) openqcd_linalg__rotate(__VA_ARGS__)
#define mulg5(...) openqcd_linalg__mulg5(__VA_ARGS__)
#define mulmg5(...) openqcd_linalg__mulmg5(__VA_ARGS__)

/* SALG_DBLE_C */
#define spinor_prod_dble(...) openqcd_linalg__spinor_prod_dble(__VA_ARGS__)
#define spinor_prod_re_dble(...)                                               \
  openqcd_linalg__spinor_prod_re_dble(__VA_ARGS__)
#define spinor_prod5_dble(...) openqcd_linalg__spinor_prod5_dble(__VA_ARGS__)
#define norm_square_dble(...) openqcd_linalg__norm_square_dble(__VA_ARGS__)
#define mulr_spinor_assign_dble(...)                                           \
  openqcd_linalg__mulr_spinor_assign_dble(__VA_ARGS__)
#define mulc_spinor_add_dble(...)                                              \
  openqcd_linalg__mulc_spinor_add_dble(__VA_ARGS__)
#define mulr_spinor_add_dble(...)                                              \
  openqcd_linalg__mulr_spinor_add_dble(__VA_ARGS__)
#define combine_spinor_dble(...)                                               \
  openqcd_linalg__combine_spinor_dble(__VA_ARGS__)
#define project_dble(...) openqcd_linalg__project_dble(__VA_ARGS__)
#define scale_dble(...) openqcd_linalg__scale_dble(__VA_ARGS__)
#define normalize_dble(...) openqcd_linalg__normalize_dble(__VA_ARGS__)
#define rotate_dble(...) openqcd_linalg__rotate_dble(__VA_ARGS__)
#define mulg5_dble(...) openqcd_linalg__mulg5_dble(__VA_ARGS__)
#define mulmg5_dble(...) openqcd_linalg__mulmg5_dble(__VA_ARGS__)

/* VALG_C */
#define vprod(...) openqcd_linalg__vprod(__VA_ARGS__)
#define vnorm_square(...) openqcd_linalg__vnorm_square(__VA_ARGS__)
#define mulc_vadd(...) openqcd_linalg__mulc_vadd(__VA_ARGS__)
#define vproject(...) openqcd_linalg__vproject(__VA_ARGS__)
#define vscale(...) openqcd_linalg__vscale(__VA_ARGS__)
#define vnormalize(...) openqcd_linalg__vnormalize(__VA_ARGS__)
#define vrotate(...) openqcd_linalg__vrotate(__VA_ARGS__)

/* VALG_DBLE_C */
#define vprod_dble(...) openqcd_linalg__vprod_dble(__VA_ARGS__)
#define vnorm_square_dble(...) openqcd_linalg__vnorm_square_dble(__VA_ARGS__)
#define mulc_vadd_dble(...) openqcd_linalg__mulc_vadd_dble(__VA_ARGS__)
#define vproject_dble(...) openqcd_linalg__vproject_dble(__VA_ARGS__)
#define vscale_dble(...) openqcd_linalg__vscale_dble(__VA_ARGS__)
#define vnormalize_dble(...) openqcd_linalg__vnormalize_dble(__VA_ARGS__)
#define vrotate_dble(...) openqcd_linalg__vrotate_dble(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif
