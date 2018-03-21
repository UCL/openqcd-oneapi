
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
extern void cmat_vec(int n, complex const *a, complex const *v, complex *w);
extern void cmat_vec_assign(int n, complex const *a, complex const *v,
                            complex *w);
extern void cmat_add(int n, complex const *a, complex const *b, complex *c);
extern void cmat_sub(int n, complex const *a, complex const *b, complex *c);
extern void cmat_mul(int n, complex const *a, complex const *b, complex *c);
extern void cmat_dag(int n, complex const *a, complex *b);

/* CMATRIX_DBLE_C */
extern void cmat_vec_dble(int n, complex_dble const *a, complex_dble const *v,
                          complex_dble *w);
extern void cmat_vec_assign_dble(int n, complex_dble const *a,
                                 complex_dble const *v, complex_dble *w);
extern void cmat_add_dble(int n, complex_dble const *a, complex_dble const *b,
                          complex_dble *c);
extern void cmat_sub_dble(int n, complex_dble const *a, complex_dble const *b,
                          complex_dble *c);
extern void cmat_mul_dble(int n, complex_dble const *a, complex_dble const *b,
                          complex_dble *c);
extern void cmat_dag_dble(int n, complex_dble const *a, complex_dble *b);
extern int cmat_inv_dble(int n, complex_dble const *a, complex_dble *b,
                         double *k);

/* LIEALG_C */
extern void random_alg(int vol, su3_alg_dble *X);
extern double norm_square_alg(int vol, int icom, su3_alg_dble const *X);
extern double scalar_prod_alg(int vol, int icom, su3_alg_dble const *X,
                              su3_alg_dble const *Y);
extern void set_alg2zero(int vol, su3_alg_dble *X);
extern void set_ualg2zero(int vol, u3_alg_dble *X);
extern void assign_alg2alg(int vol, su3_alg_dble const *X, su3_alg_dble *Y);
extern void swap_alg(int vol, su3_alg_dble *X, su3_alg_dble *Y);
extern void add_alg(int vol, su3_alg_dble const *X, su3_alg_dble *Y);
extern void muladd_assign_alg(int vol, double r, su3_alg_dble const *X,
                              su3_alg_dble *Y);
extern void project_to_su3alg(su3_dble const *u, su3_alg_dble *X);
extern void su3alg_to_cm3x3(su3_alg_dble const *X, su3_dble *u);

/* SALG_C */
extern complex spinor_prod(int vol, int icom, spinor const *s, spinor const *r);
extern float spinor_prod_re(int vol, int icom, spinor const *s,
                            spinor const *r);
extern float norm_square(int vol, int icom, spinor const *s);
extern void mulc_spinor_add(int vol, spinor *s, spinor const *r, complex z);
extern void mulr_spinor_add(int vol, spinor *s, spinor const *r, float c);
extern void project(int vol, int icom, spinor *s, spinor const *r);
extern void scale(int vol, float c, spinor *s);
extern float normalize(int vol, int icom, spinor *s);
extern void rotate(int vol, int n, spinor **ppk, complex const *v);
extern void mulg5(int vol, spinor *s);
extern void mulmg5(int vol, spinor *s);

/* SALG_DBLE_C */
extern complex_dble spinor_prod_dble(int vol, int icom, spinor_dble const *s,
                                     spinor_dble const *r);
extern double spinor_prod_re_dble(int vol, int icom, spinor_dble const *s,
                                  spinor_dble const *r);
extern complex_dble spinor_prod5_dble(int vol, int icom, spinor_dble const *s,
                                      spinor_dble const *r);
extern double norm_square_dble(int vol, int icom, spinor_dble const *s);
extern void mulc_spinor_add_dble(int vol, spinor_dble *s, spinor_dble const *r,
                                 complex_dble z);
extern void mulr_spinor_add_dble(int vol, spinor_dble *s, spinor_dble const *r,
                                 double c);
extern void combine_spinor_dble(int vol, spinor_dble *s, spinor_dble const *r,
                                double cs, double cr);
extern void project_dble(int vol, int icom, spinor_dble *s,
                         spinor_dble const *r);
extern void scale_dble(int vol, double c, spinor_dble *s);
extern double normalize_dble(int vol, int icom, spinor_dble *s);
extern void rotate_dble(int vol, int n, spinor_dble **ppk,
                        complex_dble const *v);
extern void mulg5_dble(int vol, spinor_dble *s);
extern void mulmg5_dble(int vol, spinor_dble *s);

/* VALG_C */
extern complex vprod(int n, int icom, complex const *v, complex const *w);
extern float vnorm_square(int n, int icom, complex const *v);
extern void mulc_vadd(int n, complex *v, complex const *w, complex z);
extern void vproject(int n, int icom, complex *v, complex const *w);
extern void vscale(int n, float r, complex *v);
extern float vnormalize(int n, int icom, complex *v);
extern void vrotate(int n, int nv, complex **pv, complex const *a);

/* VALG_DBLE_C */
extern complex_dble vprod_dble(int n, int icom, complex_dble const *v,
                               complex_dble const *w);
extern double vnorm_square_dble(int n, int icom, complex_dble const *v);
extern void mulc_vadd_dble(int n, complex_dble *v, complex_dble const *w,
                           complex_dble z);
extern void vproject_dble(int n, int icom, complex_dble *v,
                          complex_dble const *w);
extern void vscale_dble(int n, double r, complex_dble *v);
extern double vnormalize_dble(int n, int icom, complex_dble *v);
extern void vrotate_dble(int n, int nv, complex_dble **pv,
                         complex_dble const *a);

#endif
