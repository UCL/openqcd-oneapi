
/*******************************************************************************
 *
 * File nompi/extras.h
 *
 * Copyright (C) 2009, 2010, 2011 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef EXTRAS_H
#define EXTRAS_H

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* CHEBYSHEV_C */
extern int openqcd_nompi_extras__cheby_fit(double a, double b,
                                           double (*f)(double x), int dmax,
                                           double eps, double c[]);
extern double openqcd_nompi_extras__cheby_int(double a, double b,
                                              double (*f)(double x), int dmax,
                                              double eps);
extern double openqcd_nompi_extras__cheby_val(double a, double b, int n,
                                              double c[], double x);

/* FSOLVE_C */
extern double openqcd_nompi_extras__inverse_fct(double x1, double x2,
                                                double (*f)(double x), double y,
                                                double omega1, double omega2);
extern double openqcd_nompi_extras__minimize_fct(double x0, double x1,
                                                 double x2,
                                                 double (*f)(double x),
                                                 double omega1, double omega2);
extern void openqcd_nompi_extras__powell(int n, double *x0, double *x1,
                                         double *x2,
                                         double (*f)(int n, double *x), int imx,
                                         double omega1, double omega2,
                                         double *xmin, int *status);

/* I0M_C */
extern double openqcd_nompi_extras__i0m(double x);

/* KS_TEST_C */
extern void openqcd_nompi_extras__ks_test(int n, double f[], double *pkp,
                                          double *pkm);
extern void openqcd_nompi_extras__ks_prob(int n, double kp, double km,
                                          double *pp, double *pm);

/* PCHI_SQUARE_C */
extern double openqcd_nompi_extras__pchi_square(double chi_square, int nu);

/* STAT_C */
extern double openqcd_nompi_extras__average(int n, double *a);
extern double openqcd_nompi_extras__sigma0(int n, double *a);
extern double openqcd_nompi_extras__auto_corr(int n, double *a, int tmax,
                                              double *g);
extern void openqcd_nompi_extras__sigma_auto_corr(int n, double *a, int tmax,
                                                  int lambda, double *eg);
extern double openqcd_nompi_extras__tauint(int n, double *a, int tmax,
                                           int lambda, int *w, double *sigma);
extern double openqcd_nompi_extras__print_auto(int n, double *a);
extern double openqcd_nompi_extras__jack_err(int nx, int n, double **a,
                                             double (*f)(int nx, double *x),
                                             int bmax, double *sig);
extern double openqcd_nompi_extras__print_jack(int nx, int n, double **a,
                                               double (*f)(int nx, double *x));

#if defined(OPENQCD_INTERNAL)
/* CHEBYSHEV_C */
#define cheby_fit(...) openqcd_nompi_extras__cheby_fit(__VA_ARGS__)
#define cheby_int(...) openqcd_nompi_extras__cheby_int(__VA_ARGS__)
#define cheby_val(...) openqcd_nompi_extras__cheby_val(__VA_ARGS__)

/* FSOLVE_C */
#define inverse_fct(...) openqcd_nompi_extras__inverse_fct(__VA_ARGS__)
#define minimize_fct(...) openqcd_nompi_extras__minimize_fct(__VA_ARGS__)
#define powell(...) openqcd_nompi_extras__powell(__VA_ARGS__)

/* I0M_C */
#define i0m(...) openqcd_nompi_extras__i0m(__VA_ARGS__)

/* KS_TEST_C */
#define ks_test(...) openqcd_nompi_extras__ks_test(__VA_ARGS__)
#define ks_prob(...) openqcd_nompi_extras__ks_prob(__VA_ARGS__)

/* PCHI_SQUARE_C */
#define pchi_square(...) openqcd_nompi_extras__pchi_square(__VA_ARGS__)

/* STAT_C */
#define average(...) openqcd_nompi_extras__average(__VA_ARGS__)
#define sigma0(...) openqcd_nompi_extras__sigma0(__VA_ARGS__)
#define auto_corr(...) openqcd_nompi_extras__auto_corr(__VA_ARGS__)
#define sigma_auto_corr(...) openqcd_nompi_extras__sigma_auto_corr(__VA_ARGS__)
#define tauint(...) openqcd_nompi_extras__tauint(__VA_ARGS__)
#define print_auto(...) openqcd_nompi_extras__print_auto(__VA_ARGS__)
#define jack_err(...) openqcd_nompi_extras__jack_err(__VA_ARGS__)
#define print_jack(...) openqcd_nompi_extras__print_jack(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif
