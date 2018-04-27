
/*******************************************************************************
 *
 * File ratfcts.h
 *
 * Copyright (C) 2012 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef RATFCTS_H
#define RATFCTS_H

#include "utils.h"

typedef struct
{
  int np;
  double A, delta;
  double *mu, *rmu;
  double *nu, *rnu;
} openqcd_ratfcts__ratfct_t;

/* ELLIPTIC_C */
extern double openqcd_ratfcts__ellipticK(double rk);
extern void openqcd_ratfcts__sncndn(double u, double rk, double *sn, double *cn,
                                    double *dn);

/* RATFCTS_C */
extern openqcd_ratfcts__ratfct_t openqcd_ratfcts__ratfct(int const *irat);

/* ZOLOTAREV_C */
extern void openqcd_ratfcts__zolotarev(int n, double eps, double *A, double *ar,
                                       double *delta);

#if defined(OPENQCD_INTERNAL)
#define ratfct_t openqcd_ratfcts__ratfct_t

/* ELLIPTIC_C */
#define ellipticK(...) openqcd_ratfcts__ellipticK(__VA_ARGS__)
#define sncndn(...) openqcd_ratfcts__sncndn(__VA_ARGS__)

/* RATFCTS_C */
#define ratfct(...) openqcd_ratfcts__ratfct(__VA_ARGS__)

/* ZOLOTAREV_C */
#define zolotarev(...) openqcd_ratfcts__zolotarev(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif
