
/*******************************************************************************
 *
 * File linsolv.h
 *
 * Copyright (C) 2011 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef LINSOLV_H
#define LINSOLV_H

#include "su3.h"

/* CGNE_C */
extern double openqcd_linsolv__cgne(
    int vol, int icom, void (*Dop)(openqcd__spinor *s, openqcd__spinor *r),
    void (*Dop_dble)(openqcd__spinor_dble *s, openqcd__spinor_dble *r),
    openqcd__spinor **ws, openqcd__spinor_dble **wsd, int nmx, double res,
    openqcd__spinor_dble *eta, openqcd__spinor_dble *psi, int *status);

/* FGCR4VD_C */
extern double openqcd_linsolv__fgcr4vd(
    int vol, int icom,
    void (*Dop)(openqcd__complex_dble *v, openqcd__complex_dble *w),
    void (*Mop)(int k, openqcd__complex *eta, openqcd__complex *psi,
                openqcd__complex *chi),
    openqcd__complex **wv, openqcd__complex_dble **wvd, int nkv, int nmx,
    double res, openqcd__complex_dble *eta, openqcd__complex_dble *psi,
    int *status);

/* FGCR_C */
extern double openqcd_linsolv__fgcr(
    int vol, int icom,
    void (*Dop)(openqcd__spinor_dble *s, openqcd__spinor_dble *r),
    void (*Mop)(int k, openqcd__spinor *rho, openqcd__spinor *phi,
                openqcd__spinor *chi),
    openqcd__spinor **ws, openqcd__spinor_dble **wsd, int nkv, int nmx,
    double res, openqcd__spinor_dble *eta, openqcd__spinor_dble *psi,
    int *status);

/* MSCG_C */
extern void openqcd_linsolv__mscg(int vol, int icom, int nmu, double *mu,
                                  void (*Dop_dble)(double mu,
                                                   openqcd__spinor_dble *s,
                                                   openqcd__spinor_dble *r),
                                  openqcd__spinor_dble **wsd, int nmx,
                                  double *res, openqcd__spinor_dble *eta,
                                  openqcd__spinor_dble **psi, int *status);

#if defined(OPENQCD_INTERNAL)
/* CGNE_C */
#define cgne(...) openqcd_linsolv__cgne(__VA_ARGS__)

/* FGCR4VD_C */
#define fgcr4vd(...) openqcd_linsolv__fgcr4vd(__VA_ARGS__)

/* FGCR_C */
#define fgcr(...) openqcd_linsolv__fgcr(__VA_ARGS__)

/* MSCG_C */
#define mscg(...) openqcd_linsolv__mscg(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif
