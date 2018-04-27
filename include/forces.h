
/*******************************************************************************
 *
 * File forces.h
 *
 * Copyright (C) 2011, 2012 Martin Luescher, Stefan Schaefer
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef FORCES_H
#define FORCES_H

#include "su3.h"
#include "utils.h"

/* FORCE0_C */
extern void openqcd_forces__plaq_frc(void);
extern void openqcd_forces__force0(double c);
extern double openqcd_forces__action0(int icom);

/* FORCE1_C */
extern double openqcd_forces__setpf1(double mu, int ipf, int icom);
extern void openqcd_forces__force1(double mu, int ipf, int isp, int icr,
                                   double c, int *status);
extern double openqcd_forces__action1(double mu, int ipf, int isp, int icom,
                                      int *status);

/* FORCE2_C */
extern double openqcd_forces__setpf2(double mu0, double mu1, int ipf, int isp,
                                     int icom, int *status);
extern void openqcd_forces__force2(double mu0, double mu1, int ipf, int isp,
                                   int icr, double c, int *status);
extern double openqcd_forces__action2(double mu0, double mu1, int ipf, int isp,
                                      int icom, int *status);

/* FORCE3_C */
extern double openqcd_forces__setpf3(int const *irat, int ipf, int isw, int isp,
                                     int icom, int *status);
extern void openqcd_forces__force3(int const *irat, int ipf, int isw, int isp,
                                   double c, int *status);
extern double openqcd_forces__action3(int const *irat, int ipf, int isw,
                                      int isp, int icom, int *status);

/* FORCE4_C */
extern double openqcd_forces__setpf4(double mu, int ipf, int isw, int icom);
extern void openqcd_forces__force4(double mu, int ipf, int isw, int isp,
                                   int icr, double c, int *status);
extern double openqcd_forces__action4(double mu, int ipf, int isw, int isp,
                                      int icom, int *status);

/* FORCE5_C */
extern double openqcd_forces__setpf5(double mu0, double mu1, int ipf, int isp,
                                     int icom, int *status);
extern void openqcd_forces__force5(double mu0, double mu1, int ipf, int isp,
                                   int icr, double c, int *status);
extern double openqcd_forces__action5(double mu0, double mu1, int ipf, int isp,
                                      int icom, int *status);

/* FRCFCTS_C */
extern void openqcd_forces__det2xt(openqcd__pauli_dble const *m,
                                   openqcd__u3_alg_dble *X);
extern void openqcd_forces__prod2xt(openqcd__spinor_dble const *r,
                                    openqcd__spinor_dble const *s,
                                    openqcd__u3_alg_dble *X);
extern void (*openqcd_forces__prod2xv[])(openqcd__spinor_dble const *rx,
                                         openqcd__spinor_dble const *ry,
                                         openqcd__spinor_dble const *sx,
                                         openqcd__spinor_dble const *sy,
                                         openqcd__su3_dble *u);

/* GENFRC_C */
extern void openqcd_forces__sw_frc(double c);
extern void openqcd_forces__hop_frc(double c);

/* TMCG_C */
extern double openqcd_forces__tmcg(int nmx, double res, double mu,
                                   openqcd__spinor_dble *eta,
                                   openqcd__spinor_dble *psi, int *status);
extern double openqcd_forces__tmcgeo(int nmx, double res, double mu,
                                     openqcd__spinor_dble *eta,
                                     openqcd__spinor_dble *psi, int *status);

/* TMCGM_C */
extern void openqcd_forces__tmcgm(int nmx, double *res, int nmu, double *mu,
                                  openqcd__spinor_dble *eta,
                                  openqcd__spinor_dble **psi, int *status);

/* XTENSOR_C */
extern openqcd__u3_alg_dble **openqcd_forces__xtensor(void);
extern void openqcd_forces__set_xt2zero(void);
extern int openqcd_forces__add_det2xt(double c, openqcd_utils__ptset_t set);
extern void openqcd_forces__add_prod2xt(double c, openqcd__spinor_dble *r,
                                        openqcd__spinor_dble *s);
extern openqcd__su3_dble *openqcd_forces__xvector(void);
extern void openqcd_forces__set_xv2zero(void);
extern void openqcd_forces__add_prod2xv(double c, openqcd__spinor_dble *r,
                                        openqcd__spinor_dble *s);

#if defined(OPENQCD_INTERNAL)
/* FORCE0_C */
#define plaq_frc(...) openqcd_forces__plaq_frc(__VA_ARGS__)
#define force0(...) openqcd_forces__force0(__VA_ARGS__)
#define action0(...) openqcd_forces__action0(__VA_ARGS__)

/* FORCE1_C */
#define setpf1(...) openqcd_forces__setpf1(__VA_ARGS__)
#define force1(...) openqcd_forces__force1(__VA_ARGS__)
#define action1(...) openqcd_forces__action1(__VA_ARGS__)

/* FORCE2_C */
#define setpf2(...) openqcd_forces__setpf2(__VA_ARGS__)
#define force2(...) openqcd_forces__force2(__VA_ARGS__)
#define action2(...) openqcd_forces__action2(__VA_ARGS__)

/* FORCE3_C */
#define setpf3(...) openqcd_forces__setpf3(__VA_ARGS__)
#define force3(...) openqcd_forces__force3(__VA_ARGS__)
#define action3(...) openqcd_forces__action3(__VA_ARGS__)

/* FORCE4_C */
#define setpf4(...) openqcd_forces__setpf4(__VA_ARGS__)
#define force4(...) openqcd_forces__force4(__VA_ARGS__)
#define action4(...) openqcd_forces__action4(__VA_ARGS__)

/* FORCE5_C */
#define setpf5(...) openqcd_forces__setpf5(__VA_ARGS__)
#define force5(...) openqcd_forces__force5(__VA_ARGS__)
#define action5(...) openqcd_forces__action5(__VA_ARGS__)

/* FRCFCTS_C */
#define det2xt(...) openqcd_forces__det2xt(__VA_ARGS__)
#define prod2xt(...) openqcd_forces__prod2xt(__VA_ARGS__)
#define prod2xv openqcd_forces__prod2xv

/* GENFRC_C */
#define sw_frc(...) openqcd_forces__sw_frc(__VA_ARGS__)
#define hop_frc(...) openqcd_forces__hop_frc(__VA_ARGS__)

/* TMCG_C */
#define tmcg(...) openqcd_forces__tmcg(__VA_ARGS__)
#define tmcgeo(...) openqcd_forces__tmcgeo(__VA_ARGS__)

/* TMCGM_C */
#define tmcgm(...) openqcd_forces__tmcgm(__VA_ARGS__)

/* XTENSOR_C */
#define xtensor(...) openqcd_forces__xtensor(__VA_ARGS__)
#define set_xt2zero(...) openqcd_forces__set_xt2zero(__VA_ARGS__)
#define add_det2xt(...) openqcd_forces__add_det2xt(__VA_ARGS__)
#define add_prod2xt(...) openqcd_forces__add_prod2xt(__VA_ARGS__)
#define xvector(...) openqcd_forces__xvector(__VA_ARGS__)
#define set_xv2zero(...) openqcd_forces__set_xv2zero(__VA_ARGS__)
#define add_prod2xv(...) openqcd_forces__add_prod2xv(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif
