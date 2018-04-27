
/*******************************************************************************
 *
 * File update.h
 *
 * Copyright (C) 2011 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef UPDATE_H
#define UPDATE_H

#include "flags.h"
#include "su3.h"

typedef struct
{
  int iop;
  double eps;
} openqcd_update__mdstep_t;

/* CHRONO */
extern void openqcd_update__setup_chrono(void);
extern double openqcd_update__mdtime(void);
extern void openqcd_update__step_mdtime(double dt);
extern void openqcd_update__add_chrono(int icr, openqcd__spinor_dble *psi);
extern int openqcd_update__get_chrono(int icr, openqcd__spinor_dble *psi);
extern void openqcd_update__reset_chrono(void);

/* COUNTERS */
extern void openqcd_update__setup_counters(void);
extern void openqcd_update__clear_counters(void);
extern void openqcd_update__add2counter(char const *type, int idx,
                                        int const *status);
extern int openqcd_update__get_count(char const *type, int idx, int *status);
extern void openqcd_update__print_avgstat(char const *type, int idx);
extern void openqcd_update__print_all_avgstat(void);

/* MDSTEPS_C */
extern void openqcd_update__set_mdsteps(void);
extern openqcd_update__mdstep_t *
openqcd_update__mdsteps(size_t *nop, int *ismear, int *iunsmear, int *itu);
extern void openqcd_update__print_mdsteps(int ipr);

/* MDINT_C */
extern void openqcd_update__run_mdint(void);

/* HMC_C */
extern void openqcd_update__hmc_sanity_check(void);
extern void openqcd_update__hmc_wsize(int *nwud, int *nws, int *nwsd, int *nwv,
                                      int *nwvd);
extern int openqcd_update__run_hmc(double *act0, double *act1);

/* RWRAT_C */
extern double openqcd_update__rwrat(int irp, int n, int const *np,
                                    int const *isp, double *sqn, int **status);

/* RWTM_C */
extern double openqcd_update__rwtm1(double mu1, double mu2, int isp,
                                    double *sqn, int *status);
extern double openqcd_update__rwtm2(double mu1, double mu2, int isp,
                                    double *sqn, int *status);

/* RWTMEO_C */
extern double openqcd_update__rwtm1eo(double mu1, double mu2, int isp,
                                      double *sqn, int *status);
extern double openqcd_update__rwtm2eo(double mu1, double mu2, int isp,
                                      double *sqn, int *status);

#if defined(OPENQCD_INTERNAL)
#define mdstep_t openqcd_update__mdstep_t

/* CHRONO */
#define setup_chrono(...) openqcd_update__setup_chrono(__VA_ARGS__)
#define mdtime(...) openqcd_update__mdtime(__VA_ARGS__)
#define step_mdtime(...) openqcd_update__step_mdtime(__VA_ARGS__)
#define add_chrono(...) openqcd_update__add_chrono(__VA_ARGS__)
#define get_chrono(...) openqcd_update__get_chrono(__VA_ARGS__)
#define reset_chrono(...) openqcd_update__reset_chrono(__VA_ARGS__)

/* COUNTERS */
#define setup_counters(...) openqcd_update__setup_counters(__VA_ARGS__)
#define clear_counters(...) openqcd_update__clear_counters(__VA_ARGS__)
#define add2counter(...) openqcd_update__add2counter(__VA_ARGS__)
#define get_count(...) openqcd_update__get_count(__VA_ARGS__)
#define print_avgstat(...) openqcd_update__print_avgstat(__VA_ARGS__)
#define print_all_avgstat(...) openqcd_update__print_all_avgstat(__VA_ARGS__)

/* MDSTEPS_C */
#define set_mdsteps(...) openqcd_update__set_mdsteps(__VA_ARGS__)
#define mdsteps(...) openqcd_update__mdsteps(__VA_ARGS__)
#define print_mdsteps(...) openqcd_update__print_mdsteps(__VA_ARGS__)

/* MDINT_C */
#define run_mdint(...) openqcd_update__run_mdint(__VA_ARGS__)

/* HMC_C */
#define hmc_sanity_check(...) openqcd_update__hmc_sanity_check(__VA_ARGS__)
#define hmc_wsize(...) openqcd_update__hmc_wsize(__VA_ARGS__)
#define run_hmc(...) openqcd_update__run_hmc(__VA_ARGS__)

/* RWRAT_C */
#define rwrat(...) openqcd_update__rwrat(__VA_ARGS__)

/* RWTM_C */
#define rwtm1(...) openqcd_update__rwtm1(__VA_ARGS__)
#define rwtm2(...) openqcd_update__rwtm2(__VA_ARGS__)

/* RWTMEO_C */
#define rwtm1eo(...) openqcd_update__rwtm1eo(__VA_ARGS__)
#define rwtm2eo(...) openqcd_update__rwtm2eo(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif
