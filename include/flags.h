
/*******************************************************************************
 *
 * File flags.h
 *
 * Copyright (C) 2009-2014, 2016 Martin Luescher, Isabel Campos
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef FLAGS_H
#define FLAGS_H

#include "block.h"
#include <stdio.h>

typedef enum
{
  UPDATED_U,
  UPDATED_UD,
  ASSIGNED_UD2U,
  COPIED_BND_UD,
  SET_BSTAP,
  SHIFTED_UD,
  COMPUTED_FTS,
  ERASED_SW,
  ERASED_SWD,
  COMPUTED_SWD,
  ASSIGNED_SWD2SW,
  INVERTED_SW_E,
  INVERTED_SW_O,
  INVERTED_SWD_E,
  INVERTED_SWD_O,
  ASSIGNED_U2UBGR,
  ASSIGNED_UD2UBGR,
  ASSIGNED_UD2UDBGR,
  ASSIGNED_SWD2SWBGR,
  ASSIGNED_SWD2SWDBGR,
  ERASED_AW,
  ERASED_AWHAT,
  COMPUTED_AW,
  COMPUTED_AWHAT,
  SMEARED_UD,
  UNSMEARED_UD,
  SET_UD_PHASE,
  UNSET_UD_PHASE,
  EVENTS
} openqcd_flags__event_t;

typedef enum
{
  U_MATCH_UD,
  UDBUF_UP2DATE,
  BSTAP_UP2DATE,
  FTS_UP2DATE,
  UBGR_MATCH_UD,
  UDBGR_MATCH_UD,
  SW_UP2DATE,
  SW_E_INVERTED,
  SW_O_INVERTED,
  SWD_UP2DATE,
  SWD_E_INVERTED,
  SWD_O_INVERTED,
  AW_UP2DATE,
  AWHAT_UP2DATE,
  UD_IS_CLEAN,
  UD_IS_SMEARED,
  SMEARED_UD_UP2DATE,
  UD_PHASE_SET,
  QUERIES
} openqcd_flags__query_t;

typedef enum
{
  ACG,
  ACF_TM1,
  ACF_TM1_EO,
  ACF_TM1_EO_SDET,
  ACF_TM2,
  ACF_TM2_EO,
  ACF_RAT,
  ACF_RAT_SDET,
  ACTIONS
} openqcd_flags__action_t;

typedef enum
{
  LPFR,
  OMF2,
  OMF4,
  INTEGRATORS
} openqcd_flags__integrator_t;

typedef enum
{
  FRG,
  FRF_TM1,
  FRF_TM1_EO,
  FRF_TM1_EO_SDET,
  FRF_TM2,
  FRF_TM2_EO,
  FRF_RAT,
  FRF_RAT_SDET,
  FORCES
} openqcd_flags__force_t;

typedef enum
{
  RWTM1,
  RWTM1_EO,
  RWTM2,
  RWTM2_EO,
  RWRAT,
  RWFACTS
} openqcd_flags__rwfact_t;

typedef enum
{
  CGNE,
  MSCG,
  SAP_GCR,
  DFL_SAP_GCR,
  SOLVERS
} openqcd_flags__solver_t;

typedef struct
{
  openqcd_flags__action_t action;
  int ipf, im0;
  int irat[3], imu[4];
  int isp[4];
  int smear;
} openqcd_flags__action_parms_t;

typedef struct
{
  int type;
  double cG[2], cF[2];
  double phi[2][3];
  double theta[3];
} openqcd_flags__bc_parms_t;

typedef struct
{
  int bs[4];
  int Ns;
} openqcd_flags__dfl_parms_t;

typedef struct
{
  int nkv, nmx;
  double res;
} openqcd_flags__dfl_pro_parms_t;

typedef struct
{
  int ninv, nmr, ncy;
  double kappa, m0, mu;
} openqcd_flags__dfl_gen_parms_t;

typedef struct
{
  int nsm;
  double dtau;
} openqcd_flags__dfl_upd_parms_t;

typedef struct
{
  openqcd_flags__force_t force;
  int ipf, im0;
  int irat[3], imu[4];
  int isp[4];
  int ncr[4], icr[4];
} openqcd_flags__force_parms_t;

typedef struct
{
  int npf, nlv;
  int nact, nmu;
  int *iact;
  double tau, *mu;
} openqcd_flags__hmc_parms_t;

typedef struct
{
  int nk;
  double beta, c0, c1;
  double *kappa, *m0;
  double csw;
} openqcd_flags__lat_parms_t;

typedef struct
{
  openqcd_flags__integrator_t integrator;
  double lambda;
  int nstep, nfr;
  int *ifr;
} openqcd_flags__mdint_parms_t;

typedef struct
{
  int degree;
  double range[2];
} openqcd_flags__rat_parms_t;

typedef struct
{
  openqcd_flags__rwfact_t rwfact;
  int im0, nsrc;
  int irp, nfct;
  double *mu;
  int *np, *isp;
} openqcd_flags__rw_parms_t;

typedef struct
{
  double m0, csw, cF[2];
} openqcd_flags__sw_parms_t;

typedef struct
{
  int bs[4];
  int isolv;
  int nmr, ncy;
} openqcd_flags__sap_parms_t;

typedef struct
{
  openqcd_flags__solver_t solver;
  int nmx, nkv;
  int isolv, nmr, ncy;
  double res;
} openqcd_flags__solver_parms_t;

typedef struct
{
  int eoflg;
} openqcd_flags__tm_parms_t;

typedef struct
{
  int has_ani;
  int has_tts;

  double nu;
  double xi;
  double cR;
  double cT;
  double us_gauge;
  double ut_gauge;
  double us_fermion;
  double ut_fermion;
} openqcd_flags__ani_params_t;

typedef struct
{
  int num_smear;

  int smear_temporal;
  double rho_temporal;

  int smear_spatial;
  double rho_spatial;

  int smear_gauge;
  int smear_fermion;
} openqcd_flags__stout_smearing_params_t;

/* FLAGS_C */
extern void openqcd_flags__set_flags(openqcd_flags__event_t event);
extern void openqcd_flags__set_grid_flags(openqcd_block__blk_grid_t grid,
                                          openqcd_flags__event_t event);
extern int openqcd_flags__query_flags(openqcd_flags__query_t query);
extern int openqcd_flags__query_grid_flags(openqcd_block__blk_grid_t grid,
                                           openqcd_flags__query_t query);
extern void openqcd_flags__print_flags(void);
extern void openqcd_flags__print_grid_flags(openqcd_block__blk_grid_t grid);

/* ACTION_PARMS_C */
extern openqcd_flags__action_parms_t
openqcd_flags__set_action_parms(int iact, openqcd_flags__action_t action,
                                int ipf, int im0, int const *irat,
                                int const *imu, int const *isp, int smear);

extern openqcd_flags__action_parms_t openqcd_flags__action_parms(int iact);
extern void openqcd_flags__read_action_parms(int iact);
extern void openqcd_flags__print_action_parms(void);
extern void openqcd_flags__write_action_parms(FILE *fdat);
extern void openqcd_flags__check_action_parms(FILE *fdat, int read_only);

/* DFL_PARMS_C */
extern openqcd_flags__dfl_parms_t openqcd_flags__set_dfl_parms(int const *bs,
                                                               int Ns);
extern openqcd_flags__dfl_parms_t openqcd_flags__dfl_parms(void);
extern openqcd_flags__dfl_pro_parms_t
openqcd_flags__set_dfl_pro_parms(int nkv, int nmx, double res);
extern openqcd_flags__dfl_pro_parms_t openqcd_flags__dfl_pro_parms(void);
extern openqcd_flags__dfl_gen_parms_t
openqcd_flags__set_dfl_gen_parms(double kappa, double mu, int ninv, int nmr,
                                 int ncy);
extern openqcd_flags__dfl_gen_parms_t openqcd_flags__dfl_gen_parms(void);
extern openqcd_flags__dfl_upd_parms_t
openqcd_flags__set_dfl_upd_parms(double dtau, int nsm);
extern openqcd_flags__dfl_upd_parms_t openqcd_flags__dfl_upd_parms(void);
extern void openqcd_flags__print_dfl_parms(int ipr);
extern void openqcd_flags__write_dfl_parms(FILE *fdat);
extern void openqcd_flags__check_dfl_parms(FILE *fdat, int read_only);

/* FORCE_PARMS_C */
extern openqcd_flags__force_parms_t
openqcd_flags__set_force_parms(int ifr, openqcd_flags__force_t force, int ipf,
                               int im0, int const *irat, int const *imu,
                               int const *isp, int const *ncr);

extern openqcd_flags__force_parms_t openqcd_flags__force_parms(int ifr);
extern void openqcd_flags__read_force_parms(int ifr);
extern void openqcd_flags__read_force_parms2(int ifr);
extern void openqcd_flags__print_force_parms(void);
extern void openqcd_flags__print_force_parms2(void);
extern void openqcd_flags__write_force_parms(FILE *fdat);
extern void openqcd_flags__check_force_parms(FILE *fdat, int read_only);

/* HMC_PARMS_C */
extern openqcd_flags__hmc_parms_t
openqcd_flags__set_hmc_parms(int nact, int const *iact, int npf, int nmu,
                             double const *mu, int nlv, double tau);
extern openqcd_flags__hmc_parms_t openqcd_flags__hmc_parms(void);
extern void openqcd_flags__print_hmc_parms(void);
extern void openqcd_flags__write_hmc_parms(FILE *fdat);
extern void openqcd_flags__check_hmc_parms(FILE *fdat);

/* LAT_PARMS_C */
extern openqcd_flags__lat_parms_t
openqcd_flags__set_lat_parms(double beta, double c0, int nk,
                             double const *kappa, double csw);

extern openqcd_flags__lat_parms_t openqcd_flags__lat_parms(void);
extern void openqcd_flags__print_lat_parms(void);
extern void openqcd_flags__write_lat_parms(FILE *fdat);
extern void openqcd_flags__check_lat_parms(FILE *fdat);

extern openqcd_flags__bc_parms_t
openqcd_flags__set_bc_parms(int type, double cG, double cG_prime, double cF,
                            double cF_prime, double const *phi,
                            double const *phi_prime, double const *theta);

extern openqcd_flags__bc_parms_t openqcd_flags__bc_parms(void);
extern void openqcd_flags__print_bc_parms(int ipr);
extern void openqcd_flags__write_bc_parms(FILE *fdat);
extern void openqcd_flags__check_bc_parms(FILE *fdat);
extern double openqcd_flags__sea_quark_mass(int im0);
extern int openqcd_flags__bc_type(void);
extern openqcd_flags__sw_parms_t openqcd_flags__set_sw_parms(double m0);
extern openqcd_flags__sw_parms_t openqcd_flags__sw_parms(void);
extern openqcd_flags__tm_parms_t openqcd_flags__set_tm_parms(int eoflg);
extern openqcd_flags__tm_parms_t openqcd_flags__tm_parms(void);

/* MDINT_PARMS_C */
extern openqcd_flags__mdint_parms_t
openqcd_flags__set_mdint_parms(int ilv, openqcd_flags__integrator_t integrator,
                               double lambda, int nstep, int nfr,
                               int const *ifr);
extern openqcd_flags__mdint_parms_t openqcd_flags__mdint_parms(int ilv);
extern void openqcd_flags__read_mdint_parms(int ilv);
extern void openqcd_flags__print_mdint_parms(void);
extern void openqcd_flags__write_mdint_parms(FILE *fdat);
extern void openqcd_flags__check_mdint_parms(FILE *fdat);

/* RAT_PARMS_C */
extern openqcd_flags__rat_parms_t
openqcd_flags__set_rat_parms(int irp, int degree, double const *range);
extern openqcd_flags__rat_parms_t openqcd_flags__rat_parms(int irp);
extern void openqcd_flags__read_rat_parms(int irp);
extern void openqcd_flags__print_rat_parms(void);
extern void openqcd_flags__write_rat_parms(FILE *fdat);
extern void openqcd_flags__check_rat_parms(FILE *fdat);

/* RW_PARMS_C */
extern openqcd_flags__rw_parms_t
openqcd_flags__set_rw_parms(int irw, openqcd_flags__rwfact_t rwfact, int im0,
                            int nsrc, int irp, int nfct, double const *mu,
                            int const *np, int const *isp);

extern openqcd_flags__rw_parms_t openqcd_flags__rw_parms(int irw);
extern void openqcd_flags__read_rw_parms(int irw);
extern void openqcd_flags__print_rw_parms(void);
extern void openqcd_flags__write_rw_parms(FILE *fdat);
extern void openqcd_flags__check_rw_parms(FILE *fdat);

/* SAP_PARMS_C */
extern openqcd_flags__sap_parms_t
openqcd_flags__set_sap_parms(int const *bs, int isolv, int nmr, int ncy);
extern openqcd_flags__sap_parms_t openqcd_flags__sap_parms(void);
extern void openqcd_flags__print_sap_parms(int ipr);
extern void openqcd_flags__write_sap_parms(FILE *fdat);
extern void openqcd_flags__check_sap_parms(FILE *fdat, int read_only);

/* SOLVER_PARMS_C */
extern openqcd_flags__solver_parms_t
openqcd_flags__set_solver_parms(int isp, openqcd_flags__solver_t solver,
                                int nkv, int isolv, int nmr, int ncy, int nmx,
                                double res);
extern openqcd_flags__solver_parms_t openqcd_flags__solver_parms(int isp);
extern void openqcd_flags__read_solver_parms(int isp);
extern void openqcd_flags__print_solver_parms(int *isap, int *idfl);
extern void openqcd_flags__write_solver_parms(FILE *fdat);
extern void openqcd_flags__check_solver_parms(FILE *fdat, int read_only);

/* ANISOTROPY_PARMS_C */
extern openqcd_flags__ani_params_t
openqcd_flags__set_ani_parms(int use_tts, double nu, double xi, double cR,
                             double cT, double us, double ut, double ust,
                             double utt);

extern openqcd_flags__ani_params_t openqcd_flags__set_no_ani_parms(void);
extern openqcd_flags__ani_params_t openqcd_flags__ani_parms(void);
extern void openqcd_flags__print_ani_parms(void);
extern int openqcd_flags__ani_params_initialised(void);
extern void openqcd_flags__write_ani_parms(FILE *fdat);
extern void openqcd_flags__check_ani_parms(FILE *fdat);

/* SMEARING_PARMS_C */
extern openqcd_flags__stout_smearing_params_t
openqcd_flags__set_stout_smearing_parms(int n, double pt, double ps,
                                        int smear_gauge, int smear_fermion);

extern openqcd_flags__stout_smearing_params_t
openqcd_flags__set_no_stout_smearing_parms(void);
extern void openqcd_flags__reset_stout_smearing(void);
extern openqcd_flags__stout_smearing_params_t
openqcd_flags__stout_smearing_parms(void);
extern void openqcd_flags__print_stout_smearing_parms(void);
extern void openqcd_flags__write_stout_smearing_parms(FILE *fdat);
extern void openqcd_flags__check_stout_smearing_parms(FILE *fdat);

#if defined(OPENQCD_INTERNAL)
#define event_t openqcd_flags__event_t
#define query_t openqcd_flags__query_t
#define action_t openqcd_flags__action_t
#define integrator_t openqcd_flags__integrator_t
#define force_t openqcd_flags__force_t
#define rwfact_t openqcd_flags__rwfact_t
#define solver_t openqcd_flags__solver_t
#define action_parms_t openqcd_flags__action_parms_t
#define bc_parms_t openqcd_flags__bc_parms_t
#define dfl_parms_t openqcd_flags__dfl_parms_t
#define dfl_pro_parms_t openqcd_flags__dfl_pro_parms_t
#define dfl_gen_parms_t openqcd_flags__dfl_gen_parms_t
#define dfl_upd_parms_t openqcd_flags__dfl_upd_parms_t
#define force_parms_t openqcd_flags__force_parms_t
#define hmc_parms_t openqcd_flags__hmc_parms_t
#define lat_parms_t openqcd_flags__lat_parms_t
#define mdint_parms_t openqcd_flags__mdint_parms_t
#define rat_parms_t openqcd_flags__rat_parms_t
#define rw_parms_t openqcd_flags__rw_parms_t
#define sw_parms_t openqcd_flags__sw_parms_t
#define sap_parms_t openqcd_flags__sap_parms_t
#define solver_parms_t openqcd_flags__solver_parms_t
#define tm_parms_t openqcd_flags__tm_parms_t
#define ani_params_t openqcd_flags__ani_params_t
#define stout_smearing_params_t openqcd_flags__stout_smearing_params_t

/* FLAGS_C */
#define set_flags(...) openqcd_flags__set_flags(__VA_ARGS__)
#define set_grid_flags(...) openqcd_flags__set_grid_flags(__VA_ARGS__)
#define query_flags(...) openqcd_flags__query_flags(__VA_ARGS__)
#define query_grid_flags(...) openqcd_flags__query_grid_flags(__VA_ARGS__)
#define print_flags(...) openqcd_flags__print_flags(__VA_ARGS__)
#define print_grid_flags(...) openqcd_flags__print_grid_flags(__VA_ARGS__)

/* ACTION_PARMS_C */
#define set_action_parms(...) openqcd_flags__set_action_parms(__VA_ARGS__)

#define action_parms(...) openqcd_flags__action_parms(__VA_ARGS__)
#define read_action_parms(...) openqcd_flags__read_action_parms(__VA_ARGS__)
#define print_action_parms(...) openqcd_flags__print_action_parms(__VA_ARGS__)
#define write_action_parms(...) openqcd_flags__write_action_parms(__VA_ARGS__)
#define check_action_parms(...) openqcd_flags__check_action_parms(__VA_ARGS__)

/* DFL_PARMS_C */
#define set_dfl_parms(...) openqcd_flags__set_dfl_parms(__VA_ARGS__)
#define dfl_parms(...) openqcd_flags__dfl_parms(__VA_ARGS__)
#define set_dfl_pro_parms(...) openqcd_flags__set_dfl_pro_parms(__VA_ARGS__)
#define dfl_pro_parms(...) openqcd_flags__dfl_pro_parms(__VA_ARGS__)
#define set_dfl_gen_parms(...) openqcd_flags__set_dfl_gen_parms(__VA_ARGS__)
#define dfl_gen_parms(...) openqcd_flags__dfl_gen_parms(__VA_ARGS__)
#define set_dfl_upd_parms(...) openqcd_flags__set_dfl_upd_parms(__VA_ARGS__)
#define dfl_upd_parms(...) openqcd_flags__dfl_upd_parms(__VA_ARGS__)
#define print_dfl_parms(...) openqcd_flags__print_dfl_parms(__VA_ARGS__)
#define write_dfl_parms(...) openqcd_flags__write_dfl_parms(__VA_ARGS__)
#define check_dfl_parms(...) openqcd_flags__check_dfl_parms(__VA_ARGS__)

/* FORCE_PARMS_C */
#define set_force_parms(...) openqcd_flags__set_force_parms(__VA_ARGS__)

#define force_parms(...) openqcd_flags__force_parms(__VA_ARGS__)
#define read_force_parms(...) openqcd_flags__read_force_parms(__VA_ARGS__)
#define read_force_parms2(...) openqcd_flags__read_force_parms2(__VA_ARGS__)
#define print_force_parms(...) openqcd_flags__print_force_parms(__VA_ARGS__)
#define print_force_parms2(...) openqcd_flags__print_force_parms2(__VA_ARGS__)
#define write_force_parms(...) openqcd_flags__write_force_parms(__VA_ARGS__)
#define check_force_parms(...) openqcd_flags__check_force_parms(__VA_ARGS__)

/* HMC_PARMS_C */
#define set_hmc_parms(...) openqcd_flags__set_hmc_parms(__VA_ARGS__)
#define hmc_parms(...) openqcd_flags__hmc_parms(__VA_ARGS__)
#define print_hmc_parms(...) openqcd_flags__print_hmc_parms(__VA_ARGS__)
#define write_hmc_parms(...) openqcd_flags__write_hmc_parms(__VA_ARGS__)
#define check_hmc_parms(...) openqcd_flags__check_hmc_parms(__VA_ARGS__)

/* LAT_PARMS_C */
#define set_lat_parms(...) openqcd_flags__set_lat_parms(__VA_ARGS__)

#define lat_parms(...) openqcd_flags__lat_parms(__VA_ARGS__)
#define print_lat_parms(...) openqcd_flags__print_lat_parms(__VA_ARGS__)
#define write_lat_parms(...) openqcd_flags__write_lat_parms(__VA_ARGS__)
#define check_lat_parms(...) openqcd_flags__check_lat_parms(__VA_ARGS__)

#define set_bc_parms(...) openqcd_flags__set_bc_parms(__VA_ARGS__)

#define bc_parms(...) openqcd_flags__bc_parms(__VA_ARGS__)
#define print_bc_parms(...) openqcd_flags__print_bc_parms(__VA_ARGS__)
#define write_bc_parms(...) openqcd_flags__write_bc_parms(__VA_ARGS__)
#define check_bc_parms(...) openqcd_flags__check_bc_parms(__VA_ARGS__)
#define sea_quark_mass(...) openqcd_flags__sea_quark_mass(__VA_ARGS__)
#define bc_type(...) openqcd_flags__bc_type(__VA_ARGS__)
#define set_sw_parms(...) openqcd_flags__set_sw_parms(__VA_ARGS__)
#define sw_parms(...) openqcd_flags__sw_parms(__VA_ARGS__)
#define set_tm_parms(...) openqcd_flags__set_tm_parms(__VA_ARGS__)
#define tm_parms(...) openqcd_flags__tm_parms(__VA_ARGS__)

/* MDINT_PARMS_C */
#define set_mdint_parms(...) openqcd_flags__set_mdint_parms(__VA_ARGS__)
#define mdint_parms(...) openqcd_flags__mdint_parms(__VA_ARGS__)
#define read_mdint_parms(...) openqcd_flags__read_mdint_parms(__VA_ARGS__)
#define print_mdint_parms(...) openqcd_flags__print_mdint_parms(__VA_ARGS__)
#define write_mdint_parms(...) openqcd_flags__write_mdint_parms(__VA_ARGS__)
#define check_mdint_parms(...) openqcd_flags__check_mdint_parms(__VA_ARGS__)

/* RAT_PARMS_C */
#define set_rat_parms(...) openqcd_flags__set_rat_parms(__VA_ARGS__)
#define rat_parms(...) openqcd_flags__rat_parms(__VA_ARGS__)
#define read_rat_parms(...) openqcd_flags__read_rat_parms(__VA_ARGS__)
#define print_rat_parms(...) openqcd_flags__print_rat_parms(__VA_ARGS__)
#define write_rat_parms(...) openqcd_flags__write_rat_parms(__VA_ARGS__)
#define check_rat_parms(...) openqcd_flags__check_rat_parms(__VA_ARGS__)

/* RW_PARMS_C */
#define set_rw_parms(...) openqcd_flags__set_rw_parms(__VA_ARGS__)

#define rw_parms(...) openqcd_flags__rw_parms(__VA_ARGS__)
#define read_rw_parms(...) openqcd_flags__read_rw_parms(__VA_ARGS__)
#define print_rw_parms(...) openqcd_flags__print_rw_parms(__VA_ARGS__)
#define write_rw_parms(...) openqcd_flags__write_rw_parms(__VA_ARGS__)
#define check_rw_parms(...) openqcd_flags__check_rw_parms(__VA_ARGS__)

/* SAP_PARMS_C */
#define set_sap_parms(...) openqcd_flags__set_sap_parms(__VA_ARGS__)
#define sap_parms(...) openqcd_flags__sap_parms(__VA_ARGS__)
#define print_sap_parms(...) openqcd_flags__print_sap_parms(__VA_ARGS__)
#define write_sap_parms(...) openqcd_flags__write_sap_parms(__VA_ARGS__)
#define check_sap_parms(...) openqcd_flags__check_sap_parms(__VA_ARGS__)

/* SOLVER_PARMS_C */
#define set_solver_parms(...) openqcd_flags__set_solver_parms(__VA_ARGS__)
#define solver_parms(...) openqcd_flags__solver_parms(__VA_ARGS__)
#define read_solver_parms(...) openqcd_flags__read_solver_parms(__VA_ARGS__)
#define print_solver_parms(...) openqcd_flags__print_solver_parms(__VA_ARGS__)
#define write_solver_parms(...) openqcd_flags__write_solver_parms(__VA_ARGS__)
#define check_solver_parms(...) openqcd_flags__check_solver_parms(__VA_ARGS__)

/* ANISOTROPY_PARMS_C */
#define set_ani_parms(...) openqcd_flags__set_ani_parms(__VA_ARGS__)

#define set_no_ani_parms(...) openqcd_flags__set_no_ani_parms(__VA_ARGS__)
#define ani_parms(...) openqcd_flags__ani_parms(__VA_ARGS__)
#define print_ani_parms(...) openqcd_flags__print_ani_parms(__VA_ARGS__)
#define ani_params_initialised(...)                                            \
  openqcd_flags__ani_params_initialised(__VA_ARGS__)
#define write_ani_parms(...) openqcd_flags__write_ani_parms(__VA_ARGS__)
#define check_ani_parms(...) openqcd_flags__check_ani_parms(__VA_ARGS__)

/* SMEARING_PARMS_C */
#define set_stout_smearing_parms(...)                                          \
  openqcd_flags__set_stout_smearing_parms(__VA_ARGS__)

#define set_no_stout_smearing_parms(...)                                       \
  openqcd_flags__set_no_stout_smearing_parms(__VA_ARGS__)
#define reset_stout_smearing(...)                                              \
  openqcd_flags__reset_stout_smearing(__VA_ARGS__)
#define stout_smearing_parms(...)                                              \
  openqcd_flags__stout_smearing_parms(__VA_ARGS__)
#define print_stout_smearing_parms(...)                                        \
  openqcd_flags__print_stout_smearing_parms(__VA_ARGS__)
#define write_stout_smearing_parms(...)                                        \
  openqcd_flags__write_stout_smearing_parms(__VA_ARGS__)
#define check_stout_smearing_parms(...)                                        \
  openqcd_flags__check_stout_smearing_parms(__VA_ARGS__)

#endif /* defined OPENQCD_INTERNAL */

#endif
