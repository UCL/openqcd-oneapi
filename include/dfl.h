
/*******************************************************************************
 *
 * File dfl.h
 *
 * Copyright (C) 2011 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef DFL_H
#define DFL_H

#include "su3.h"

typedef struct
{
  int nb, nbb;
  int nbbe[8], nbbo[8];
  int obbe[8], obbo[8];
  int (*inn)[8];
  int *idx, *ipp, *map;
} openqcd_dfl__dfl_grid_t;

/* DFL_GEOMETRY_C */
extern openqcd_dfl__dfl_grid_t openqcd_dfl__dfl_geometry(void);

/* DFL_MODES_C */
extern void openqcd_dfl__dfl_modes(int *status);
extern void openqcd_dfl__dfl_update(int nsm, int *status);
extern void openqcd_dfl__dfl_modes2(int *status);
extern void openqcd_dfl__dfl_update2(int nsm, int *status);

/* DFL_SAP_GCR_C */
extern double openqcd_dfl__dfl_sap_gcr(int nkv, int nmx, double res, double mu,
                                       openqcd__spinor_dble *eta,
                                       openqcd__spinor_dble *psi, int *status);
extern double openqcd_dfl__dfl_sap_gcr2(int nkv, int nmx, double res, double mu,
                                        openqcd__spinor_dble *eta,
                                        openqcd__spinor_dble *psi, int *status);

/* DFL_SUBSPACE_C */
extern void openqcd_dfl__dfl_sd2vd(openqcd__spinor_dble const *sd,
                                   openqcd__complex_dble *vd);
extern void openqcd_dfl__dfl_vd2sd(openqcd__complex_dble const *vd,
                                   openqcd__spinor_dble *sd);
extern void openqcd_dfl__dfl_sub_vd2sd(openqcd__complex_dble const *vd,
                                       openqcd__spinor_dble *sd);
extern void openqcd_dfl__dfl_s2v(openqcd__spinor const *s, openqcd__complex *v);
extern void openqcd_dfl__dfl_v2s(openqcd__complex const *v, openqcd__spinor *s);
extern void openqcd_dfl__dfl_sub_v2s(openqcd__complex const *v,
                                     openqcd__spinor *s);
extern void openqcd_dfl__dfl_subspace(openqcd__spinor **mds);

/* LTL_GCR */
extern double openqcd_dfl__ltl_gcr(int nkv, int nmx, double res, double mu,
                                   openqcd__complex_dble *eta,
                                   openqcd__complex_dble *psi, int *status);

#if defined(OPENQCD_INTERNAL)
#define dfl_grid_t openqcd_dfl__dfl_grid_t

/* DFL_GEOMETRY_C */
#define dfl_geometry(...) openqcd_dfl__dfl_geometry(__VA_ARGS__)

/* DFL_MODES_C */
#define dfl_modes(...) openqcd_dfl__dfl_modes(__VA_ARGS__)
#define dfl_update(...) openqcd_dfl__dfl_update(__VA_ARGS__)
#define dfl_modes2(...) openqcd_dfl__dfl_modes2(__VA_ARGS__)
#define dfl_update2(...) openqcd_dfl__dfl_update2(__VA_ARGS__)

/* DFL_SAP_GCR_C */
#define dfl_sap_gcr(...) openqcd_dfl__dfl_sap_gcr(__VA_ARGS__)
#define dfl_sap_gcr2(...) openqcd_dfl__dfl_sap_gcr2(__VA_ARGS__)

/* DFL_SUBSPACE_C */
#define dfl_sd2vd(...) openqcd_dfl__dfl_sd2vd(__VA_ARGS__)
#define dfl_vd2sd(...) openqcd_dfl__dfl_vd2sd(__VA_ARGS__)
#define dfl_sub_vd2sd(...) openqcd_dfl__dfl_sub_vd2sd(__VA_ARGS__)
#define dfl_s2v(...) openqcd_dfl__dfl_s2v(__VA_ARGS__)
#define dfl_v2s(...) openqcd_dfl__dfl_v2s(__VA_ARGS__)
#define dfl_sub_v2s(...) openqcd_dfl__dfl_sub_v2s(__VA_ARGS__)
#define dfl_subspace(...) openqcd_dfl__dfl_subspace(__VA_ARGS__)

/* LTL_GCR */
#define ltl_gcr(...) openqcd_dfl__ltl_gcr(__VA_ARGS__)

#endif /* defined OPENQCD_INTERNAL */

#endif
