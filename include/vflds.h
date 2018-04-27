
/*******************************************************************************
 *
 * File vflds.h
 *
 * Copyright (C) 2011 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef VFLDS_H
#define VFLDS_H

#include "su3.h"

/* VCOM_C */
extern void openqcd_vflds__cpv_int_bnd(openqcd__complex *v);
extern void openqcd_vflds__cpv_ext_bnd(openqcd__complex *v);

/* VDCOM_C */
extern void openqcd_vflds__cpvd_int_bnd(openqcd__complex_dble *vd);
extern void openqcd_vflds__cpvd_ext_bnd(openqcd__complex_dble *vd);

/* VFLDS_C */
extern openqcd__complex **openqcd_vflds__vflds(void);
extern openqcd__complex_dble **openqcd_vflds__vdflds(void);

/* VINIT_C */
extern void openqcd_vflds__set_v2zero(int n, openqcd__complex *v);
extern void openqcd_vflds__set_vd2zero(int n, openqcd__complex_dble *vd);
extern void openqcd_vflds__random_v(int n, openqcd__complex *v, float sigma);
extern void openqcd_vflds__random_vd(int n, openqcd__complex_dble *vd,
                                     double sigma);
extern void openqcd_vflds__assign_v2v(int n, openqcd__complex const *v,
                                      openqcd__complex *w);
extern void openqcd_vflds__assign_v2vd(int n, openqcd__complex const *v,
                                       openqcd__complex_dble *wd);
extern void openqcd_vflds__assign_vd2v(int n, openqcd__complex_dble const *vd,
                                       openqcd__complex *w);
extern void openqcd_vflds__assign_vd2vd(int n, openqcd__complex_dble const *vd,
                                        openqcd__complex_dble *wd);
extern void openqcd_vflds__add_v2vd(int n, openqcd__complex const *v,
                                    openqcd__complex_dble *wd);
extern void openqcd_vflds__diff_vd2v(int n, openqcd__complex_dble const *vd,
                                     openqcd__complex_dble const *wd,
                                     openqcd__complex *w);

#if defined(OPENQCD_INTERNAL)
/* VCOM_C */
#define cpv_int_bnd(...) openqcd_vflds__cpv_int_bnd(__VA_ARGS__)
#define cpv_ext_bnd(...) openqcd_vflds__cpv_ext_bnd(__VA_ARGS__)

/* VDCOM_C */
#define cpvd_int_bnd(...) openqcd_vflds__cpvd_int_bnd(__VA_ARGS__)
#define cpvd_ext_bnd(...) openqcd_vflds__cpvd_ext_bnd(__VA_ARGS__)

/* VFLDS_C */
#define vflds(...) openqcd_vflds__vflds(__VA_ARGS__)
#define vdflds(...) openqcd_vflds__vdflds(__VA_ARGS__)

/* VINIT_C */
#define set_v2zero(...) openqcd_vflds__set_v2zero(__VA_ARGS__)
#define set_vd2zero(...) openqcd_vflds__set_vd2zero(__VA_ARGS__)
#define random_v(...) openqcd_vflds__random_v(__VA_ARGS__)
#define random_vd(...) openqcd_vflds__random_vd(__VA_ARGS__)
#define assign_v2v(...) openqcd_vflds__assign_v2v(__VA_ARGS__)
#define assign_v2vd(...) openqcd_vflds__assign_v2vd(__VA_ARGS__)
#define assign_vd2v(...) openqcd_vflds__assign_vd2v(__VA_ARGS__)
#define assign_vd2vd(...) openqcd_vflds__assign_vd2vd(__VA_ARGS__)
#define add_v2vd(...) openqcd_vflds__add_v2vd(__VA_ARGS__)
#define diff_vd2v(...) openqcd_vflds__diff_vd2v(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif
