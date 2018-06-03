
/*******************************************************************************
 *
 * File uflds.h
 *
 * Copyright (C) 2011, 2012, 2013 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef UFLDS_H
#define UFLDS_H

#include "flags.h"
#include "su3.h"

/* BSTAP_C */
extern openqcd__su3_dble *openqcd_uflds__bstap(void);
extern void openqcd_uflds__set_bstap(void);

/* PLAQ_SUM_C */
extern void openqcd_uflds__plaq_sum_split_dble(int icom, double *result);
extern double openqcd_uflds__plaq_sum_dble(int icom);
extern void openqcd_uflds__plaq_wsum_split_dble(int icom, double *result);
extern double openqcd_uflds__plaq_wsum_dble(int icom);
extern double openqcd_uflds__plaq_action_slices(double *asl);
extern double openqcd_uflds__spatial_link_sum(int icom);
extern double openqcd_uflds__temporal_link_sum(int icom);

/* POLYAKOV_LOOP_C */
extern double openqcd_uflds__polyakov_loop(void);

/* SHIFT_C */
extern int openqcd_uflds__shift_ud(int *s);

/* UFLDS_C */
extern openqcd__su3 *openqcd_uflds__ufld(void);
extern openqcd__su3_dble *openqcd_uflds__udfld(void);
extern void openqcd_uflds__apply_ani_ud(void);
extern void openqcd_uflds__remove_ani_ud(void);
extern void openqcd_uflds__random_ud(void);
extern void openqcd_uflds__set_ud_phase(void);
extern void openqcd_uflds__unset_ud_phase(void);
extern void openqcd_uflds__renormalize_ud(void);
extern void openqcd_uflds__assign_ud2u(void);
extern void openqcd_uflds__swap_udfld(openqcd__su3_dble **new_field);
extern void openqcd_uflds__copy_bnd_ud(void);

#if defined(OPENQCD_INTERNAL)
/* BSTAP_C */
#define bstap(...) openqcd_uflds__bstap(__VA_ARGS__)
#define set_bstap(...) openqcd_uflds__set_bstap(__VA_ARGS__)

/* PLAQ_SUM_C */
#define plaq_sum_split_dble(...) openqcd_uflds__plaq_sum_split_dble(__VA_ARGS__)
#define plaq_sum_dble(...) openqcd_uflds__plaq_sum_dble(__VA_ARGS__)
#define plaq_wsum_split_dble(...)                                              \
  openqcd_uflds__plaq_wsum_split_dble(__VA_ARGS__)
#define plaq_wsum_dble(...) openqcd_uflds__plaq_wsum_dble(__VA_ARGS__)
#define plaq_action_slices(...) openqcd_uflds__plaq_action_slices(__VA_ARGS__)
#define spatial_link_sum(...) openqcd_uflds__spatial_link_sum(__VA_ARGS__)
#define temporal_link_sum(...) openqcd_uflds__temporal_link_sum(__VA_ARGS__)

/* POLYAKOV_LOOP_C */
#define polyakov_loop(...) openqcd_uflds__polyakov_loop(__VA_ARGS__)

/* SHIFT_C */
#define shift_ud(...) openqcd_uflds__shift_ud(__VA_ARGS__)

/* UFLDS_C */
#define ufld(...) openqcd_uflds__ufld(__VA_ARGS__)
#define udfld(...) openqcd_uflds__udfld(__VA_ARGS__)
#define apply_ani_ud(...) openqcd_uflds__apply_ani_ud(__VA_ARGS__)
#define remove_ani_ud(...) openqcd_uflds__remove_ani_ud(__VA_ARGS__)
#define random_ud(...) openqcd_uflds__random_ud(__VA_ARGS__)
#define set_ud_phase(...) openqcd_uflds__set_ud_phase(__VA_ARGS__)
#define unset_ud_phase(...) openqcd_uflds__unset_ud_phase(__VA_ARGS__)
#define renormalize_ud(...) openqcd_uflds__renormalize_ud(__VA_ARGS__)
#define assign_ud2u(...) openqcd_uflds__assign_ud2u(__VA_ARGS__)
#define swap_udfld(...) openqcd_uflds__swap_udfld(__VA_ARGS__)
#define copy_bnd_ud(...) openqcd_uflds__copy_bnd_ud(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif
