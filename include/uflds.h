
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
extern su3_dble *bstap(void);
extern void set_bstap(void);

/* PLAQ_SUM_C */
extern void plaq_sum_split_dble(int icom, double *result);
extern double plaq_sum_dble(int icom);
extern void plaq_wsum_split_dble(int icom, double *result);
extern double plaq_wsum_dble(int icom);
extern double plaq_action_slices(double *asl);
extern double spatial_link_sum(int icom);
extern double temporal_link_sum(int icom);

/* SHIFT_C */
extern int shift_ud(int *s);

/* UFLDS_C */
extern su3 *ufld(void);
extern su3_dble *udfld(void);
extern void apply_ani_ud(void);
extern void apply_ani_u(void);
extern void remove_ani_ud(void);
extern void remove_ani_u(void);
extern void random_u(void);
extern void random_ud(void);
extern void renormalize_u(void);
extern void set_ud_phase(void);
extern void unset_ud_phase(void);
extern void renormalize_ud(void);
extern void assign_ud2u(void);
extern void swap_udfld(su3_dble **new_field);
extern void copy_bnd_ud(void);

#endif
