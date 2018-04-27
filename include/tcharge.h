
/*******************************************************************************
 *
 * File tcharge.h
 *
 * Copyright (C) 2010, 2011 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef TCHARGE_H
#define TCHARGE_H

#include "su3.h"

/* FTCOM_C */
extern void openqcd_tcharge__copy_bnd_ft(int n, openqcd__u3_alg_dble *ft);
extern void openqcd_tcharge__add_bnd_ft(int n, openqcd__u3_alg_dble *ft);
extern void openqcd_tcharge__free_ftcom_bufs(void);

/* FTENSOR_C */
extern openqcd__u3_alg_dble **openqcd_tcharge__ftensor(void);

/* TCHARGE_C */
extern double openqcd_tcharge__tcharge(void);
extern double openqcd_tcharge__tcharge_slices(double *qsl);

/* YM_ACTION_C */
extern double openqcd_tcharge__ym_action(void);
extern double openqcd_tcharge__ym_action_slices(double *asl);

#if defined(OPENQCD_INTERNAL)
/* FTCOM_C */
#define copy_bnd_ft(...) openqcd_tcharge__copy_bnd_ft(__VA_ARGS__)
#define add_bnd_ft(...) openqcd_tcharge__add_bnd_ft(__VA_ARGS__)
#define free_ftcom_bufs(...) openqcd_tcharge__free_ftcom_bufs(__VA_ARGS__)

/* FTENSOR_C */
#define ftensor(...) openqcd_tcharge__ftensor(__VA_ARGS__)

/* TCHARGE_C */
#define tcharge(...) openqcd_tcharge__tcharge(__VA_ARGS__)
#define tcharge_slices(...) openqcd_tcharge__tcharge_slices(__VA_ARGS__)

/* YM_ACTION_C */
#define ym_action(...) openqcd_tcharge__ym_action(__VA_ARGS__)
#define ym_action_slices(...) openqcd_tcharge__ym_action_slices(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif
