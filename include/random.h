
/*******************************************************************************
 *
 * File random.h
 *
 * Copyright (C) 2005, 2011, 2013 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef RANDOM_H
#define RANDOM_H

#include "utils.h"
#include <stdio.h>

/* GAUSS_C */
#ifdef SITERANDOM
extern void openqcd_random__gauss(float r[], int n, int ix);
extern void openqcd_random__gauss_dble(double r[], int n, int ix);
#else
extern void openqcd_random__gauss(float r[], int n);
extern void openqcd_random__gauss_dble(double r[], int n);
#endif

/* RANLUX_C */
extern void openqcd_random__start_ranlux(int level, int seed);
extern void openqcd_random__export_ranlux(int tag, char *out);
extern int openqcd_random__import_ranlux(char const *in);

/* RANLXS_C */
extern void openqcd_random__ranlxs(float r[], int n);
extern void openqcd_random__rlxs_init(int level, int seed);
extern int openqcd_random__rlxs_size(void);
extern void openqcd_random__rlxs_get(int state[]);
extern void openqcd_random__rlxs_reset(int state[]);

/* RANLXD_C */
extern void openqcd_random__ranlxd(double r[], int n);
extern void openqcd_random__rlxd_init(int level, int seed);
extern int openqcd_random__rlxd_size(void);
extern void openqcd_random__rlxd_get(int state[]);
extern void openqcd_random__rlxd_reset(int state[]);

/* RANLUX_SITE_C */
void openqcd_random__ranlxs_site(float r[], int n, int x);
void openqcd_random__ranlxd_site(double r[], int n, int x);
void openqcd_random__start_ranlux_site(int level, int seed);

#if defined(OPENQCD_INTERNAL)
/* GAUSS_C */
#define gauss(...) openqcd_random__gauss(__VA_ARGS__)
#define gauss_dble(...) openqcd_random__gauss_dble(__VA_ARGS__)

/* RANLUX_C */
#define start_ranlux(...) openqcd_random__start_ranlux(__VA_ARGS__)
#define export_ranlux(...) openqcd_random__export_ranlux(__VA_ARGS__)
#define import_ranlux(...) openqcd_random__import_ranlux(__VA_ARGS__)

/* RANLXS_C */
#define ranlxs(...) openqcd_random__ranlxs(__VA_ARGS__)
#define rlxs_init(...) openqcd_random__rlxs_init(__VA_ARGS__)
#define rlxs_size(...) openqcd_random__rlxs_size(__VA_ARGS__)
#define rlxs_get(...) openqcd_random__rlxs_get(__VA_ARGS__)
#define rlxs_reset(...) openqcd_random__rlxs_reset(__VA_ARGS__)

/* RANLXD_C */
#define ranlxd(...) openqcd_random__ranlxd(__VA_ARGS__)
#define rlxd_init(...) openqcd_random__rlxd_init(__VA_ARGS__)
#define rlxd_size(...) openqcd_random__rlxd_size(__VA_ARGS__)
#define rlxd_get(...) openqcd_random__rlxd_get(__VA_ARGS__)
#define rlxd_reset(...) openqcd_random__rlxd_reset(__VA_ARGS__)

/* RANLUX_SITE_C */
#define ranlxs_site(...) openqcd_random__ranlxs_site(__VA_ARGS__)
#define ranlxd_site(...) openqcd_random__ranlxd_site(__VA_ARGS__)
#define start_ranlux_site(...) openqcd_random__start_ranlux_site(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif
