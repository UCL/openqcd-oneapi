
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

/* GAUSS_C */
#ifdef SITERANDOM
extern void gauss(float r[], int n, int ix);
extern void gauss_dble(double r[], int n, int ix);
#else
extern void gauss(float r[], int n);
extern void gauss_dble(double r[], int n);
#endif

/* RANLUX_C */
extern void start_ranlux(int level, int seed);
extern void export_ranlux(int tag, char *out);
extern int import_ranlux(char *in);

/* RANLXS_C */
extern void ranlxs(float r[], int n);
extern void rlxs_init(int level, int seed);
extern int rlxs_size(void);
extern void rlxs_get(int state[]);
extern void rlxs_reset(int state[]);

/* RANLXD_C */
extern void ranlxd(double r[], int n);
extern void rlxd_init(int level, int seed);
extern int rlxd_size(void);
extern void rlxd_get(int state[]);
extern void rlxd_reset(int state[]);

/* RANLUX_SITE_C */
void ranlxs_site(float r[], int n, int x);
void ranlxd_site(double r[], int n, int x);
void start_ranlux_site(int level, int seed);

#endif
