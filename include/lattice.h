
/*******************************************************************************
 *
 * File lattice.h
 *
 * Copyright (C) 2011, 2012, 2013 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef LATTICE_H
#define LATTICE_H

#include "block.h"

typedef struct
{
  int nu0, nuk;
  int *iu0, *iuk;
} openqcd_lattice__uidx_t;

typedef struct
{
  int nft[2];
  int *ift[2];
} openqcd_lattice__ftidx_t;

/* BCNDS_C */
extern int *openqcd_lattice__bnd_lks(int *n);
extern int *openqcd_lattice__bnd_bnd_lks(int *n);
extern int *openqcd_lattice__bnd_pts(int *n);
extern void openqcd_lattice__set_bc(void);
extern int openqcd_lattice__check_bc(double tol);
extern void openqcd_lattice__bnd_s2zero(openqcd_utils__ptset_t set,
                                        openqcd__spinor *s);
extern void openqcd_lattice__bnd_sd2zero(openqcd_utils__ptset_t set,
                                         openqcd__spinor_dble *sd);

/* FTIDX_C */
extern openqcd_lattice__ftidx_t *openqcd_lattice__ftidx(void);
extern void openqcd_lattice__plaq_ftidx(int n, int ix, int *ip);

/* GEOMETRY_C */
extern int openqcd_lattice__ipr_global(int const *n);
extern void openqcd_lattice__ipt_global(int const *x, int *ip, int *ix);
extern int openqcd_lattice__global_time(int ix);
extern void openqcd_lattice__geometry(void);
#if ((defined GEOMETRY_C) || (defined BLOCK_C))
extern void blk_geometry(block_t *b);
extern void blk_imbed(block_t *b);
extern void bnd_geometry(block_t *b);
extern void bnd_imbed(block_t *b);
#endif

/* UIDX_C */
extern openqcd_lattice__uidx_t *openqcd_lattice__uidx(void);
void openqcd_lattice__alloc_uidx(void);
extern void openqcd_lattice__plaq_uidx(int n, int ix, int *ip);

#if defined(OPENQCD_INTERNAL)
#define uidx_t openqcd_lattice__uidx_t
#define ftidx_t openqcd_lattice__ftidx_t

/* BCNDS_C */
#define bnd_lks(...) openqcd_lattice__bnd_lks(__VA_ARGS__)
#define bnd_bnd_lks(...) openqcd_lattice__bnd_bnd_lks(__VA_ARGS__)
#define bnd_pts(...) openqcd_lattice__bnd_pts(__VA_ARGS__)
#define set_bc(...) openqcd_lattice__set_bc(__VA_ARGS__)
#define check_bc(...) openqcd_lattice__check_bc(__VA_ARGS__)
#define bnd_s2zero(...) openqcd_lattice__bnd_s2zero(__VA_ARGS__)
#define bnd_sd2zero(...) openqcd_lattice__bnd_sd2zero(__VA_ARGS__)

/* FTIDX_C */
#define ftidx(...) openqcd_lattice__ftidx(__VA_ARGS__)
#define plaq_ftidx(...) openqcd_lattice__plaq_ftidx(__VA_ARGS__)

/* GEOMETRY_C */
#define ipr_global(...) openqcd_lattice__ipr_global(__VA_ARGS__)
#define ipt_global(...) openqcd_lattice__ipt_global(__VA_ARGS__)
#define global_time(...) openqcd_lattice__global_time(__VA_ARGS__)
#define geometry(...) openqcd_lattice__geometry(__VA_ARGS__)

/* UIDX_C */
#define uidx(...) openqcd_lattice__uidx(__VA_ARGS__)
#define alloc_uidx(...) openqcd_lattice__alloc_uidx(__VA_ARGS__)
#define plaq_uidx(...) openqcd_lattice__plaq_uidx(__VA_ARGS__)

#endif /* defined OPENQCD_INTERNAL */

#endif
