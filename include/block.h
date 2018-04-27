
/*******************************************************************************
 *
 * File block.h
 *
 * Copyright (C) 2005, 2011 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef BLOCK_H
#define BLOCK_H

#include "su3.h"
#include "utils.h"

typedef struct
{
  int ifc, ibn, vol, nw, nwd;
  int *ipp, *map, *imb;
  openqcd__su3 *u;
  openqcd__su3_dble *ud;
  openqcd__weyl **w;
  openqcd__weyl_dble **wd;
} openqcd_block__bndry_t;

typedef struct
{
  int *bo, *bs, vol, vbb, nbp, ns, nsd, shf;
  int *ipt, *imb, *ibp;
  int (*iup)[4], (*idn)[4];
  openqcd__su3 *u;
  openqcd__su3_dble *ud;
  openqcd__pauli *sw;
  openqcd__pauli_dble *swd;
  openqcd__spinor **s;
  openqcd__spinor_dble **sd;
  openqcd_block__bndry_t *bb;
} openqcd_block__block_t;

typedef enum
{
  SAP_BLOCKS,
  DFL_BLOCKS,
  BLK_GRIDS
} openqcd_block__blk_grid_t;

/* BLOCK_C */
extern void openqcd_block__alloc_blk(openqcd_block__block_t *b, int const *bo,
                                     int const *bs, int iu, int iud, int ns,
                                     int nsd);
extern void openqcd_block__alloc_bnd(openqcd_block__block_t *b, int iu, int iud,
                                     int nw, int nwd);
extern void openqcd_block__clone_blk(openqcd_block__block_t const *b, int shf,
                                     int const *bo, openqcd_block__block_t *c);
extern void openqcd_block__free_blk(openqcd_block__block_t *b);
extern int openqcd_block__ipt_blk(openqcd_block__block_t const *b,
                                  int const *x);

/* BLK_GRID_C */
extern void openqcd_block__alloc_bgr(openqcd_block__blk_grid_t grid);
extern openqcd_block__block_t *
openqcd_block__blk_list(openqcd_block__blk_grid_t grid, int *nb, int *isw);

/* MAP_U2BLK_C */
extern void openqcd_block__assign_ud2ubgr(openqcd_block__blk_grid_t grid);
extern void openqcd_block__assign_ud2udblk(openqcd_block__blk_grid_t grid,
                                           int n);

/* MAP_SW2BLK_C */
extern int openqcd_block__assign_swd2swbgr(openqcd_block__blk_grid_t grid,
                                           openqcd_utils__ptset_t set);
extern int openqcd_block__assign_swd2swdblk(openqcd_block__blk_grid_t grid,
                                            int n, openqcd_utils__ptset_t set);

/* MAP_S2BLK_C */
extern void openqcd_block__assign_s2sblk(openqcd_block__blk_grid_t grid, int n,
                                         openqcd_utils__ptset_t set,
                                         openqcd__spinor const *s, int k);
extern void openqcd_block__assign_sblk2s(openqcd_block__blk_grid_t grid, int n,
                                         openqcd_utils__ptset_t set, int k,
                                         openqcd__spinor *s);
extern void openqcd_block__assign_s2sdblk(openqcd_block__blk_grid_t grid, int n,
                                          openqcd_utils__ptset_t set,
                                          openqcd__spinor const *s, int k);
extern void openqcd_block__assign_sd2sdblk(openqcd_block__blk_grid_t grid,
                                           int n, openqcd_utils__ptset_t set,
                                           openqcd__spinor_dble const *sd,
                                           int k);
extern void openqcd_block__assign_sdblk2sd(openqcd_block__blk_grid_t grid,
                                           int n, openqcd_utils__ptset_t set,
                                           int k, openqcd__spinor_dble *sd);

#if defined(OPENQCD_INTERNAL)
#define bndry_t openqcd_block__bndry_t
#define block_t openqcd_block__block_t
#define blk_grid_t openqcd_block__blk_grid_t

/* BLOCK_C */
#define alloc_blk(...) openqcd_block__alloc_blk(__VA_ARGS__)
#define alloc_bnd(...) openqcd_block__alloc_bnd(__VA_ARGS__)
#define clone_blk(...) openqcd_block__clone_blk(__VA_ARGS__)
#define free_blk(...) openqcd_block__free_blk(__VA_ARGS__)
#define ipt_blk(...) openqcd_block__ipt_blk(__VA_ARGS__)

/* BLK_GRID_C */
#define alloc_bgr(...) openqcd_block__alloc_bgr(__VA_ARGS__)
#define blk_list(...) openqcd_block__blk_list(__VA_ARGS__)

/* MAP_U2BLK_C */
#define assign_ud2ubgr(...) openqcd_block__assign_ud2ubgr(__VA_ARGS__)
#define assign_ud2udblk(...) openqcd_block__assign_ud2udblk(__VA_ARGS__)

/* MAP_SW2BLK_C */
#define assign_swd2swbgr(...) openqcd_block__assign_swd2swbgr(__VA_ARGS__)
#define assign_swd2swdblk(...) openqcd_block__assign_swd2swdblk(__VA_ARGS__)

/* MAP_S2BLK_C */
#define assign_s2sblk(...) openqcd_block__assign_s2sblk(__VA_ARGS__)
#define assign_sblk2s(...) openqcd_block__assign_sblk2s(__VA_ARGS__)
#define assign_s2sdblk(...) openqcd_block__assign_s2sdblk(__VA_ARGS__)
#define assign_sd2sdblk(...) openqcd_block__assign_sd2sdblk(__VA_ARGS__)
#define assign_sdblk2sd(...) openqcd_block__assign_sdblk2sd(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif
