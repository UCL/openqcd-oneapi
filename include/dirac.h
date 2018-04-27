
/*******************************************************************************
 *
 * File dirac.h
 *
 * Copyright (C) 2011 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef DIRAC_H
#define DIRAC_H

#include "block.h"
#include "su3.h"

/* DW_BND_C */
extern void openqcd_dirac__Dw_bnd(openqcd_block__blk_grid_t grid, int n, int k,
                                  int l);

/* DW_C */
extern void openqcd_dirac__Dw(float mu, openqcd__spinor *s, openqcd__spinor *r);
extern void openqcd_dirac__Dwee(float mu, openqcd__spinor *s,
                                openqcd__spinor *r);
extern void openqcd_dirac__Dwoo(float mu, openqcd__spinor *s,
                                openqcd__spinor *r);
extern void openqcd_dirac__Dweo(openqcd__spinor *s, openqcd__spinor *r);
extern void openqcd_dirac__Dwoe(openqcd__spinor *s, openqcd__spinor *r);
extern void openqcd_dirac__Dwhat(float mu, openqcd__spinor *s,
                                 openqcd__spinor *r);
extern void openqcd_dirac__Dw_blk(openqcd_block__blk_grid_t grid, int n,
                                  float mu, int k, int l);
extern void openqcd_dirac__Dwee_blk(openqcd_block__blk_grid_t grid, int n,
                                    float mu, int k, int l);
extern void openqcd_dirac__Dwoo_blk(openqcd_block__blk_grid_t grid, int n,
                                    float mu, int k, int l);
extern void openqcd_dirac__Dwoe_blk(openqcd_block__blk_grid_t grid, int n,
                                    int k, int l);
extern void openqcd_dirac__Dweo_blk(openqcd_block__blk_grid_t grid, int n,
                                    int k, int l);
extern void openqcd_dirac__Dwhat_blk(openqcd_block__blk_grid_t grid, int n,
                                     float mu, int k, int l);

/* DW_DBLE_Copenqcd_dirac__ */
extern void openqcd_dirac__Dw_dble(double mu, openqcd__spinor_dble *s,
                                   openqcd__spinor_dble *r);
extern void openqcd_dirac__Dwee_dble(double mu, openqcd__spinor_dble *s,
                                     openqcd__spinor_dble *r);
extern void openqcd_dirac__Dwoo_dble(double mu, openqcd__spinor_dble *s,
                                     openqcd__spinor_dble *r);
extern void openqcd_dirac__Dweo_dble(openqcd__spinor_dble *s,
                                     openqcd__spinor_dble *r);
extern void openqcd_dirac__Dwoe_dble(openqcd__spinor_dble *s,
                                     openqcd__spinor_dble *r);
extern void openqcd_dirac__Dwhat_dble(double mu, openqcd__spinor_dble *s,
                                      openqcd__spinor_dble *r);
extern void openqcd_dirac__Dw_blk_dble(openqcd_block__blk_grid_t grid, int n,
                                       double mu, int k, int l);
extern void openqcd_dirac__Dwee_blk_dble(openqcd_block__blk_grid_t grid, int n,
                                         double mu, int k, int l);
extern void openqcd_dirac__Dwoo_blk_dble(openqcd_block__blk_grid_t grid, int n,
                                         double mu, int k, int l);
extern void openqcd_dirac__Dwoe_blk_dble(openqcd_block__blk_grid_t grid, int n,
                                         int k, int l);
extern void openqcd_dirac__Dweo_blk_dble(openqcd_block__blk_grid_t grid, int n,
                                         int k, int l);
extern void openqcd_dirac__Dwhat_blk_dble(openqcd_block__blk_grid_t grid, int n,
                                          double mu, int k, int l);

#if defined(OPENQCD_INTERNAL)
/* DW_BND_C */
#define Dw_bnd(...) openqcd_dirac__Dw_bnd(__VA_ARGS__)

/* DW_C */
#define Dw(...) openqcd_dirac__Dw(__VA_ARGS__)
#define Dwee(...) openqcd_dirac__Dwee(__VA_ARGS__)
#define Dwoo(...) openqcd_dirac__Dwoo(__VA_ARGS__)
#define Dweo(...) openqcd_dirac__Dweo(__VA_ARGS__)
#define Dwoe(...) openqcd_dirac__Dwoe(__VA_ARGS__)
#define Dwhat(...) openqcd_dirac__Dwhat(__VA_ARGS__)
#define Dw_blk(...) openqcd_dirac__Dw_blk(__VA_ARGS__)
#define Dwee_blk(...) openqcd_dirac__Dwee_blk(__VA_ARGS__)
#define Dwoo_blk(...) openqcd_dirac__Dwoo_blk(__VA_ARGS__)
#define Dwoe_blk(...) openqcd_dirac__Dwoe_blk(__VA_ARGS__)
#define Dweo_blk(...) openqcd_dirac__Dweo_blk(__VA_ARGS__)
#define Dwhat_blk(...) openqcd_dirac__Dwhat_blk(__VA_ARGS__)

/* DW_DBLE_Copenqcd_dirac__ */
#define Dw_dble(...) openqcd_dirac__Dw_dble(__VA_ARGS__)
#define Dwee_dble(...) openqcd_dirac__Dwee_dble(__VA_ARGS__)
#define Dwoo_dble(...) openqcd_dirac__Dwoo_dble(__VA_ARGS__)
#define Dweo_dble(...) openqcd_dirac__Dweo_dble(__VA_ARGS__)
#define Dwoe_dble(...) openqcd_dirac__Dwoe_dble(__VA_ARGS__)
#define Dwhat_dble(...) openqcd_dirac__Dwhat_dble(__VA_ARGS__)
#define Dw_blk_dble(...) openqcd_dirac__Dw_blk_dble(__VA_ARGS__)
#define Dwee_blk_dble(...) openqcd_dirac__Dwee_blk_dble(__VA_ARGS__)
#define Dwoo_blk_dble(...) openqcd_dirac__Dwoo_blk_dble(__VA_ARGS__)
#define Dwoe_blk_dble(...) openqcd_dirac__Dwoe_blk_dble(__VA_ARGS__)
#define Dweo_blk_dble(...) openqcd_dirac__Dweo_blk_dble(__VA_ARGS__)
#define Dwhat_blk_dble(...) openqcd_dirac__Dwhat_blk_dble(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif
