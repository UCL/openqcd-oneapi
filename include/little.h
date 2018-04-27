
/*******************************************************************************
 *
 * File little.h
 *
 * Copyright (C) 2011 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef LITTLE_H
#define LITTLE_H

#include "su3.h"

typedef struct
{
  int Ns, nb;
  openqcd__complex **Aee, **Aoo, **Aoe, **Aeo;
} openqcd_little__Aw_t;

typedef struct
{
  int Ns, nb;
  openqcd__complex_dble **Aee, **Aoo, **Aoe, **Aeo;
} openqcd_little__Aw_dble_t;

typedef struct
{
  int n[2];
  int vol, ibn;
  openqcd__spinor_dble **sde[2];
  openqcd__spinor_dble **sdo[2];
} openqcd_little__b2b_flds_t;

/* AW_COM_C */
extern openqcd_little__b2b_flds_t *openqcd_little__b2b_flds(int n, int mu);
extern void openqcd_little__cpAoe_ext_bnd(void);
extern void openqcd_little__cpAee_int_bnd(void);

/* AW_C */
extern void openqcd_little__Aw(openqcd__complex *v, openqcd__complex *w);
extern void openqcd_little__Aweeinv(openqcd__complex *v, openqcd__complex *w);
extern void openqcd_little__Awooinv(openqcd__complex *v, openqcd__complex *w);
extern void openqcd_little__Awoe(openqcd__complex *v, openqcd__complex *w);
extern void openqcd_little__Aweo(openqcd__complex *v, openqcd__complex *w);
extern void openqcd_little__Awhat(openqcd__complex *v, openqcd__complex *w);

/* AW_DBLE_C */
extern void openqcd_little__Aw_dble(openqcd__complex_dble *v,
                                    openqcd__complex_dble *w);
extern void openqcd_little__Aweeinv_dble(openqcd__complex_dble *v,
                                         openqcd__complex_dble *w);
extern void openqcd_little__Awooinv_dble(openqcd__complex_dble *v,
                                         openqcd__complex_dble *w);
extern void openqcd_little__Awoe_dble(openqcd__complex_dble *v,
                                      openqcd__complex_dble *w);
extern void openqcd_little__Aweo_dble(openqcd__complex_dble *v,
                                      openqcd__complex_dble *w);
extern void openqcd_little__Awhat_dble(openqcd__complex_dble *v,
                                       openqcd__complex_dble *w);

/* AW_GEN_C */
extern void openqcd_little__gather_ud(int vol, int const *imb,
                                      openqcd__su3_dble const *ud,
                                      openqcd__su3_dble *vd);
extern void openqcd_little__gather_sd(int vol, int const *imb,
                                      openqcd__spinor_dble const *sd,
                                      openqcd__spinor_dble *rd);
extern void openqcd_little__apply_u2sd(int vol, int const *imb,
                                       openqcd__su3_dble const *ud,
                                       openqcd__spinor_dble const *sd,
                                       openqcd__spinor_dble *rd);
extern void openqcd_little__apply_udag2sd(int vol, int const *imb,
                                          openqcd__su3_dble const *ud,
                                          openqcd__spinor_dble const *sd,
                                          openqcd__spinor_dble *rd);
extern void (*openqcd_little__spinor_prod_gamma[])(
    int vol, openqcd__spinor_dble const *sd, openqcd__spinor_dble const *rd,
    openqcd__complex_dble *sp);

/* AW_OPS_C */
extern openqcd_little__Aw_t openqcd_little__Awop(void);
extern openqcd_little__Aw_t openqcd_little__Awophat(void);
extern openqcd_little__Aw_dble_t openqcd_little__Awop_dble(void);
extern openqcd_little__Aw_dble_t openqcd_little__Awophat_dble(void);
extern void openqcd_little__set_Aw(double mu);
extern int openqcd_little__set_Awhat(double mu);

/* LTL_MODES_C */
extern int openqcd_little__set_ltl_modes(void);
extern openqcd__complex_dble *openqcd_little__ltl_matrix(void);

#if defined(OPENQCD_INTERNAL)
#define Aw_t openqcd_little__Aw_t
#define Aw_dble_t openqcd_little__Aw_dble_t
#define b2b_flds_t openqcd_little__b2b_flds_t

/* AW_COM_C */
#define b2b_flds(...) openqcd_little__b2b_flds(__VA_ARGS__)
#define cpAoe_ext_bnd(...) openqcd_little__cpAoe_ext_bnd(__VA_ARGS__)
#define cpAee_int_bnd(...) openqcd_little__cpAee_int_bnd(__VA_ARGS__)

/* AW_C */
#define Aw(...) openqcd_little__Aw(__VA_ARGS__)
#define Aweeinv(...) openqcd_little__Aweeinv(__VA_ARGS__)
#define Awooinv(...) openqcd_little__Awooinv(__VA_ARGS__)
#define Awoe(...) openqcd_little__Awoe(__VA_ARGS__)
#define Aweo(...) openqcd_little__Aweo(__VA_ARGS__)
#define Awhat(...) openqcd_little__Awhat(__VA_ARGS__)

/* AW_DBLE_C */
#define Aw_dble(...) openqcd_little__Aw_dble(__VA_ARGS__)
#define Aweeinv_dble(...) openqcd_little__Aweeinv_dble(__VA_ARGS__)
#define Awooinv_dble(...) openqcd_little__Awooinv_dble(__VA_ARGS__)
#define Awoe_dble(...) openqcd_little__Awoe_dble(__VA_ARGS__)
#define Aweo_dble(...) openqcd_little__Aweo_dble(__VA_ARGS__)
#define Awhat_dble(...) openqcd_little__Awhat_dble(__VA_ARGS__)

/* AW_GEN_C */
#define gather_ud(...) openqcd_little__gather_ud(__VA_ARGS__)
#define gather_sd(...) openqcd_little__gather_sd(__VA_ARGS__)
#define apply_u2sd(...) openqcd_little__apply_u2sd(__VA_ARGS__)
#define apply_udag2sd(...) openqcd_little__apply_udag2sd(__VA_ARGS__)
#define spinor_prod_gamma openqcd_little__spinor_prod_gamma

/* AW_OPS_C */
#define Awop(...) openqcd_little__Awop(__VA_ARGS__)
#define Awophat(...) openqcd_little__Awophat(__VA_ARGS__)
#define Awop_dble(...) openqcd_little__Awop_dble(__VA_ARGS__)
#define Awophat_dble(...) openqcd_little__Awophat_dble(__VA_ARGS__)
#define set_Aw(...) openqcd_little__set_Aw(__VA_ARGS__)
#define set_Awhat(...) openqcd_little__set_Awhat(__VA_ARGS__)

/* LTL_MODES_C */
#define set_ltl_modes(...) openqcd_little__set_ltl_modes(__VA_ARGS__)
#define ltl_matrix(...) openqcd_little__ltl_matrix(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif
