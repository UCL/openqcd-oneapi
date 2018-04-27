
/*******************************************************************************
 *
 * File sflds.h
 *
 * Copyright (C) 2011 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef SFLDS_H
#define SFLDS_H

#include "su3.h"

/* PBND_C */
extern void (*openqcd_sflds__assign_s2w[8])(int const *imb, int vol,
                                            openqcd__spinor const *s,
                                            openqcd__weyl *r);
extern void (*openqcd_sflds__add_assign_w2s[8])(int const *imb, int vol,
                                                openqcd__weyl const *s,
                                                openqcd__spinor *r);
extern void (*openqcd_sflds__sub_assign_w2s[8])(int const *imb, int vol,
                                                openqcd__weyl const *s,
                                                openqcd__spinor *r);
extern void (*openqcd_sflds__mulg5_sub_assign_w2s[8])(int const *imb, int vol,
                                                      openqcd__weyl const *s,
                                                      openqcd__spinor *r);

/* PBND_DBLE_C */
extern void (*openqcd_sflds__assign_sd2wd[8])(int const *imb, int vol,
                                              openqcd__spinor_dble const *sd,
                                              openqcd__weyl_dble *rd);
extern void (*openqcd_sflds__add_assign_wd2sd[8])(int const *imb, int vol,
                                                  openqcd__weyl_dble const *sd,
                                                  openqcd__spinor_dble *rd);
extern void (*openqcd_sflds__sub_assign_wd2sd[8])(int const *imb, int vol,
                                                  openqcd__weyl_dble const *sd,
                                                  openqcd__spinor_dble *rd);
extern void (*openqcd_sflds__mulg5_sub_assign_wd2sd[8])(
    int const *imb, int vol, openqcd__weyl_dble const *sd,
    openqcd__spinor_dble *rd);

/* SFLDS_C */
extern void openqcd_sflds__set_s2zero(int vol, openqcd__spinor *s);
extern void openqcd_sflds__set_sd2zero(int vol, openqcd__spinor_dble *sd);
extern void openqcd_sflds__random_s(int vol, openqcd__spinor *s, float sigma);
extern void openqcd_sflds__random_sd(int vol, openqcd__spinor_dble *sd,
                                     double sigma);
extern void openqcd_sflds__assign_s2s(int vol, openqcd__spinor const *s,
                                      openqcd__spinor *r);
extern void openqcd_sflds__assign_s2sd(int vol, openqcd__spinor const *s,
                                       openqcd__spinor_dble *rd);
extern void openqcd_sflds__assign_sd2s(int vol, openqcd__spinor_dble const *sd,
                                       openqcd__spinor *r);
extern void openqcd_sflds__assign_sd2sd(int vol, openqcd__spinor_dble const *sd,
                                        openqcd__spinor_dble *rd);
extern void openqcd_sflds__diff_s2s(int vol, openqcd__spinor const *s,
                                    openqcd__spinor *r);
extern void openqcd_sflds__add_s2sd(int vol, openqcd__spinor const *s,
                                    openqcd__spinor_dble *rd);
extern void openqcd_sflds__diff_sd2s(int vol, openqcd__spinor_dble const *sd,
                                     openqcd__spinor_dble const *rd,
                                     openqcd__spinor *r);

/* SCOM_C */
extern void openqcd_sflds__cps_int_bnd(int is, openqcd__spinor *s);
extern void openqcd_sflds__cps_ext_bnd(int is, openqcd__spinor *s);
extern void openqcd_sflds__free_sbufs(void);

/* SDCOM_C */
extern void openqcd_sflds__cpsd_int_bnd(int is, openqcd__spinor_dble *sd);
extern void openqcd_sflds__cpsd_ext_bnd(int is, openqcd__spinor_dble *sd);

#if defined(OPENQCD_INTERNAL)
/* PBND_C */
#define assign_s2w openqcd_sflds__assign_s2w
#define add_assign_w2s openqcd_sflds__add_assign_w2s
#define sub_assign_w2s openqcd_sflds__sub_assign_w2s
#define mulg5_sub_assign_w2s openqcd_sflds__mulg5_sub_assign_w2s

/* PBND_DBLE_C */
#define assign_sd2wd openqcd_sflds__assign_sd2wd
#define add_assign_wd2sd openqcd_sflds__add_assign_wd2sd
#define sub_assign_wd2sd openqcd_sflds__sub_assign_wd2sd
#define mulg5_sub_assign_wd2sd openqcd_sflds__mulg5_sub_assign_wd2sd

/* SFLDS_C */
#define set_s2zero(...) openqcd_sflds__set_s2zero(__VA_ARGS__)
#define set_sd2zero(...) openqcd_sflds__set_sd2zero(__VA_ARGS__)
#define random_s(...) openqcd_sflds__random_s(__VA_ARGS__)
#define random_sd(...) openqcd_sflds__random_sd(__VA_ARGS__)
#define assign_s2s(...) openqcd_sflds__assign_s2s(__VA_ARGS__)
#define assign_s2sd(...) openqcd_sflds__assign_s2sd(__VA_ARGS__)
#define assign_sd2s(...) openqcd_sflds__assign_sd2s(__VA_ARGS__)
#define assign_sd2sd(...) openqcd_sflds__assign_sd2sd(__VA_ARGS__)
#define diff_s2s(...) openqcd_sflds__diff_s2s(__VA_ARGS__)
#define add_s2sd(...) openqcd_sflds__add_s2sd(__VA_ARGS__)
#define diff_sd2s(...) openqcd_sflds__diff_sd2s(__VA_ARGS__)

/* SCOM_C */
#define cps_int_bnd(...) openqcd_sflds__cps_int_bnd(__VA_ARGS__)
#define cps_ext_bnd(...) openqcd_sflds__cps_ext_bnd(__VA_ARGS__)
#define free_sbufs(...) openqcd_sflds__free_sbufs(__VA_ARGS__)

/* SDCOM_C */
#define cpsd_int_bnd(...) openqcd_sflds__cpsd_int_bnd(__VA_ARGS__)
#define cpsd_ext_bnd(...) openqcd_sflds__cpsd_ext_bnd(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif
