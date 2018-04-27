
/*******************************************************************************
 *
 * File sw_term.h
 *
 * Copyright (C) 2005, 2009, 2011 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef SW_TERM_H
#define SW_TERM_H

#include "su3.h"
#include "utils.h"

/* PAULI_C */
extern void openqcd_sw_term__mul_pauli(float mu, openqcd__pauli const *m,
                                       openqcd__weyl const *s,
                                       openqcd__weyl *r);
extern void openqcd_sw_term__mul_pauli2(float mu, openqcd__pauli const *m,
                                        openqcd__spinor const *s,
                                        openqcd__spinor *r);
extern void openqcd_sw_term__assign_pauli(int vol,
                                          openqcd__pauli_dble const *md,
                                          openqcd__pauli *m);
extern void openqcd_sw_term__apply_sw(int vol, float mu,
                                      openqcd__pauli const *m,
                                      openqcd__spinor const *s,
                                      openqcd__spinor *r);

/* PAULI_DBLE_C */
extern void openqcd_sw_term__mul_pauli_dble(double mu,
                                            openqcd__pauli_dble const *m,
                                            openqcd__weyl_dble const *s,
                                            openqcd__weyl_dble *r);
extern void openqcd_sw_term__mul_pauli2_dble(double mu,
                                             openqcd__pauli_dble const *m,
                                             openqcd__weyl_dble const *s,
                                             openqcd__weyl_dble *r);
extern int openqcd_sw_term__inv_pauli_dble(double mu,
                                           openqcd__pauli_dble const *m,
                                           openqcd__pauli_dble *im);
extern openqcd__complex_dble
openqcd_sw_term__det_pauli_dble(double mu, openqcd__pauli_dble const *m);
extern void openqcd_sw_term__apply_sw_dble(int vol, double mu,
                                           openqcd__pauli_dble const *m,
                                           openqcd__spinor_dble const *s,
                                           openqcd__spinor_dble *r);
extern int openqcd_sw_term__apply_swinv_dble(int vol, double mu,
                                             openqcd__pauli_dble const *m,
                                             openqcd__spinor_dble const *s,
                                             openqcd__spinor_dble *r);

/* SWFLDS_C */
extern openqcd__pauli *openqcd_sw_term__swfld(void);
extern openqcd__pauli_dble *openqcd_sw_term__swdfld(void);
extern void openqcd_sw_term__free_sw(void);
extern void openqcd_sw_term__free_swd(void);
extern void openqcd_sw_term__assign_swd2sw(void);

/* SW_TERM_C */
extern int openqcd_sw_term__sw_term(openqcd_utils__ptset_t set);

#if defined(OPENQCD_INTERNAL)
/* PAULI_C */
#define mul_pauli(...) openqcd_sw_term__mul_pauli(__VA_ARGS__)
#define mul_pauli2(...) openqcd_sw_term__mul_pauli2(__VA_ARGS__)
#define assign_pauli(...) openqcd_sw_term__assign_pauli(__VA_ARGS__)
#define apply_sw(...) openqcd_sw_term__apply_sw(__VA_ARGS__)

/* PAULI_DBLE_C */
#define mul_pauli_dble(...) openqcd_sw_term__mul_pauli_dble(__VA_ARGS__)
#define mul_pauli2_dble(...) openqcd_sw_term__mul_pauli2_dble(__VA_ARGS__)
#define inv_pauli_dble(...) openqcd_sw_term__inv_pauli_dble(__VA_ARGS__)
#define det_pauli_dble(...) openqcd_sw_term__det_pauli_dble(__VA_ARGS__)
#define apply_sw_dble(...) openqcd_sw_term__apply_sw_dble(__VA_ARGS__)
#define apply_swinv_dble(...) openqcd_sw_term__apply_swinv_dble(__VA_ARGS__)

/* SWFLDS_C */
#define swfld(...) openqcd_sw_term__swfld(__VA_ARGS__)
#define swdfld(...) openqcd_sw_term__swdfld(__VA_ARGS__)
#define free_sw(...) openqcd_sw_term__free_sw(__VA_ARGS__)
#define free_swd(...) openqcd_sw_term__free_swd(__VA_ARGS__)
#define assign_swd2sw(...) openqcd_sw_term__assign_swd2sw(__VA_ARGS__)

/* SW_TERM_C */
#define sw_term(...) openqcd_sw_term__sw_term(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif
