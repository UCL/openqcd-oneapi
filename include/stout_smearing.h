
/*******************************************************************************
 *
 * File stout_smearing.h
 *
 * Author (2017, 2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef STOUT_SMEARING_H
#define STOUT_SMEARING_H

#include "flags.h"
#include "su3.h"
#include "su3fcts.h"

typedef struct
{
  openqcd__su3_alg_dble X;
  ch_drv0_t coeff;
} openqcd_stout_smearing__ch_mat_coeff_pair_t;

/* FORCE_UNSMEARING_C */
extern void openqcd_stout_smearing__unsmear_force(openqcd__su3_alg_dble *force);
extern void openqcd_stout_smearing__unsmear_mdforce(void);

/* SMEARED_FIELDS_C */
extern openqcd__su3_dble **openqcd_stout_smearing__smeared_fields(void);
extern void openqcd_stout_smearing__free_smearing_ch_coeff_fields(void);
extern openqcd_stout_smearing__ch_mat_coeff_pair_t **
openqcd_stout_smearing__smearing_ch_coeff_fields(void);

/* STOUT_SMEARING_C */
extern void openqcd_stout_smearing__smear_fields(void);
extern void openqcd_stout_smearing__unsmear_fields(void);

#if defined(OPENQCD_INTERNAL)
#define ch_mat_coeff_pair_t openqcd_stout_smearing__ch_mat_coeff_pair_t

/* FORCE_UNSMEARING_C */
#define unsmear_force(...) openqcd_stout_smearing__unsmear_force(__VA_ARGS__)
#define unsmear_mdforce(...)                                                   \
  openqcd_stout_smearing__unsmear_mdforce(__VA_ARGS__)

/* SMEARED_FIELDS_C */
#define smeared_fields(...) openqcd_stout_smearing__smeared_fields(__VA_ARGS__)
#define free_smearing_ch_coeff_fields(...)                                     \
  openqcd_stout_smearing__free_smearing_ch_coeff_fields(__VA_ARGS__)
#define smearing_ch_coeff_fields(...)                                          \
  openqcd_stout_smearing__smearing_ch_coeff_fields(__VA_ARGS__)

/* STOUT_SMEARING_C */
#define smear_fields(...) openqcd_stout_smearing__smear_fields(__VA_ARGS__)
#define unsmear_fields(...) openqcd_stout_smearing__unsmear_fields(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif /* STOUT_SMEARING_H */
