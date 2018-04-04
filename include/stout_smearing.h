
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
  su3_alg_dble X;
  ch_drv0_t coeff;
} ch_mat_coeff_pair_t;

/* FORCE_UNSMEARING_C */
extern void unsmear_force(su3_alg_dble *force);
extern void unsmear_mdforce(void);

/* SMEARED_FIELDS_C */
extern su3_dble **smeared_fields(void);
extern void free_smearing_ch_coeff_fields(void);
extern ch_mat_coeff_pair_t **smearing_ch_coeff_fields(void);

/* STOUT_SMEARING_C */
extern void smear_fields(void);
extern void unsmear_fields(void);

#endif /* STOUT_SMEARING_H */
