#ifndef STOUT_SMEARING_H
#define STOUT_SMEARING_H

#ifndef SU3_H
#include "su3.h"
#endif

#include "su3fcts.h"

typedef struct
{
  su3_alg_dble X;
  ch_drv0_t coeff;
} ch_mat_coeff_pair_t;

/* SOUT_FIELDS_COMMUNICATION_C */
extern void add_boundary_su3_field(su3_dble *su3_field);
extern void communicate_boundary_su3_alg_field(su3_alg_dble *alg_field);

/* STOUT_SMEARING_C */
extern void smear_fields(void);
extern void unsmear_fields(void);
extern void unsmear_force(su3_alg_dble *force);
extern void unsmear_mdforce(void);

/* SMEARED_FIELDS_C */
extern void allocate_smeared_fields(void);
extern void free_smeared_fields(void);
extern su3_dble **smeared_fields(void);
extern void allocate_smearing_ch_coeff_fields(void);
extern void free_smearing_ch_coeff_fields(void);
extern ch_mat_coeff_pair_t **smearing_ch_coeff_fields(void);

#endif /* STOUT_SMEARING_H */
