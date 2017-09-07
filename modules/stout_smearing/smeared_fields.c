#define SMEARED_FIELDS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "stout_smearing.h"
#include "flags.h"
#include "global.h"

static su3_dble **sfields = NULL;
static ch_mat_coeff_pair_t **exp_ch_mat_coeff_pair_field = NULL;

/* Summary:
 *   Allocated the necessary memory to store the smeared fields. For the
 *   unsmearing procedure to work, we need to store every smearing itteration.
 *   Therefore the size of the smeared fields workspace is:
 *     n_iterations * (4 * vol + 7 * boundary / 4) * sizeof(su3)
 */
void allocate_smeared_fields(void)
{
  int nsmear, nlinks, i;
  stout_smearing_params_t smear_params;

  smear_params = sout_smearing_parms();
  nsmear = smear_params.num_smear;

  if (nsmear > 0) {
    error(iup[0][0] == 0, 1, "allocate_smeared_fields [smeared_fields.c]",
          "Geometry arrays are not set");

    nlinks = 4 * VOLUME + (7 * BNDRY / 4);

    sfields = malloc(nsmear * sizeof(*sfields));

    error(sfields == NULL, 1, "allocate_smeared_fields [smeared_fields.c]",
          "Unable to allocate smeared field pointer array");

    sfields[0] = amalloc(nsmear * nlinks * sizeof(**sfields), ALIGN);

    error(sfields[0] == NULL, 1, "allocate_smeared_fields [smeared_fields.c]",
          "Unable to allocate memory space for the smeared gauge fields");

    for (i = 1; i < nsmear; i++)
      sfields[i] = sfields[i - 1] + nlinks;
  }
}

/* Summary:
 *   Free the memory allocated for the smeared field configurations.
 *   XXX: Have to make sure that we do not free the thin links, or that we at
 *        least do not double free the udfld
 */
void free_smeared_fields(void)
{
  /* TODO: Test if is smeared */
  if (sfields != NULL)
    afree(sfields[0]);
}

/* Summary:
 *   Return a pointer to the array of smeared field configurations. If the
 *   memory hasn't been allocated yet, allocate the necessary memory before
 *   returning.
 */
su3_dble **smeared_fields(void)
{
  if (sfields == NULL)
    allocate_smeared_fields();

  return sfields;
}

/* Summary:
 *   Allocate the necessary memory to store the Cayley-Hamilton coefficients for
 *   the exponentiation of the smearing-matrix. These coefficients are reused
 *   for the force unsmearing algorithm and should therefore not be unecessarily
 *   recomputed. There is no real reason to store all of ch_drv0_t as we do not
 *   need t and d, but from a programming perspective it makes more sense.
 */
void allocate_smearing_ch_coeff_fields(void)
{
  int nsmear, nlinks, i;
  stout_smearing_params_t smear_params;

  smear_params = sout_smearing_parms();
  nsmear = smear_params.num_smear;

  if (nsmear > 0) {
    error(iup[0][0] == 0, 1,
          "allocate_smearing_ch_coeff_fields [smeared_fields.c]",
          "Geometry arrays are not set");

    nlinks = 4 * VOLUME;

    exp_ch_mat_coeff_pair_field =
        malloc(nsmear * sizeof(*exp_ch_mat_coeff_pair_field));

    error(exp_ch_mat_coeff_pair_field == NULL, 1,
          "allocate_smearing_ch_coeff_fields [smeared_fields.c]",
          "Unable to allocate Cayley-Hamilton coefficient field pointer array");

    /* TODO: Check and think about memory alignment for this type */
    exp_ch_mat_coeff_pair_field[0] =
        amalloc(nsmear * nlinks * sizeof(**exp_ch_mat_coeff_pair_field), ALIGN);

    error(exp_ch_mat_coeff_pair_field[0] == NULL, 1,
          "allocate_smearing_ch_coeff_fields [smeared_fields.c]",
          "Unable to allocate memory space for the Cayley-Hamilton coefficient "
          "fields");

    for (i = 1; i < nsmear; i++)
      exp_ch_mat_coeff_pair_field[i] =
          exp_ch_mat_coeff_pair_field[i - 1] + nlinks;
  }
}

/* Summary:
 *   Free the memory used by the Cayley-Hamilton coefficients needed for
 *   smearing and unsmearing.
 */
void free_smearing_ch_coeff_fields(void)
{
  if (exp_ch_mat_coeff_pair_field != NULL)
    afree(exp_ch_mat_coeff_pair_field[0]);
}

/* Summary:
 *   Return a pointer to the array of Cayley-Hamilton coefficients for the
 *   smeared fields. It allocated the necessary memory if this hasn't already
 *   happened.
 */
ch_mat_coeff_pair_t **smearing_ch_coeff_fields(void)
{
  if (exp_ch_mat_coeff_pair_field == NULL)
    allocate_smearing_ch_coeff_fields();

  return exp_ch_mat_coeff_pair_field;
}
