#define STOUT_SMEARING_C

#include "stout_smearing.h"
#include "field_com.h"
#include "global.h"
#include "lattice.h"
#include "linalg.h"
#include "uflds.h"

static su3_dble w1, w2, w3;
static su3_dble *omega_matrix = NULL;

static const int dir_unapply_smearing = -1;
static const int dir_apply_smearing = 1;

static void alloc_omega_field(void)
{
  size_t n;

  error_root(sizeof(su3_dble) != (18 * sizeof(double)), 1,
             "alloc_omega_field [stout_smearing.c]",
             "The su3_dble structures are not properly packed");

  error(iup[0][0] == 0, 1, "alloc_omega_field [stout_smearing.c]",
        "Geometry arrays are not set");

  n = 4 * VOLUME + 7 * (BNDRY / 4);

  omega_matrix = amalloc(n * sizeof(*omega_matrix), ALIGN);
  error(omega_matrix == NULL, 1, "alloc_omega_field [stout_smearing.c]",
        "Unable to allocate memory space for the gauge field");
}

/* Summary:
 *   Compute the contribution to the omega matrix from a single plaquette to its
 *   four adjacent gauge links, and multiply the contibution by a real
 *   prefactor.
 */
static void compute_omega_contribution(su3_dble const *gfield, int plane_id,
                                       int pos_id, double prefactor)
{
  int plaq_link_ids[4];

  plaq_uidx(plane_id, pos_id, plaq_link_ids);

  /* Compute the two wedges
   *       --->---
   *       |                    |
   * w1 =  ^      , w2 =        ^
   *       |                    |
   *       *              *-->---
   * used to compute the contribution to update link 0 (w1 * w2^dag)
   * and link 2 (w2 * w1^dag).
   */
  su3xsu3(gfield + plaq_link_ids[2], gfield + plaq_link_ids[3], &w1);
  su3xsu3(gfield + plaq_link_ids[0], gfield + plaq_link_ids[1], &w2);

  su3xsu3dag(&w1, &w2, &w3);
  cm3x3_mulr_add(&prefactor, &w3, omega_matrix + plaq_link_ids[0]);

  su3xsu3dag(&w2, &w1, &w3);
  cm3x3_mulr_add(&prefactor, &w3, omega_matrix + plaq_link_ids[2]);

  /* Compute the two wedges
   *                       ---<---
   *       |                     |
   * w1 =  ^      , w2 =         ^
   *       |                     |
   *       ---<--*               *
   * used to compute the contribution to update link 1 (w1 * w2^dag)
   * and link 3 (w1^dag * w2)
   */
  su3dagxsu3(gfield + plaq_link_ids[0], gfield + plaq_link_ids[2], &w1);
  su3xsu3dag(gfield + plaq_link_ids[1], gfield + plaq_link_ids[3], &w2);

  su3xsu3dag(&w1, &w2, &w3);
  cm3x3_mulr_add(&prefactor, &w3, omega_matrix + plaq_link_ids[1]);

  su3dagxsu3(&w1, &w2, &w3);
  cm3x3_mulr_add(&prefactor, &w3, omega_matrix + plaq_link_ids[3]);
}

static void compute_omega_field(su3_dble const *gfield)
{
  int num_links, plane_id, ix;
  stout_smearing_params_t smear_params;

  smear_params = stout_smearing_parms();

  if (omega_matrix == NULL) {
    alloc_omega_field();
  }

  num_links = 4 * VOLUME + 7 * (BNDRY / 4);
  cm3x3_zero(num_links, omega_matrix);

  for (ix = 0; ix < VOLUME; ix++) {
    if (smear_params.smear_temporal == 1) {
      for (plane_id = 0; plane_id < 3; plane_id++) {
        compute_omega_contribution(gfield, plane_id, ix,
                                   smear_params.rho_temporal);
      }
    }

    if (smear_params.smear_spatial == 1) {
      for (plane_id = 3; plane_id < 6; plane_id++) {
        compute_omega_contribution(gfield, plane_id, ix,
                                   smear_params.rho_spatial);
      }
    }
  }

  if (smear_params.smear_temporal == 1) {
    add_boundary_su3_field(omega_matrix);
  } else {
    add_spatial_boundary_su3_field(omega_matrix);
  }
}

static void smear_single_field(su3_dble *gfield, ch_mat_coeff_pair_t *ch_coeffs)
{
  int ix, iy, mu;
  stout_smearing_params_t smear;

  smear = stout_smearing_parms();

  compute_omega_field(gfield);

  for (ix = 0; ix < VOLUME / 2; ++ix) {
    if (smear.smear_temporal == 1) {
      for (mu = 0; mu < 2; ++mu) {
        iy = 8 * ix + mu;
        project_to_su3alg(omega_matrix + iy, &ch_coeffs[iy].X);
        expXsu3_w_factors(1., &ch_coeffs[iy].X, gfield + iy,
                          &ch_coeffs[iy].coeff);
      }
    }

    if (smear.smear_spatial == 1) {
      for (mu = 2; mu < 8; ++mu) {
        iy = 8 * ix + mu;
        project_to_su3alg(omega_matrix + iy, &ch_coeffs[iy].X);
        expXsu3_w_factors(1., &ch_coeffs[iy].X, gfield + iy,
                          &ch_coeffs[iy].coeff);
      }
    }
  }

  if (smear.smear_temporal == 1) {
    copy_boundary_su3_field(gfield);
  } else {
    copy_spatial_boundary_su3_field(gfield);
  }
}

/* Summary:
 *   Cycle the field stored in udfld with those stored in smeared_fields.
 *
 * Details:
 *   If direction > 0 then
 *       udfld() = 0, smeared_fields() = [1, 2, 3, ..., m]
 *     will be
 *       udfld() = m, smeared_fields() = [0, 1, 2, ..., m-1]
 *     which corresponds to smearing.
 *
 *   If direction <= 0 then
 *       udfld() = m, smeared_fields() = [0, 1, 2, ..., m-1]
 *     will be
 *       udfld() = 0, smeared_fields() = [1, 2, 3, ..., m]
 *     which corresponds to unsmearing
 */
static void cycle_smeared_fields(int direction)
{
  int begin, dir, end, sign_flipped_q;
  stout_smearing_params_t smear_params;
  su3_dble **sfields;

  smear_params = stout_smearing_parms();

  if (direction > 0) {
    begin = 0;
    end = smear_params.num_smear;
    dir = 1;
    /*message("Cycle smearing\n");*/
  } else {
    begin = smear_params.num_smear - 1;
    end = -1;
    dir = -1;
    /*message("Cycle unsmearing\n");*/
  }

  /* Return early if there is no smearing */
  if (begin == end) {
    return;
  }

  sfields = smeared_fields();

  sign_flipped_q = query_flags(UD_PHASE_SET);
  if (sign_flipped_q == 1) {
    unset_ud_phase();
  }

  for (; begin != end; begin += dir) {
    swap_udfld(sfields + begin);
  }

  if (sign_flipped_q == 1) {
    set_ud_phase();
  }
}

/* Summary:
 *   Main smearing routine that computes the smeared fields and applies them to
 *   the configuration stored in udfld().
 */
static void compute_and_apply_smearing(void)
{
  int nsmear, i, sign_flipped_q;
  long num_links;
  stout_smearing_params_t smear_params;
  su3_dble **sfields;
  su3_dble *udb;
  ch_mat_coeff_pair_t **ch_coeffs;

  smear_params = stout_smearing_parms();
  nsmear = smear_params.num_smear;

  num_links = 4 * VOLUME + 7 * BNDRY / 4;

  if (nsmear > 0) {
    if (query_flags(UDBUF_UP2DATE) != 1) {
      copy_bnd_ud();
    }

    /* Default boundaries before smearing */
    sign_flipped_q = query_flags(UD_PHASE_SET);
    if (sign_flipped_q == 1) {
      unset_ud_phase();
    }

    sfields = smeared_fields();
    ch_coeffs = smearing_ch_coeff_fields();

    for (i = 0; i < nsmear; i++) {
      udb = udfld();
      cm3x3_assign(num_links, udb, sfields[i]);
      smear_single_field(udb, ch_coeffs[i]);
    }

    /* Reapply boundaries after smearing */
    if (sign_flipped_q != 0) {
      set_ud_phase();
    }
  }

  set_flags(SMEARED_UD);
}

/* Summary:
 *   Checks a flag to see if the gauge field has been updated since it was
 *   called last. If the fields have been updated, then compute the N-smeared
 *   fields and store them in the smeared field container. The current udfld is
 *   placed as the first element of this container, while udfld is replaced by
 *   the smeared links. This routine should do what you expect regardless of
 *   whatever has been done previously. This means that if the config is already
 *   smeared and the fields haven't been updated, the routine returns. If the
 *   fields are currently smeared but are out of date (somehow), the fields are
 *   unsmeared first. If the field is unsmeared but the thin links haven't been
 *   updated, the previous smeared field is used.
 *
 * Effects:
 *   * udfld will be the smeared fields
 *   * the event SMEARED_UD is called
 */
void smear_fields(void)
{
  stout_smearing_params_t smear_params;
  smear_params = stout_smearing_parms();

  if (smear_params.num_smear == 0) {
    return;
  }

  error(bc_type() != 3, 1, "smear fields [stout_smearing.c]",
        "Stout smearing is only implemented for periodic boundaries");

  if (query_flags(UD_IS_SMEARED) == 1) {
    if (query_flags(SMEARED_UD_UP2DATE) == 1) {
      return;
    } else {
      unsmear_fields();
    }
  } else {
    if (query_flags(SMEARED_UD_UP2DATE) == 1) {
      cycle_smeared_fields(dir_apply_smearing);
      set_flags(SMEARED_UD);
      return;
    }
  }

  compute_and_apply_smearing();
}

/* Summary:
 *   Resets the unsmeared fields to its state before smearing. I.e. udfld is set
 *   to be the first element of the smeared field container, while its contents
 *   (presumably smeared links) is stored as the last element of the smeared
 *   field container.
 *
 * Effects:
 *   * udfld is reset to be the unsmeared field
 *   * sets a flag signalling the the field is no longer smeared
 *   * (might have to change the communication flag)
 */
void unsmear_fields(void)
{
  if (query_flags(UD_IS_SMEARED) != 1) {
    return;
  }

  cycle_smeared_fields(dir_unapply_smearing);
  set_flags(UNSMEARED_UD);
}
