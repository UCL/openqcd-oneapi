#define FORCE_UNSMEARING_C

#include "field_com.h"
#include "global.h"
#include "lattice.h"
#include "linalg.h"
#include "mdflds.h"
#include "stout_smearing.h"
#include "uflds.h"

static su3_dble w[4];

static su3_alg_dble *lambda_field = NULL;
static su3_dble *xi_field = NULL;

static void alloc_lambda_field(void)
{
  size_t n;

  error_root(sizeof(su3_alg_dble) != (8 * sizeof(double)), 1,
             "alloc_lambda_field [force_unsmearing.c]",
             "The su3_alg_dble structures are not properly packed");

  error(iup[0][0] == 0, 1, "alloc_lambda_field [force_unsmearing.c]",
        "Geometry arrays are not set");

  n = 4 * VOLUME + 7 * (BNDRY / 4);

  lambda_field = amalloc(n * sizeof(*lambda_field), ALIGN);
  error(lambda_field == NULL, 1, "alloc_lambda_field [force_unsmearing.c]",
        "Unable to allocate memory space for the lambda field");
}

static void alloc_xi_field(void)
{
  size_t n;

  error_root(sizeof(su3_dble) != (18 * sizeof(double)), 1,
             "alloc_xi_field [force_unsmearing.c]",
             "The su3_dble structures are not properly packed");

  error(iup[0][0] == 0, 1, "alloc_xi_field [force_unsmearing.c]",
        "Geometry arrays are not set");

  n = 4 * VOLUME + 7 * (BNDRY / 4);

  xi_field = amalloc(n * sizeof(*xi_field), ALIGN);
  error(xi_field == NULL, 1, "alloc_xi_field [force_unsmearing.c]",
        "Unable to allocate memory space for the gauge field");
}

/* Summary:
 *  Compute the coefficients r_{i,j} which is used in the computation of the B
 *  matrices. r1 and r2 are both size 3 arrays corresponding to the elements
 *  r_{1,j} and r_{2,j} respectively. Corresponds to eqs (60-65) in
 *  hep-lat/0311018.
 */
static void construct_r_coeffs(double u, double w, complex_dble *r1,
                               complex_dble *r2)
{
  double cu, su, c2u, s2u, cw, xi0w, xi1w;
  double uu, ww;

  /* Pre-computed expressions */

  cu = cos(u);
  su = sin(u);
  c2u = cos(2 * u);
  s2u = sin(2 * u);

  cw = cos(w);

  uu = square_dble(u);
  ww = square_dble(w);

  xi0w = smear_xi0_dble(w);
  xi1w = smear_xi1_dble(w);

  /* Compute r1 */

  r1[0].re =
      2 * (c2u * u + 8 * cu * cw * u - s2u * uu - 4 * cw * su * uu + s2u * ww +
           (cu * u * (3 * uu + ww) + su * (9 * uu + ww)) * xi0w);

  r1[0].im =
      -2 * (u * (-s2u + 8 * cw * su + 4 * cu * cw * u) + c2u * (-uu + ww) +
            (su * u * (3 * uu + ww) - cu * (9 * uu + ww)) * xi0w);

  r1[1].re = 2 * (s2u + cw * su + 2 * c2u * u + cu * cw * u) +
             (6 * cu * u + su * (-3 * uu + ww)) * xi0w;

  r1[1].im = -2 * (c2u + cw * su * u - cu * (cw + 4 * su * u)) +
             (-6 * su * u + cu * (-3 * uu + ww)) * xi0w;

  r1[2].re = (4 * cu - cw) * su + 3 * (su + cu * u) * xi0w;

  r1[2].im = -2 * c2u - cu * cw + 3 * (cu - su * u) * xi0w;

  /* Compute r2 */

  r2[0].re =
      -2 * c2u + 2 * u * ((su - 4 * cu * u) * xi0w + su * (cw + 3 * uu * xi1w));

  r2[0].im = 2 * u * (cu + 4 * su * u) * xi0w +
             2 * cu * (-2 * su + u * (cw + 3 * uu * xi1w));

  r2[1].re = -((cu + 2 * su * u) * xi0w) - cu * (cw - 3 * uu * xi1w);

  r2[1].im = (su - 2 * cu * u) * xi0w + su * (cw - 3 * uu * xi1w);

  r2[2].re = -(cu * xi0w) + 3 * su * u * xi1w;

  r2[2].im = su * xi0w + 3 * cu * u * xi1w;
}

/* Summary:
 *  Compute a single element b_{1,i} where i is determined by the arguments
 *  passed as r1, r2 and f. Corresponds to eq. (57) in hep-lat/0311018.
 */
static complex_dble compute_b1(double u, double w, complex_dble r1,
                               complex_dble r2, complex_dble f)
{
  double inv_denom, uu, ww;
  complex_dble b1;

  uu = u * u;
  ww = w * w;

  inv_denom = 1. / (2. * square_dble(9 * uu - ww));

  b1.re = -(2 * u * r1.re + (3 * uu - ww) * r2.re - 2 * (15 * uu + ww) * f.re) *
          inv_denom;
  b1.im = -(2 * u * r1.im + (3 * uu - ww) * r2.im - 2 * (15 * uu + ww) * f.im) *
          inv_denom;

  return b1;
}

/* Summary:
 *  Compute a single element b_{2,i} where i is determined by the arguments
 *  passed as r1, r2 and f. Corresponds to eq. (58) in hep-lat/0311018.
 */
static complex_dble compute_b2(double u, double w, complex_dble r1,
                               complex_dble r2, complex_dble f)
{
  double inv_denom;
  complex_dble b2;

  inv_denom = 1. / (2 * square_dble(9 * u * u - w * w));

  b2.re = -(r1.im - 3 * u * r2.im - 24 * u * f.im) * inv_denom;
  b2.im = (r1.re - 3 * u * r2.re - 24 * u * f.re) * inv_denom;

  return b2;
}

/* Summary:
 *   Compute the eigenvalue components of the smearing matrix
 *   pre-exponentiation. Corresponds to eqs. (23-25) in hep-lat/0311018.
 */
static void construct_uw(ch_drv0_t const *coeff, double *u, double *w)
{
  double theta_third, d_max, one_third;

  one_third = 1. / 3;

  d_max = 2 * pow((*coeff).t * one_third, 3. / 2);
  theta_third = acos((*coeff).d / d_max) * one_third;

  (*u) = sqrt((*coeff).t * one_third) * cos(theta_third);
  (*w) = sqrt((*coeff).t) * sin(theta_third);
}

/* Summary:
 *   Compute the b_{i,j} indices needed to compute the matrices B_{i}.
 *   Corresponds to eqs. (57-58) in hep-lat/0311018.
 */
static void construct_b_arrays(ch_drv0_t const *coeff, complex_dble *b1_array,
                               complex_dble *b2_array)
{
  int i;
  double u, w;
  complex_dble r1_array[3], r2_array[3];

  construct_uw(coeff, &u, &w);
  construct_r_coeffs(u, w, r1_array, r2_array);

  for (i = 0; i < 3; ++i) {
    b1_array[i] = compute_b1(u, w, r1_array[i], r2_array[i], (*coeff).p[i]);
    b2_array[i] = compute_b2(u, w, r1_array[i], r2_array[i], (*coeff).p[i]);
  }
}

/* Summary:
 *  Computes the B matrices needed to compute the Λ matrices needed for
 *  smearing. Corresponding to eq. (69) in hep-lat/0311018.
 *
 *  NOTE: X_pair should be const, but we can't make it const as ch2mat isn't
 *  const correct.
 */
static void construct_b_matrices(ch_mat_coeff_pair_t const *X_pair,
                                 su3_dble *b1, su3_dble *b2)
{
  complex_dble b1_array[3], b2_array[3];

  construct_b_arrays(&(*X_pair).coeff, b1_array, b2_array);

  ch2mat(b1_array, &(*X_pair).X, b1);
  ch2mat(b2_array, &(*X_pair).X, b2);
}

static void compute_unsmearing_gamma_matrix(su3_dble const *g_force,
                                            su3_dble const *u,
                                            ch_mat_coeff_pair_t const *X_pair,
                                            su3_dble *gamma)
{
  su3_dble B1, B2;
  su3_dble tmp[2];
  complex_dble tmp_idx[3];

  construct_b_matrices(X_pair, &B1, &B2);

  /* tmp[0] = U force */
  su3xsu3(u, g_force, tmp);

  tmp_idx[0].re = 0.;
  tmp_idx[0].im = 0.;

  cm3x3_tr(tmp, &B1, tmp_idx + 1);
  cm3x3_tr(tmp, &B2, tmp_idx + 2);

  /* Compute gamma = 0 + tr(tmp[0] B1) X + tr(tmp[0] B2) X^2 */
  ch2mat(tmp_idx, &(*X_pair).X, gamma);

  /* gamma += p1 * tmp[0] */
  cm3x3_mulc_add((*X_pair).coeff.p + 1, tmp, gamma);

  /* tmp[1] = U * force * X */
  su3xsu3alg(tmp, &(*X_pair).X, tmp + 1);

  /* gamma += p2 * tmp[1] */
  cm3x3_mulc_add((*X_pair).coeff.p + 2, tmp + 1, gamma);

  /* tmp[1] = X * U * force */
  su3algxsu3(&(*X_pair).X, tmp, tmp + 1);

  /* gamma += p2 * tmp[1] */
  cm3x3_mulc_add((*X_pair).coeff.p + 2, tmp + 1, gamma);
}

static void compute_lambda_field(su3_alg_dble const *force,
                                 su3_dble const *gfield,
                                 su3_dble const *sgfield,
                                 ch_mat_coeff_pair_t const *ch_coeffs)
{
  size_t ix, iy, mu;
  stout_smearing_params_t smear;
  su3_dble gamma_tmp;
  su3_dble g_force;

  if (lambda_field == NULL)
    alloc_lambda_field();

  smear = stout_smearing_parms();

  for (ix = 0; ix < VOLUME / 2; ++ix) {
    if (smear.smear_temporal == 1) {
      for (mu = 0; mu < 2; ++mu) {
        iy = 8 * ix + mu;
        su3dagxsu3alg(sgfield + iy, force + iy, &g_force);
        compute_unsmearing_gamma_matrix(&g_force, gfield + iy, ch_coeffs + iy,
                                        &gamma_tmp);
        project_to_su3alg(&gamma_tmp, lambda_field + iy);
      }
    }

    if (smear.smear_spatial == 1) {
      for (mu = 2; mu < 8; ++mu) {
        iy = 8 * ix + mu;
        su3dagxsu3alg(sgfield + iy, force + iy, &g_force);
        compute_unsmearing_gamma_matrix(&g_force, gfield + iy, ch_coeffs + iy,
                                        &gamma_tmp);
        project_to_su3alg(&gamma_tmp, lambda_field + iy);
      }
    }
  }

  if (smear.smear_temporal == 1) {
    copy_boundary_su3_alg_field(lambda_field);
  } else {
    copy_spatial_boundary_su3_alg_field(lambda_field);
  }
}

static void compute_xi_single_plaquette(su3_dble const *gfield, int plane_id,
                                        int pos_id, double prefactor)
{
  int plaq_ids[4];
  double p_mu_nu;

  plaq_uidx(plane_id, pos_id, plaq_ids);

  /* Compute the wedge
   *          ---<---
   *                |
   * w[0] =         ^
   *                |
   * used to compute four of the contributions to xi for link 0 and 2
   */

  su3xsu3dag(gfield + plaq_ids[1], gfield + plaq_ids[3], w);

  /* Two contributions to link 0 */
  su3xsu3dag(w, gfield + plaq_ids[2], w + 1);
  su3xsu3alg(w + 1, lambda_field + plaq_ids[2], w + 2);

  p_mu_nu = prefactor;
  cm3x3_mulr_add(&p_mu_nu, w + 2, xi_field + plaq_ids[0]);

  su3xsu3alg(w, lambda_field + plaq_ids[3], w + 1);
  su3xsu3dag(w + 1, gfield + plaq_ids[2], w + 2);

  p_mu_nu = prefactor;
  cm3x3_mulr_add(&p_mu_nu, w + 2, xi_field + plaq_ids[0]);

  /* Two contributions to link 2*/
  su3dagxsu3dag(w, gfield + plaq_ids[0], w + 1);
  su3xsu3alg(w + 1, lambda_field + plaq_ids[0], w + 2);

  p_mu_nu = prefactor;
  cm3x3_mulr_add(&p_mu_nu, w + 2, xi_field + plaq_ids[2]);

  su3dagxsu3alg(w, lambda_field + plaq_ids[1], w + 1);
  su3xsu3dag(w + 1, gfield + plaq_ids[0], w + 2);

  p_mu_nu = prefactor;
  cm3x3_mulr_add(&p_mu_nu, w + 2, xi_field + plaq_ids[2]);

  /* Compute the wedge
   *         ---<---
   *         |
   * w[0] =  v
   *         |
   * used to compute four of the contributions to link 0 and 1
   */

  su3dagxsu3dag(gfield + plaq_ids[3], gfield + plaq_ids[2], w);

  /* Two contributions to link 0*/
  su3algxsu3(lambda_field + plaq_ids[1], gfield + plaq_ids[1], w + 1);
  su3xsu3(w + 1, w, w + 2);

  p_mu_nu = -prefactor;
  cm3x3_mulr_add(&p_mu_nu, w + 2, xi_field + plaq_ids[0]);

  su3xsu3(gfield + plaq_ids[1], w, w + 1);
  su3xsu3alg(w + 1, lambda_field + plaq_ids[0], w + 2);

  p_mu_nu = -prefactor;
  cm3x3_mulr_add(&p_mu_nu, w + 2, xi_field + plaq_ids[0]);

  /* Two contributions to link 1*/
  su3xsu3alg(w, lambda_field + plaq_ids[2], w + 1);
  su3xsu3(w + 1, gfield + plaq_ids[0], w + 2);

  p_mu_nu = prefactor;
  cm3x3_mulr_add(&p_mu_nu, w + 2, xi_field + plaq_ids[1]);

  su3xsu3alg(w, lambda_field + plaq_ids[0], w + 1);
  su3xsu3(w + 1, gfield + plaq_ids[0], w + 2);

  p_mu_nu = -prefactor;
  cm3x3_mulr_add(&p_mu_nu, w + 2, xi_field + plaq_ids[1]);

  /* Compute the wedge
   *         |
   * w[0] =  v
   *         |
   *         --->---
   * used to compute four of the contributions to link 1 and 3
   */

  su3dagxsu3(gfield + plaq_ids[2], gfield + plaq_ids[0], w);

  /* Two contributions to link 1*/
  su3dagxsu3alg(gfield + plaq_ids[3], lambda_field + plaq_ids[3], w + 1);
  su3xsu3(w + 1, w, w + 2);

  p_mu_nu = prefactor;
  cm3x3_mulr_add(&p_mu_nu, w + 2, xi_field + plaq_ids[1]);

  su3dagxsu3(gfield + plaq_ids[3], w, w + 1);
  su3xsu3alg(w + 1, lambda_field + plaq_ids[1], w + 2);

  p_mu_nu = -prefactor;
  cm3x3_mulr_add(&p_mu_nu, w + 2, xi_field + plaq_ids[1]);

  /* Two contributions to link 3*/
  su3dagxsu3dag(gfield + plaq_ids[1], w, w + 1);
  su3xsu3alg(w + 1, lambda_field + plaq_ids[3], w + 2);

  p_mu_nu = -prefactor;
  cm3x3_mulr_add(&p_mu_nu, w + 2, xi_field + plaq_ids[3]);

  su3dagxsu3alg(gfield + plaq_ids[1], lambda_field + plaq_ids[1], w + 1);
  su3xsu3dag(w + 1, w, w + 2);

  p_mu_nu = prefactor;
  cm3x3_mulr_add(&p_mu_nu, w + 2, xi_field + plaq_ids[3]);

  /* Compute the wedge
   *               |
   * w[0] =        v
   *               |
   *         ---<---
   * used to compute four of the contributions to link 2 and 3
   */

  su3dagxsu3dag(gfield + plaq_ids[1], gfield + plaq_ids[0], w);

  /* Two contributions to link 2*/
  su3algxsu3(lambda_field + plaq_ids[3], gfield + plaq_ids[3], w + 1);
  su3xsu3(w + 1, w, w + 2);

  p_mu_nu = -prefactor;
  cm3x3_mulr_add(&p_mu_nu, w + 2, xi_field + plaq_ids[2]);

  su3xsu3(gfield + plaq_ids[3], w, w + 1);
  su3xsu3alg(w + 1, lambda_field + plaq_ids[2], w + 2);

  p_mu_nu = -prefactor;
  cm3x3_mulr_add(&p_mu_nu, w + 2, xi_field + plaq_ids[2]);

  /* Two contributions to link 3*/
  su3xsu3alg(w, lambda_field + plaq_ids[0], w + 1);
  su3xsu3(w + 1, gfield + plaq_ids[2], w + 2);

  p_mu_nu = prefactor;
  cm3x3_mulr_add(&p_mu_nu, w + 2, xi_field + plaq_ids[3]);

  su3xsu3alg(w, lambda_field + plaq_ids[2], w + 1);
  su3xsu3(w + 1, gfield + plaq_ids[2], w + 2);

  p_mu_nu = -prefactor;
  cm3x3_mulr_add(&p_mu_nu, w + 2, xi_field + plaq_ids[3]);
}

static void compute_xi_field(su3_alg_dble const *force, su3_dble const *gfield,
                             su3_dble const *sgfield,
                             ch_mat_coeff_pair_t const *ch_coeffs)
{
  int num_links, plane_id, ix;
  stout_smearing_params_t smear_params;

  smear_params = stout_smearing_parms();

  compute_lambda_field(force, gfield, sgfield, ch_coeffs);

  if (xi_field == NULL)
    alloc_xi_field();

  num_links = 4 * VOLUME + 7 * (BNDRY / 4);
  cm3x3_zero(num_links, xi_field);

  for (ix = 0; ix < VOLUME; ix++) {
    if (smear_params.smear_temporal == 1) {
      for (plane_id = 0; plane_id < 3; plane_id++) {
        compute_xi_single_plaquette(gfield, plane_id, ix,
                                    smear_params.rho_temporal);
      }
    }

    if (smear_params.smear_spatial == 1) {
      for (plane_id = 3; plane_id < 6; plane_id++) {
        compute_xi_single_plaquette(gfield, plane_id, ix,
                                    smear_params.rho_spatial);
      }
    }
  }

  if (smear_params.smear_temporal == 1) {
    add_boundary_su3_field(xi_field);
  } else {
    add_spatial_boundary_su3_field(xi_field);
  }
}

static void unsmear_single_force(su3_alg_dble *force, su3_dble const *gfield,
                                 su3_dble const *sgfield,
                                 ch_mat_coeff_pair_t const *ch_coeffs)
{
  size_t ix, iy, mu;
  su3_dble w[2];
  su3_dble exp_X;
  stout_smearing_params_t smear;

  compute_xi_field(force, gfield, sgfield, ch_coeffs);

  smear = stout_smearing_parms();

  for (ix = 0; ix < VOLUME / 2; ++ix) {
    if (smear.smear_temporal == 1) {
      for (mu = 0; mu < 2; ++mu) {
        iy = 8 * ix + mu;
        ch2mat(ch_coeffs[iy].coeff.p, &ch_coeffs[iy].X, &exp_X);
        su3dagxsu3alg(sgfield + iy, force + iy, w);
        su3xsu3(w, &exp_X, w + 1);

        cm3x3_add(xi_field + iy, w + 1);
        prod2su3alg(gfield + iy, w + 1, force + iy);
      }
    }

    if (smear.smear_spatial == 1) {
      for (mu = 2; mu < 8; ++mu) {
        iy = 8 * ix + mu;
        ch2mat(ch_coeffs[iy].coeff.p, &ch_coeffs[iy].X, &exp_X);
        su3dagxsu3alg(sgfield + iy, force + iy, w);
        su3xsu3(w, &exp_X, w + 1);

        cm3x3_add(xi_field + iy, w + 1);
        prod2su3alg(gfield + iy, w + 1, force + iy);
      }
    }
  }
}

/* Summary:
 *   Takes a force term computed with smeared links and applies the recursive
 *   unsmearing formula to get the correct force term.
 */
void unsmear_force(su3_alg_dble *force)
{
  int i, sign_flipped_q;
  su3_dble **sfields;
  ch_mat_coeff_pair_t **ch_coeffs;
  stout_smearing_params_t smear_params;

  smear_params = stout_smearing_parms();

  error(smear_params.num_smear == 0, 1, "unsmear_force [force_unsmearing.c]",
        "Stout smearing is not turned on, there is no need to unsmear the MD "
        "forces.");

  error(query_flags(SMEARED_UD_UP2DATE) != 1, 1,
        "unsmear_force [force_unsmearing.c]",
        "The stout links are not up to date with the thin links");

  /* Make sure that the fields are in a smeared state, this will not compute a
   * new set of smeared fields as we have already checked that the smeared
   * fields have already been computed */
  smear_fields();

  sign_flipped_q = query_flags(UD_PHASE_SET);

  if (sign_flipped_q)
    unset_ud_phase();

  sfields = smeared_fields();
  ch_coeffs = smearing_ch_coeff_fields();

  i = smear_params.num_smear - 1;

  unsmear_single_force(force, sfields[i], udfld(), ch_coeffs[i]);

  for (--i; i >= 0; --i) {
    unsmear_single_force(force, sfields[i], sfields[i + 1], ch_coeffs[i]);
  }

  if (sign_flipped_q == 1)
    set_ud_phase();
}

/* Summary:
 *   Applies the unsmearing procedure to what is currently stored in the global
 *   mdfld.frc storage.
 *
 * Effects:
 *   * modifies the contents of mdfld.frc
 *   * (possibly changes a flag that indicates whether the forces have been
 *     unsmeared)
 */
void unsmear_mdforce(void) { unsmear_force((*mdflds()).frc); }
