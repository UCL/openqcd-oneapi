
/*
 * Created: 10-08-2017
 * Modified:
 * Author: Jonas R. Glesaaen (jonas@glesaaen.com)
 */

#define MAIN_PROGRAM

#include "global.h"
#include "lattice.h"
#include "mpi.h"
#include "stout_smearing.h"
#include <time.h>

#include <devel/testing_utilities/data_type_diffs.c>
#include <devel/testing_utilities/diff_printing.c>
#include <devel/testing_utilities/test_counter.c>
#include <modules/stout_smearing/force_unsmearing.c>

#include "inputs/test08_inputs.c"

int get_link_index(int pos, int dir)
{
  if (pos < (VOLUME / 2)) {
    pos = iup[pos][dir];
    return 8 * (pos - (VOLUME / 2)) + 2 * dir + 1;
  } else {
    return 8 * (pos - (VOLUME / 2)) + 2 * dir;
  }
}

void get_staple_dirs(int staple_dirs[], int mu)
{
  switch (mu) {
  case 1:
    staple_dirs[0] = 2;
    staple_dirs[1] = 3;
    break;
  case 2:
    staple_dirs[0] = 1;
    staple_dirs[1] = 3;
    break;
  case 3:
    staple_dirs[0] = 1;
    staple_dirs[1] = 2;
    break;
  default:
    error(1, 1, "get_staple_dirs", "Only implemented for dirs {1,2,3}");
  }
}

void get_staple_indices(int staple_pos[], int pos, int mu, int nu)
{
  staple_pos[0] = get_link_index(pos, nu);
  staple_pos[1] = get_link_index(iup[pos][nu], mu);
  staple_pos[2] = get_link_index(iup[pos][mu], nu);

  pos = idn[pos][nu];
  staple_pos[3] = get_link_index(pos, nu);
  staple_pos[4] = get_link_index(pos, mu);
  staple_pos[5] = get_link_index(iup[pos][mu], nu);
}

void fill_link_staple_array(su3_dble *staples[], su3_dble *gfield, int ix,
                            int mu)
{
  int nu, i, n;
  int staple_dirs[2], staple_idx[6];
  get_staple_dirs(staple_dirs, mu);

  n = 0;
  for (nu = 0; nu < 2; ++nu) {
    get_staple_indices(staple_idx, ix, mu, staple_dirs[nu]);

    for (i = 0; i < 6; ++i)
      staples[i + n] = gfield + staple_idx[i];

    n += 6;
  }
}

void fill_lambda_staple_array(su3_alg_dble *staples[], su3_alg_dble *lambda,
                              int ix, int mu)
{
  int nu, i, n;
  int staple_dirs[2], staple_idx[6];
  get_staple_dirs(staple_dirs, mu);

  n = 0;
  for (nu = 0; nu < 2; ++nu) {
    get_staple_indices(staple_idx, ix, mu, staple_dirs[nu]);

    for (i = 0; i < 6; ++i)
      staples[i + n] = lambda + staple_idx[i];

    n += 6;
  }
}

void compute_xi_fixed_lambda(su3_dble const *gfield)
{
  int num_links, plane_id, ix;
  stout_smearing_params_t smear_params;

  smear_params = stout_smearing_parms();

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

  add_boundary_su3_field(xi_field);
}

int main(int argc, char *argv[])
{
  int my_rank;
  int ix, mu, link_idx;
  int test_failed;
  int staple_idx_array[6], plaq_ids[4];
  double diff;
  double theta[3] = {0.0, 0.0, 0.0};
  su3_dble *staples[12];
  su3_alg_dble *lambda_staples[12];
  su3_dble expected_Xi;
  su3_alg_dble link_force;
  char failed_name[256];

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  srand(time(NULL));

  error(NPROC > 1, 1, "test08",
        "Test not yet implemented for multi-thread runs");

  new_test_module();

  if (my_rank == 0) {
    printf("Checks of the programs in the module stout_smearing\n");
    printf("-------------------------------------------\n\n");

    printf("%dx%dx%dx%d lattice, ", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid, ", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n", L0, L1, L2, L3);
    printf("\n-------------------------------------------\n\n");
  }

  set_bc_parms(3, 0., 0., 0., 0., NULL, NULL, theta);
  set_stout_smearing_parms(1, 0., 0.14, 1, 1);

  geometry();

  alloc_lambda_field();
  alloc_xi_field();

  { /* Test 0 */
    if (my_rank == 0) {
      register_test(0, "Test index manipulation");
      print_test_header(0);

      ix = rand() % VOLUME;

      link_idx = get_link_index(ix, 1);

      get_staple_indices(staple_idx_array, ix, 1, 2);
      plaq_uidx(5, ix, plaq_ids);

      printf("staple_idx[0] == plaq_ids[2] = %d\n",
             staple_idx_array[0] == plaq_ids[2]);
      printf("staple_idx[1] == plaq_ids[3] = %d\n",
             staple_idx_array[1] == plaq_ids[3]);
      printf("staple_idx[2] == plaq_ids[1] = %d\n",
             staple_idx_array[2] == plaq_ids[1]);
      printf("link_idx      == plaq_ids[0] = %d\n", link_idx == plaq_ids[0]);

      fail_test_if(0, staple_idx_array[0] != plaq_ids[2] ||
                          staple_idx_array[1] != plaq_ids[3] ||
                          staple_idx_array[2] != plaq_ids[1] ||
                          link_idx != plaq_ids[0]);

      plaq_uidx(5, idn[ix][2], plaq_ids);

      printf("staple_idx[3] == plaq_ids[2] = %d\n",
             staple_idx_array[3] == plaq_ids[2]);
      printf("staple_idx[4] == plaq_ids[0] = %d\n",
             staple_idx_array[4] == plaq_ids[0]);
      printf("staple_idx[5] == plaq_ids[1] = %d\n",
             staple_idx_array[5] == plaq_ids[1]);
      printf("link_idx      == plaq_ids[3] = %d\n", link_idx == plaq_ids[3]);

      fail_test_if(0, staple_idx_array[3] != plaq_ids[2] ||
                          staple_idx_array[4] != plaq_ids[0] ||
                          staple_idx_array[5] != plaq_ids[1] ||
                          link_idx != plaq_ids[3]);

      printf("\n-------------------------------------------\n\n");
    }
  }

  { /* Test 1 */
    if (my_rank == 0) {
      register_test(1, "Test Xi matrix field");
      print_test_header(1);

      printf("Looking for mismatches...");
      fflush(stdout);

      test_failed = 0;
      for (ix = 0; ix < VOLUME; ix += 13) {

        for (mu = 0; mu < 4; ++mu) {
          random_alg(4 * VOLUME, lambda_field);
          random_ud();

          link_idx = get_link_index(ix, mu);
          get_target_link_inputs(udfld() + link_idx, &link_force,
                                 lambda_field + link_idx);

          if (mu != 0) {
            fill_link_staple_array(staples, udfld(), ix, mu);
            fill_lambda_staple_array(lambda_staples, lambda_field, ix, mu);

            get_staple_inputs(staples, lambda_staples);
          }

          compute_xi_fixed_lambda(udfld());

          if (mu == 0) {
            cm3x3_zero(1, &expected_Xi);
          } else {
            get_expected_Xi(&expected_Xi);
          }

          diff = norm_diff_su3(xi_field + link_idx, &expected_Xi);

          if (diff > 1e-10) {
            fail_test(1);
            test_failed = 1;

            printf(" found\n");
            sprintf(failed_name, "Xi[%d][%d]", ix, mu);
            report_su3_diff(xi_field + link_idx, &expected_Xi, failed_name);
            break;
          }
        }

        if (test_failed)
          break;
      }

      if (!test_failed)
        printf(" none found\n");

      printf("\n-------------------------------------------\n\n");
    }
  }

  if (my_rank == 0)
    report_test_results();

  MPI_Finalize();
  return 0;
}
