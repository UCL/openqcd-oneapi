
/*******************************************************************************
 *
 * File test01_omega_matrices.c
 *
 * Author (2017, 2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Checks of the computation of the X matrix with diagonal matrices
 *
 *******************************************************************************/

#define MAIN_PROGRAM

#include "global.h"
#include "lattice.h"
#include "mpi.h"
#include "stout_smearing.h"

#include <tests/testing_utilities/data_type_diffs.c>
#include <tests/testing_utilities/test_counter.c>
#include <modules/stout_smearing/stout_smearing.c>

double diff_identity(su3_dble const *X, double val)
{
  su3_dble identity;
  cm3x3_unity(1, &identity);
  cm3x3_mulr(&val, &identity, &identity);

  return norm_diff_su3(X, &identity);
}

int main(int argc, char *argv[])
{
  int my_rank, ix, mu;
  double expected_value = 0.;
  double theta[3] = {0.0, 0.0, 0.0};
  su3_dble *ud;
  double local_diff, total_diff, rho_t, rho_s;
  double volume_inv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    printf("Checking omega field for diagonal gauge fields\n");
    printf("---------------------------------------------------\n\n");

    printf("%dx%dx%dx%d lattice,\n", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid,\n", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n\n", L0, L1, L2, L3);
    printf("---------------------------------------------------\n\n");
  }

  new_test_module();

  set_bc_parms(3, 0., 0., 0., 0., NULL, NULL, theta);
  set_stout_smearing_parms(1, 1., 1., 1, 1);

  geometry();
  volume_inv = 1. / ((double)(NPROC) * (double)(L0 * L1) * (double)(L2 * L3));

  ud = udfld();

  for (ix = 0; ix < 4 * VOLUME; ix++) {
    ud[ix].c11.re = 1. * ((ix % 8) / 2 + 1);
    ud[ix].c22.re = 1. * ((ix % 8) / 2 + 1);
    ud[ix].c33.re = 1. * ((ix % 8) / 2 + 1);
  }

  copy_bnd_ud();
  compute_omega_field(ud);

  local_diff = 0.;

  for (ix = 0; ix < 4 * VOLUME; ix++) {
    mu = (ix % 8) / 2;

    switch (mu) {
    case 0:
      expected_value = 58.;
      break;
    case 1:
      expected_value = 208.;
      break;
    case 2:
      expected_value = 378.;
      break;
    case 3:
      expected_value = 448.;
    }

    local_diff += diff_identity(omega_matrix + ix, expected_value);
  }

  MPI_Reduce(&local_diff, &total_diff, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  if (my_rank == 0) {
    register_test(1, "Check of compute_omega_field() with rho = 1.0");
    print_test_header(1);

    printf("Total diff: %.1e (should be 0.0)\n", total_diff);
    printf("Average total diff: %.1e (should be 0.0)\n",
           total_diff * volume_inv);

    fail_test_if(1, total_diff * volume_inv > 1e-12);

    printf("\n---------------------------------------------------\n\n");
  }

  rho_t = 0.24;
  rho_s = 0.79;

  reset_stout_smearing();
  set_stout_smearing_parms(1, rho_t, rho_s, 1, 1);
  compute_omega_field(ud);

  local_diff = 0.;

  for (ix = 0; ix < 4 * VOLUME; ix++) {
    mu = (ix % 8) / 2;

    switch (mu) {
    case 0:
      expected_value = 58 * rho_t;
      break;
    case 1:
      expected_value = 8 * rho_t + 200 * rho_s;
      break;
    case 2:
      expected_value = 18 * rho_t + 360 * rho_s;
      break;
    case 3:
      expected_value = 32 * rho_t + 416 * rho_s;
    }

    local_diff += diff_identity(omega_matrix + ix, expected_value);
  }

  MPI_Reduce(&local_diff, &total_diff, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  if (my_rank == 0) {
    register_test(
        2, "Check of compute_omega_field() with rho_t:  0.24, rho_s:  0.79");
    print_test_header(2);

    printf("Total diff: %.1e (should be < 1e-8)\n", total_diff);
    printf("Average total diff: %.1e (should be < 1e-12)\n",
           total_diff * volume_inv);

    fail_test_if(2, total_diff * volume_inv > 1e-12);

    printf("\n---------------------------------------------------\n\n");
  }

  if (my_rank == 0) {
    report_test_results();
  }

  MPI_Finalize();
  return 0;
}
