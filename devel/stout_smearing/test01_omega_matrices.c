
/*
 * Created: 01-06-2017
 * Modified:
 * Author: Jonas R. Glesaaen (jonas@glesaaen.com)
 */

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "su3.h"
#include "lattice.h"
#include "global.h"
#include "stout_smearing.h"

#include <devel/testing_utilities/data_type_diffs.c>
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
  su3_dble *ud;
  double total_diff, rho_t, rho_s;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    printf("Checks of the programs in the module stout_smearing\n");
    printf("-------------------------------------------\n\n");

    printf("%dx%dx%dx%d lattice, ", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid, ", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n\n", L0, L1, L2, L3);
  }

  set_bc_parms(3, 0., 0., 0., 0., NULL, NULL);
  set_stout_smearing_parms(1, 1., 1.);

  geometry();

  ud = udfld();

  for (ix = 0; ix < 4 * VOLUME; ix++) {
    ud[ix].c11.re = 1. * ((ix % 8) / 2 + 1);
    ud[ix].c22.re = 1. * ((ix % 8) / 2 + 1);
    ud[ix].c33.re = 1. * ((ix % 8) / 2 + 1);
  }

  copy_bnd_ud();
  compute_omega_field(ud);

  total_diff = 0.;

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

    total_diff += diff_identity(omega_matrix + ix, expected_value);
  }

  if (my_rank == 0) {
    printf("Check of compute_omega_field() with rho = 1.:\n");
    printf("|omega - expected_value| = %.1e (should be 0.0)\n\n", total_diff);
  }

  rho_t = 0.24;
  rho_s = 0.79;

  set_stout_smearing_parms(1, rho_t, rho_s);
  compute_omega_field(ud);

  total_diff = 0.;

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

    total_diff += diff_identity(omega_matrix + ix, expected_value);
  }

  if (my_rank == 0) {
    printf("Check of compute_omega_field() with rho_t: %.2f, rho_s: %.2f:\n",
           rho_t, rho_s);
    printf("|omega - expected_value| = %.1e (should be 0.0)\n\n", total_diff);
  }

  MPI_Finalize();
  return 0;
}
