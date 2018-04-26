
/*******************************************************************************
 *
 * File test01_chs_bcnd_boundaries.c
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Checks of the copy_bnd_ud() function and whether it is correctly applied to
 * the halo if bc = 3
 *
 *******************************************************************************/

#define MAIN_PROGRAM

#include "global.h"
#include "lattice.h"
#include "mpi.h"
#include "su3fcts.h"
#include "uflds.h"

#include <devel/testing_utilities/data_type_diffs.c>
#include <devel/testing_utilities/test_counter.c>

int main(int argc, char *argv[])
{
  int my_rank, ix, num_links_bnd;
  double phi[2], phi_prime[2];
  double theta[3] = {0.0, 0.0, 0.0};
  double diff, total_diff;
  su3_dble *udb, *boundary_copy;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  new_test_module();

  if (my_rank == 0) {
    printf(
        "Test of whether copy_bnd_ud() is correctly applied to the boundary\n");
    printf("-------------------------------------------\n\n");

    printf("%dx%dx%dx%d lattice, ", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid, ", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n", L0, L1, L2, L3);
    printf("\n-------------------------------------------\n\n");
  }

  /* Eventually allow other bc's if I implement this in the set_ud_phase
   * function */
  phi[0] = 0.123;
  phi[1] = -0.534;
  phi_prime[0] = 0.912;
  phi_prime[1] = 0.078;
  theta[0] = 0.35;
  theta[1] = -1.25;
  theta[2] = 0.78;
  set_bc_parms(3, 0.55, 0.78, 0.9012, 1.2034, phi, phi_prime, theta);

  print_bc_parms(2);
  if (my_rank == 0) {
    printf("-------------------------------------------\n\n");
  }

  num_links_bnd = 7 * (BNDRY / 4);
  boundary_copy = malloc(num_links_bnd * sizeof(*boundary_copy));

  geometry();

  random_ud();
  copy_bnd_ud();

  set_ud_phase();

  udb = udfld();
  cm3x3_assign(num_links_bnd, udb + 4 * VOLUME, boundary_copy);

  copy_bnd_ud();

  diff = 0.;
  for (ix = 0; ix < num_links_bnd; ++ix) {
    diff += norm_diff_su3(boundary_copy + ix, udb + 4 * VOLUME + ix);
  }

  total_diff = 0.;
  MPI_Reduce(&diff, &total_diff, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (my_rank == 0) {
    register_test(1, "Sign of boundary links after set_ud_phase()");
    print_test_header(1);

    printf("Total diff: %.2e (should be 0.0)\n", total_diff);
    fail_test_if(1, total_diff > 1e-10);

    printf("\n-------------------------------------------\n\n");
  }

  unset_ud_phase();

  udb = udfld();
  cm3x3_assign(num_links_bnd, udb + 4 * VOLUME, boundary_copy);

  copy_bnd_ud();

  if (my_rank == 0) {
    register_test(2, "Sign of boundary links after unset_ud_phase()");
    print_test_header(2);

    printf("Total diff: %.2e (should be 0.0)\n", total_diff);
    fail_test_if(2, total_diff > 1e-10);

    printf("\n-------------------------------------------\n\n");
  }

  if (my_rank == 0) {
    report_test_results();
  }

  MPI_Finalize();
  return 0;
}
