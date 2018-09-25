
/*******************************************************************************
 *
 * File test02_check_parms.c
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Checks of the write_..._parms check_..._parms functions
 *
 *******************************************************************************/

#define MAIN_PROGRAM

#include "flags.h"
#include "global.h"
#include "mpi.h"

#include <tests/testing_utilities/test_counter.c>

const char file_name[] = "test02_output.dat";

int main(int argc, char *argv[])
{
  int my_rank;
  double theta[3];
  FILE *fdat;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  new_test_module();

  if (my_rank == 0) {
    printf("Checks of the write_..._parms check_..._parms functions\n");
    printf("------------------------------------------\n");

    printf("%dx%dx%dx%d lattice\n", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid\n", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n", L0, L1, L2, L3);
    printf("-------------------------------------------\n\n");
  }

  theta[0] = 0.11;
  theta[1] = 3.15;
  theta[2] = -0.44;

  /* Setup the test by filling a dat file */
  fdat = fopen(file_name, "w");

  set_ani_parms(0, 1.5, 4.3, 1.5, 0.9, 0.7, 1.2, 0.22, 2.11);
  write_ani_parms(fdat);

  set_stout_smearing_parms(2, 0.0, 0.14, 0, 1);
  write_stout_smearing_parms(fdat);

  set_lat_parms(1.5, 5. / 3., 0, NULL, 1.0);
  write_lat_parms(fdat);

  set_bc_parms(3, 0., 0., 0., 0., NULL, NULL, theta);
  write_bc_parms(fdat);

  fclose(fdat);

  /* The tests do not really test for anything, they simply test that openQCD
   * did not emit any errors */

  fdat = fopen(file_name, "r");

  if (my_rank == 0) {
    register_test(1, "Checking ani parameters");
    print_test_header(1);

    check_ani_parms(fdat);

    printf("TEST %d PASSED\n", 1);
    printf("-------------------------------------------\n\n");
  }

  if (my_rank == 0) {
    register_test(2, "Checking stout smearing parameters");
    print_test_header(2);

    check_stout_smearing_parms(fdat);

    printf("TEST %d PASSED\n", 2);
    printf("-------------------------------------------\n\n");
  }

  if (my_rank == 0) {
    register_test(3, "Checking lattice parameters");
    print_test_header(3);

    check_lat_parms(fdat);

    printf("TEST %d PASSED\n", 3);
    printf("-------------------------------------------\n\n");
  }

  if (my_rank == 0) {
    register_test(4, "Checking boundary conditions");
    print_test_header(4);

    check_bc_parms(fdat);

    printf("TEST %d PASSED\n", 4);
    printf("-------------------------------------------\n\n");
  }

  if (my_rank == 0) {
    report_test_results();
  }

  MPI_Finalize();
  return 0;
}
