
/*******************************************************************************
 *
 * File test01_polyakov_loop.c
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * The externally accessible functions are
 *
 *
 * Notes:
 *
 *
 *******************************************************************************/

#define OPENQCD_INTERNAL

#if !defined(STATIC_SIZES)
#error : This test cannot be compiled with dynamic lattice sizes
#endif

#include "global.h"
#include "mpi.h"
#include "lattice.h"
#include "su3fcts.h"
#include "uflds.h"

#include <tests/testing_utilities/data_type_diffs.c>
#include <tests/testing_utilities/test_counter.c>

int main(int argc, char *argv[])
{
  int my_rank;
  double theta[3] = {0.24, -1.1, 0.97};
  double ploop;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    printf("Test polyakov loop\n");
    printf("---------------------------------------------\n\n");

    printf("%dx%dx%dx%d lattice,\n", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid,\n", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n\n", L0, L1, L2, L3);
  }

  set_bc_parms(3, 0., 0., 0., 0., NULL, NULL, theta);
  print_bc_parms(2);

  set_no_ani_parms();

  if (my_rank == 0) {
    printf("---------------------------------------------\n\n");
  }

  geometry();
  new_test_module();

  cm3x3_unity(4*VOLUME, udfld());

  { /* Test 1 */

    if (my_rank == 0) {
      register_test(1, "Polyakov loop for unit gauge field");
      print_test_header(1);
    }

    ploop = polyakov_loop();

    if (my_rank == 0) {
      printf("Polyakov loop = %lf (should be 1.0)\n", ploop);
      fail_test_if(1, fabs(ploop - 1.0) > 1e-12);
      printf("\n-------------------------------------------\n\n");
    }
  }

  { /* Test 2 */

    if (my_rank == 0) {
      register_test(2, "Polyakov loop with phase set");
      print_test_header(2);
    }

    set_ud_phase();
    ploop = polyakov_loop();

    if (my_rank == 0) {
      printf("Polyakov loop = %lf (should be 1.0)\n", ploop);
      fail_test_if(2, fabs(ploop - 1.0) > 1e-12);
      printf("\n-------------------------------------------\n\n");
    }
  }

  if (my_rank == 0) {
    report_test_results();
  }

  MPI_Finalize();
  return 0;
}
