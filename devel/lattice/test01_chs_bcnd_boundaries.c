
/*
 * Created: 08-06-2017
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
#include "su3fcts.h"
#include "lattice.h"
#include "global.h"
#include "uflds.h"

#include <devel/testing_utilities/data_type_diffs.c>
#include <devel/testing_utilities/test_counter.c>

int main(int argc, char *argv[])
{
  int my_rank, ix, num_links_bnd;
  double theta[3] = {0.0, 0.0, 0.0};
  double diff, total_diff;
  su3_dble *udb, *boundary_copy;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  new_test_module();

  if (my_rank == 0) {
    printf("Checks of the programs in the module lattice\n");
    printf("-------------------------------------------\n\n");

    printf("%dx%dx%dx%d lattice, ", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid, ", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n", L0, L1, L2, L3);
    printf("\n-------------------------------------------\n\n");
  }

  num_links_bnd = 7 * (BNDRY / 4);
  boundary_copy = malloc(num_links_bnd * sizeof(*boundary_copy));

  theta[0] = 0.35;
  theta[1] = -1.25;
  theta[2] = 0.78;

  set_bc_parms(3, 0., 0., 0., 0., NULL, NULL, theta);
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

  if (my_rank == 0)
    report_test_results();

  MPI_Finalize();
  return 0;
}
