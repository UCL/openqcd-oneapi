
/*******************************************************************************
 *
 * File test06_unsmearing_1.c
 *
 * Author (2017, 2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Basic check on force unsmearing routine
 *
 *******************************************************************************/

#include "archive.h"
#include "global.h"
#include "lattice.h"
#include "mpi.h"
#include "stout_smearing.h"
#include "uflds.h"

#include <tests/testing_utilities/data_type_diffs.c>
#include <tests/testing_utilities/test_counter.c>
#include <modules/stout_smearing/force_unsmearing.c>

int main(int argc, char *argv[])
{
  int my_rank, ix, mu;
  double t_diff, s_diff, diff_total;
  double theta[3] = {0.0, 0.0, 0.0};

  double inv_tot_links =
      0.25 / ((double)(NPROC) * (double)(L0 * L1) * (double)(L2 * L3));

  su3_alg_dble *force, *force_copy;
  su3_alg_dble const_force;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  new_test_module();

  if (my_rank == 0) {
    printf("Basic check on force unsmearing routine\n");
    printf("-------------------------------------------\n");

    printf("%dx%dx%dx%d lattice,\n", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid,\n", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n", L0, L1, L2, L3);
    printf("-------------------------------------------\n\n");
  }

  set_bc_parms(3, 0., 0., 0., 0., NULL, NULL, theta);
  set_stout_smearing_parms(4, 0., 0.25, 1, 1);

  geometry();

  import_cnfg("configurations/orig_config.conf");

  smear_fields();

  force = (*mdflds()).frc;

  random_alg(4 * VOLUME, force);

  force_copy = amalloc((4 * VOLUME) * sizeof(*force_copy), ALIGN);
  assign_alg2alg(4 * VOLUME, force, force_copy);

  unsmear_force(force);

  t_diff = 0.;
  s_diff = 0.;

  random_alg(1, &const_force);

  for (ix = 0; ix < VOLUME / 2; ++ix) {
    t_diff += abs_diff_su3_alg(force + 8 * ix, force_copy + 8 * ix);
    t_diff += abs_diff_su3_alg(force + 8 * ix + 1, force_copy + 8 * ix + 1);

    for (mu = 1; mu < 4; ++mu) {
      s_diff += abs_diff_su3_alg(force + 8 * ix + 2 * mu,
                                 force_copy + 8 * ix + 2 * mu);
      s_diff += abs_diff_su3_alg(force + 8 * ix + 2 * mu + 1,
                                 force_copy + 8 * ix + 2 * mu + 1);
    }
  }

  diff_total = 0.;
  MPI_Reduce(&t_diff, &diff_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (my_rank == 0) {
    register_test(1, "Diff unsmeared force temporal and spatial links");

    print_test_header(1);

    printf("Total force diff for temporal-links: %+.2e (should be 0.0)\n",
           diff_total);
    printf("Average force diff for temporal-links: %+.2e (should be 0.0)\n",
           diff_total * inv_tot_links);

    fail_test_if(1, (diff_total * inv_tot_links) > 1e-10);
  }

  diff_total = 0.;
  MPI_Reduce(&s_diff, &diff_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (my_rank == 0) {
    printf("Diff force for spatial-links:  %+.2e (should be non-zero)\n",
           diff_total);

    fail_test_if(1, diff_total < 1e-8);

    printf("\n-------------------------------------------\n\n");
  }

  if (my_rank == 0) {
    report_test_results();
  }

  MPI_Finalize();
  return 0;
}
