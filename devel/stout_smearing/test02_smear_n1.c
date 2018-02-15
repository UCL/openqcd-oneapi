
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
#include "lattice.h"
#include "global.h"
#include "archive.h"
#include "flags.h"
#include "uflds.h"
#include "stout_smearing.h"

#include <devel/testing_utilities/data_type_diffs.c>

#define N0 (NPROC0 * L0)
#define N1 (NPROC1 * L1)
#define N2 (NPROC2 * L2)
#define N3 (NPROC3 * L3)

int main(int argc, char *argv[])
{
  int my_rank, ix;
  double npl, nlinks;
  double theta[3] = {0.0, 0.0, 0.0};
  double plaq_temp_o, plaq_temp_r, plaq_temp_diff, config_temp_diff;
  double plaq_total_o, plaq_total_r, plaq_total_diff, config_total_diff;
  stout_smearing_params_t stout_params;

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

  set_bc_parms(3, 0., 0., 0., 0., NULL, NULL, theta);
  set_stout_smearing_parms(1, 0., 0.25, 1, 1);

  geometry();
  npl = (double)(6 * N0 * N1) * (double)(N2 * N3);
  nlinks = (double)(2 * N0 * N1) * (double)(N2 * N3);

  import_cnfg("configurations/orig_config.conf");

  smear_fields();
  plaq_temp_o = plaq_wsum_dble(0) / (3. * npl);
  MPI_Reduce(&plaq_temp_o, &plaq_total_o, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  /* TODO: Replace with unsmear when that is done */
  cm3x3_assign(4 * VOLUME + 7 * BNDRY / 4, udfld(), smeared_fields()[0]);

  import_cnfg("configurations/smeared_conf_n1.conf");
  plaq_temp_r = plaq_wsum_dble(0) / (3. * npl);
  MPI_Reduce(&plaq_temp_r, &plaq_total_r, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  plaq_temp_diff = fabs(plaq_temp_r - plaq_temp_o);
  MPI_Reduce(&plaq_temp_diff, &plaq_total_diff, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  config_temp_diff = 0.;
  for (ix = 0; ix < (4 * VOLUME); ++ix) {
    config_temp_diff +=
        norm_diff_su3(udfld() + ix, smeared_fields()[0] + ix) / nlinks;
  }

  MPI_Reduce(&config_temp_diff, &config_total_diff, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  stout_params = stout_smearing_parms();

  if (my_rank == 0) {
    printf("Check of smear_fields() with rho_t = %.2f, rho_s = %.2f, n = %d:\n",
           stout_params.rho_temporal, stout_params.rho_spatial,
           stout_params.num_smear);
    printf("|average plaquette - reference|  = %.1e (should be 0.0)\n",
           fabs(plaq_total_o - plaq_total_r));
    printf("accumulated plaq errors          = %.1e (should be 0.0)\n",
           plaq_total_diff);
    printf("|smeared links - referece| / vol = %.1e (should be 0.0)\n\n",
           config_total_diff);
  }

  MPI_Finalize();
  return 0;
}
