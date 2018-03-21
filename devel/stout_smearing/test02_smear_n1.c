
/*
 * Created: 08-06-2017
 * Modified:
 * Author: Jonas R. Glesaaen (jonas@glesaaen.com)
 */

#define MAIN_PROGRAM

#include "archive.h"
#include "flags.h"
#include "global.h"
#include "lattice.h"
#include "mpi.h"
#include "stout_smearing.h"
#include "uflds.h"

#include <devel/testing_utilities/data_type_diffs.c>
#include <devel/testing_utilities/test_counter.c>

#define N0 (NPROC0 * L0)
#define N1 (NPROC1 * L1)
#define N2 (NPROC2 * L2)
#define N3 (NPROC3 * L3)

char test_text[512];

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
    printf("Test smearing against reference configuration \n");
    printf("---------------------------------------------\n");

    printf("%dx%dx%dx%d lattice,\n", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid,\n", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n", L0, L1, L2, L3);
    printf("---------------------------------------------\n\n");
  }

  new_test_module();

  set_bc_parms(3, 0., 0., 0., 0., NULL, NULL, theta);
  set_stout_smearing_parms(1, 0., 0.25, 1, 1);
  stout_params = stout_smearing_parms();

  geometry();
  npl = (double)(6 * N0 * N1) * (double)(N2 * N3);
  nlinks = (double)(2 * N0 * N1) * (double)(N2 * N3);

  import_cnfg("configurations/orig_config.conf");

  smear_fields();
  plaq_temp_o = plaq_wsum_dble(0) / (3. * npl);
  MPI_Reduce(&plaq_temp_o, &plaq_total_o, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  unsmear_fields();

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
        norm_diff_su3(udfld() + ix,
                      smeared_fields()[stout_params.num_smear - 1] + ix) /
        nlinks;
  }

  MPI_Reduce(&config_temp_diff, &config_total_diff, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  if (my_rank == 0) {
    sprintf(
        test_text,
        "Check of smear_fields() with rho_t = %.2f, rho_s = %.2f, n = %d",
        stout_params.rho_temporal, stout_params.rho_spatial,
        stout_params.num_smear);

    register_test(1, test_text);
    print_test_header(1);

    printf("|average plaquette - reference|  = %.1e (should be 0.0)\n",
           fabs(plaq_total_o - plaq_total_r));
    fail_test_if(1, fabs(plaq_total_o - plaq_total_r) > 1e-12);
    printf("accumulated plaq errors          = %.1e (should be 0.0)\n",
           plaq_total_diff);
    fail_test_if(1, plaq_total_diff > 1e-12);
    printf("|smeared links - referece| / vol = %.1e (should be 0.0)\n",
           config_total_diff);
    fail_test_if(1, config_total_diff > 1e-12);

    printf("\n---------------------------------------------\n\n");
  }

  if (my_rank == 0) {
    report_test_results();
  }

  MPI_Finalize();
  return 0;
}
