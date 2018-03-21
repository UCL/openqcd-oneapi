
/*
 * Created: 08-08-2017
 * Modified:
 * Author: Jonas R. Glesaaen (jonas@glesaaen.com)
 */

#define MAIN_PROGRAM

#include "flags.h"
#include "global.h"
#include "mpi.h"
#include "update.h"

#include <devel/testing_utilities/data_type_diffs.c>
#include <devel/testing_utilities/diff_printing.c>
#include <devel/testing_utilities/test_counter.c>

void get_steplist(mdstep_t const *begin, mdstep_t const *end, int *out)
{
  int i = 0;
  for (; begin < end; ++begin) {
    out[i++] = begin->iop;
  }
}

int main(int argc, char *argv[])
{
  size_t nop;
  int my_rank, i, j, n;
  int iact[32], irat[3], imu[4], isp[4];
  int ifr[32], ncr[4];
  int itu, ismear, iunsmear, step_diff;
  int *expected_steplist, *computed_steplist;
  double tau, diff;
  double tau_total[32];
  mdstep_t *s, *sm;
  action_t action_type;
  force_t force_type;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  new_test_module();

  if (my_rank == 0) {
    printf("Test on the mdsteps integrator generation\n");
    printf("-------------------------------------------\n\n");

    printf("%dx%dx%dx%d lattice, ", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid, ", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n\n", L0, L1, L2, L3);
    printf("-------------------------------------------\n\n");
  }

  for (i = 0; i < 3; i++) {
    irat[i] = 0;
  }

  for (i = 0; i < 4; i++) {
    imu[i] = 0;
    isp[i] = 0;
    ncr[i] = 0;
  }

  tau = 1.0;

  /* Register HMC parameters */

  iact[0] = 0;
  iact[1] = 1;
  set_hmc_parms(2, iact, 0, 0, NULL, 2, tau);

  /* Register action 0 */

  action_type = ACG;
  set_action_parms(0, action_type, 0, 0, irat, imu, isp, 0);

  force_type = FRG;
  set_force_parms(0, force_type, 0, 0, irat, imu, isp, ncr);

  /* Register action 1 */

  action_type = ACF_TM1;
  set_action_parms(1, action_type, 0, 0, irat, imu, isp, 0);

  force_type = FRF_TM1;
  set_force_parms(1, force_type, 0, 0, irat, imu, isp, ncr);

  /* Register integrator 0 */

  ifr[0] = 0;
  set_mdint_parms(0, LPFR, 0.0, 4, 1, ifr);

  /* Register integrator 1 */

  ifr[0] = 1;
  set_mdint_parms(1, LPFR, 0.0, 10, 1, ifr);

  /* Get mdsteps */

  set_mdsteps();
  s = mdsteps(&nop, &ismear, &iunsmear, &itu);
  sm = s + nop;

  computed_steplist = malloc(nop * sizeof *computed_steplist);
  get_steplist(s, sm, computed_steplist);

  /* Construct expected steporder */

  expected_steplist = malloc(100 * sizeof *expected_steplist);

  n = 0;
  for (i = 0; i < 10; ++i) {
    expected_steplist[n++] = 0;
    expected_steplist[n++] = 1;
    expected_steplist[n++] = itu;

    for (j = 0; j < (4 - 1); ++j) {
      expected_steplist[n++] = 0;
      expected_steplist[n++] = itu;
    }
  }

  expected_steplist[n++] = 0;
  expected_steplist[n++] = 1;
  expected_steplist[n++] = itu + 1;

  /* Begin tests */

  if (my_rank == 0) {
    register_test(1, "Test operation list");
    print_test_header(1);

    if (n != nop) {
      printf("The expected and computed steplists have different number of "
             "operations\n");
      print_int_array_comparison_tail(computed_steplist, expected_steplist, nop,
                                      n, 3);
      fail_test(1);
    } else {
      step_diff = index_diff_array_int(computed_steplist, expected_steplist, n);
      printf("Checking for discrepancy between computed and expected operation "
             "list...");

      if (step_diff != n) {
        printf(" found\n");
        fail_test(1);
        print_int_array_comparison_mid(computed_steplist, expected_steplist, n,
                                       step_diff, 3);
      } else {
        printf(" none found\n");
      }
    }

    printf("\n-------------------------------------------\n\n");
  }

  for (i = 0; i <= itu; ++i) {
    tau_total[i] = 0.;
  }

  n = 0;
  for (; s < sm; ++s) {
    for (i = 0; i <= itu; ++i) {
      if (i == s->iop) {
        tau_total[i] += s->eps;
        break;
      }
    }
  }

  if (my_rank == 0) {
    register_test(2, "Testing total MD integration time");
    print_test_header(2);

    for (i = 0; i < ismear; ++i) {
      diff = abs_diff_double(tau_total[i], tau);
      printf("Force %2d:   abs|tau(%2d)  - tau| = %.1e (should be 0.0)\n", i, i,
             diff);

      if (diff > 1e-10) {
        fail_test(2);
      }
    }

    diff = abs_diff_double(tau_total[ismear], 0.0);
    printf("Smearing:   abs|tau(sm)  - 0.0| = %.1e (should be 0.0)\n", diff);

    if (diff > 1e-10) {
      fail_test(2);
    }

    diff = abs_diff_double(tau_total[iunsmear], 0.0);
    printf("Unsmearing: abs|tau(us)  - 0.0| = %.1e (should be 0.0)\n", diff);

    if (diff > 1e-10) {
      fail_test(2);
    }

    diff = abs_diff_double(tau_total[iunsmear], 0.0);
    printf("Total:      abs|tau(tot) - tau| = %.1e (should be 0.0)\n", diff);

    if (diff > 1e-10) {
      fail_test(2);
    }

    printf("\n-------------------------------------------\n\n");
  }

  if (my_rank == 0) {
    report_test_results();
  }

  MPI_Finalize();
  return 0;
}
