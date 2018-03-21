
/*
 * Created: 08-08-2017
 * Modified:
 * Author: Jonas R. Glesaaen (jonas@glesaaen.com)
 */

#define MAIN_PROGRAM

#include "global.h"
#include "mpi.h"
#include "update.h"

#include <devel/testing_utilities/data_type_diffs.c>
#include <devel/testing_utilities/diff_printing.c>
#include <devel/testing_utilities/test_counter.c>

#include <modules/update/mdsteps.c>

#define MAXSTEPS 1000

void get_steplist(mdstep_t const *begin, mdstep_t const *end, int *out)
{
  int i = 0;
  for (; begin < end; ++begin)
    out[i++] = begin->iop;
}

void get_epslist(mdstep_t const *begin, mdstep_t const *end, double *out)
{
  int i = 0;
  for (; begin < end; ++begin)
    out[i++] = begin->eps;
}

void set_steplists2zero(int n, int *steps, double *eps)
{
  int i;

  for (i = 0; i < n; i++) {
    steps[i] = iend;
    eps[i] = 0.0;
  }
}

void test_index_eps_lists(int test_id, int steps[], int steps_expec[], int end,
                          double eps[], double eps_expec[])
{
  int index_diff;

  index_diff = index_diff_array_int(steps, steps_expec, end);

  if (index_diff == end) {
    printf("List of steps are what we expect\n");
  } else {
    printf("List of steps are not what we expect\n");
    print_int_array_comparison_mid(steps, steps_expec, end + 3, index_diff, 3);
    fail_test(test_id);
  }

  index_diff = index_diff_array_double(eps, eps_expec, end, 1e-12);

  if (index_diff == end) {
    printf("List of epsilons are what we expect\n");
  } else {
    printf("List of epsilons are not what we expect\n");
    print_double_array_comparison_mid(eps, eps_expec, end + 3, index_diff, 3);
    fail_test(test_id);
  }
}

int main(int argc, char *argv[])
{
  int my_rank, i;
  mdstep_t s1[MAXSTEPS], s2[MAXSTEPS];
  int nfrc, itu, ismear, iunsmear;
  int begin, end;
  double c;
  int irat[3], imu[4], isp[4], ncr[4];

  int steps1[MAXSTEPS], steps2[MAXSTEPS], steps_expec[MAXSTEPS];
  double eps1[MAXSTEPS], eps2[MAXSTEPS], eps_expec[MAXSTEPS];

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  new_test_module();

  if (my_rank == 0) {
    printf("Test on the mdsteps smearing intrinsics\n");
    printf("-------------------------------------------\n\n");

    printf("%dx%dx%dx%d lattice, ", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid, ", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n", L0, L1, L2, L3);
    printf("\n-------------------------------------------\n\n");
  }

  /* Setup variables and workspaces */

  nfrc = 5;
  ismear = nfrc + 1;
  iunsmear = ismear + 1;
  itu = iunsmear + 1;
  iend = itu + 1;

  for (i = 0; i < 3; i++)
    irat[i] = 0;

  for (i = 0; i < 4; i++) {
    imu[i] = 0;
    isp[i] = 0;
    ncr[i] = 0;
  }

  set_force_parms(0, FRF_TM1, 0, 0, irat, imu, isp, ncr);
  set_force_parms(1, FRF_TM1, 0, 0, irat, imu, isp, ncr);
  set_force_parms(2, FRG, 0, 0, irat, imu, isp, ncr);
  set_force_parms(3, FRF_TM1, 0, 0, irat, imu, isp, ncr);
  set_force_parms(4, FRF_TM1, 0, 0, irat, imu, isp, ncr);
  set_force_parms(5, FRF_TM1, 0, 0, irat, imu, isp, ncr);

  nsmx = MAXSTEPS;
  alloc_mds();

  set_steps2zero(nsmx, s1);
  set_steps2zero(nsmx, s2);

  set_steplists2zero(nsmx, steps1, eps1);
  set_steplists2zero(nsmx, steps2, eps2);
  set_steplists2zero(nsmx, steps_expec, eps_expec);

  /* Begin tests */

  { /* Test 1 */
    s1[0].iop = itu;

    begin = smear_block_begin(s1);

    if (my_rank == 0) {
      register_test(1, "Smearing block of empty block");
      print_test_header(1);

      if (begin == 0) {
        printf("No smearing block found\n");
      } else {
        printf("Smearing block starting at %d found\n", begin);
        fail_test(1);
      }

      printf("\n-------------------------------------------\n\n");
    }
  }

  { /* Test 2 */
    set_steps2zero(nsmx, s1);

    s1[0].iop = ismear;
    s1[1].iop = 0;
    s1[2].iop = 1;
    s1[3].iop = 2;
    s1[4].iop = iunsmear;
    s1[5].iop = 3;
    s1[6].iop = 4;
    s1[7].iop = itu;

    begin = smear_block_begin(s1);
    end = begin + smear_block_end(s1 + begin);

    if (my_rank == 0) {
      register_test(2, "Smearing block at beginning of the update block");
      print_test_header(2);

      printf("block at (%d, %d), should be (1, 4)\n", begin, end);
      fail_test_if(2, (begin != 1) || (end != 4));

      printf("ids at (begin-1, end) = (%d, %d), should be (%d, %d)\n",
             s1[begin - 1].iop, s1[end].iop, ismear, iunsmear);
      fail_test_if(2,
                   (s1[begin - 1].iop != ismear) || (s1[end].iop != iunsmear));

      printf("\n-------------------------------------------\n\n");
    }
  }

  { /* Test 3 */
    set_steps2zero(nsmx, s1);

    s1[0].iop = 0;
    s1[1].iop = 1;
    s1[2].iop = ismear;
    s1[3].iop = 3;
    s1[4].iop = 4;
    s1[5].iop = iunsmear;
    s1[6].iop = 2;
    s1[7].iop = itu;

    begin = smear_block_begin(s1);
    end = begin + smear_block_end(s1 + begin);

    if (my_rank == 0) {
      register_test(3, "Smearing block in the middle of the update block");
      print_test_header(3);

      printf("block at (%d, %d), should be (3, 5)\n", begin, end);
      fail_test_if(3, (begin != 3) || (end != 5));

      printf("ids at (begin-1, end) = (%d, %d), should be (%d, %d)\n",
             s1[begin - 1].iop, s1[end].iop, ismear, iunsmear);
      fail_test_if(3,
                   (s1[begin - 1].iop != ismear) || (s1[end].iop != iunsmear));

      printf("\n-------------------------------------------\n\n");
    }
  }

  { /* Test 4 */
    set_steps2zero(nsmx, s1);

    s1[0].iop = ismear;
    s1[1].iop = iunsmear;
    s1[2].iop = 0;
    s1[3].iop = 1;
    s1[4].iop = 2;
    s1[5].iop = 3;
    s1[6].iop = 4;
    s1[7].iop = itu;

    begin = smear_block_begin(s1);
    end = begin + smear_block_end(s1 + begin);

    if (my_rank == 0) {
      register_test(4, "Empty smearing block at beginning of the update block");
      print_test_header(4);

      printf("block at (%d, %d), should be (1, 1)\n", begin, end);
      fail_test_if(4, (begin != 1) || (end != 1));

      printf("ids at (begin-1, end) = (%d, %d), should be (%d, %d)\n",
             s1[begin - 1].iop, s1[end].iop, ismear, iunsmear);
      fail_test_if(4,
                   (s1[begin - 1].iop != ismear) || (s1[end].iop != iunsmear));

      printf("\n-------------------------------------------\n\n");
    }
  }

  { /*Test 5 */
    set_steps2zero(nsmx, s1);

    s1[0].iop = 0;
    s1[1].iop = 1;
    s1[2].iop = ismear;
    s1[3].iop = iunsmear;
    s1[4].iop = 2;
    s1[5].iop = 3;
    s1[6].iop = 4;
    s1[7].iop = itu;

    begin = smear_block_begin(s1);
    end = begin + smear_block_end(s1 + begin);

    if (my_rank == 0) {
      register_test(5,
                    "Empty smearing block in the middle of the update block");
      print_test_header(5);

      printf("block at (%d, %d), should be (3, 3)\n", begin, end);
      fail_test_if(5, (begin != 3) || (end != 3));

      printf("ids at (begin-1, end) = (%d, %d), should be (%d, %d)\n",
             s1[begin - 1].iop, s1[end].iop, ismear, iunsmear);
      fail_test_if(5,
                   (s1[begin - 1].iop != ismear) || (s1[end].iop != iunsmear));

      printf("\n-------------------------------------------\n\n");
    }
  }

  { /* Test 6 */
    set_steps2zero(nsmx, s1);

    s1[0].iop = ismear;
    s1[1].iop = 0;
    s1[2].iop = 1;
    s1[3].iop = 3;
    s1[4].iop = iunsmear;
    s1[5].iop = 4;
    s1[6].iop = 2;
    s1[7].iop = itu;

    if (my_rank == 0) {
      register_test(6, "Test of is is_smeared_step for [{0, 1, 3}, 4, 2]");
      print_test_header(6);

      printf("is_smeared_step([1, 2, 3, 5, 6]) = [%d, %d, %d, %d, %d] "
             "(should be [1, 1, 1, 0, 0])\n",
             is_smeared_step(1, s1), is_smeared_step(2, s1),
             is_smeared_step(3, s1), is_smeared_step(5, s1),
             is_smeared_step(6, s1));

      fail_test_if(6, (is_smeared_step(1, s1) != 1) ||
                          (is_smeared_step(2, s1) != 1) ||
                          (is_smeared_step(3, s1) != 1) ||
                          (is_smeared_step(5, s1) != 0) ||
                          (is_smeared_step(6, s1) != 0));

      printf("\n-------------------------------------------\n\n");
    }
  }

  { /* Test 7 */
    set_steps2zero(nsmx, s1);

    s1[0].iop = 0;
    s1[1].iop = 1;
    s1[2].iop = ismear;
    s1[3].iop = 3;
    s1[4].iop = 4;
    s1[5].iop = iunsmear;
    s1[6].iop = 2;
    s1[7].iop = itu;

    if (my_rank == 0) {
      register_test(7, "Test of is is_smeared_step for [0, 1, {3, 4}, 2]");
      print_test_header(7);

      printf("is_smeared_step([0, 1, 3, 4, 6]) = [%d, %d, %d, %d, %d] "
             "(should be [0, 0, 1, 1, 0])\n",
             is_smeared_step(0, s1), is_smeared_step(1, s1),
             is_smeared_step(3, s1), is_smeared_step(4, s1),
             is_smeared_step(6, s1));

      fail_test_if(7, (is_smeared_step(0, s1) != 0) ||
                          (is_smeared_step(1, s1) != 0) ||
                          (is_smeared_step(3, s1) != 1) ||
                          (is_smeared_step(4, s1) != 1) ||
                          (is_smeared_step(6, s1) != 0));

      printf("\n-------------------------------------------\n\n");
    }
  }

  { /* Test 8 */
    set_steps2zero(nsmx, s1);
    set_steps2zero(nsmx, s2);
    set_steplists2zero(nsmx, steps_expec, eps_expec);

    s1[0].iop = 0;
    s1[0].eps = 0.11;
    s1[1].iop = 2;
    s1[1].eps = 0.22;
    s1[2].iop = itu;

    s2[0].iop = 1;
    s2[0].eps = 0.33;

    c = 0.75;

    add_normal_step(c, s2, s1);
    end = nfrc_steps(s1) + 1;

    get_steplist(s1, s1 + end, steps1);
    get_epslist(s1, s1 + end, eps1);

    steps_expec[0] = 0;
    steps_expec[1] = 2;
    steps_expec[2] = 1;
    steps_expec[3] = itu;

    eps_expec[0] = 0.11;
    eps_expec[1] = 0.22;
    eps_expec[2] = 0.33 * c;

    if (my_rank == 0) {
      register_test(8, "Test add_normal_step for smearing free block");
      print_test_header(8);

      test_index_eps_lists(8, steps1, steps_expec, end, eps1, eps_expec);

      printf("\n-------------------------------------------\n\n");
    }
  }

  { /* Test 9 */
    set_steps2zero(nsmx, s1);
    set_steps2zero(nsmx, s2);
    set_steplists2zero(nsmx, steps_expec, eps_expec);

    s1[0].iop = 0;
    s1[0].eps = 0.11;
    s1[1].iop = 2;
    s1[1].eps = 0.22;
    s1[2].iop = 1;
    s1[2].eps = 0.33;
    s1[3].iop = itu;

    s2[0].iop = 2;
    s2[0].eps = 0.44;

    c = 0.55;

    add_normal_step(c, s2, s1);
    end = nfrc_steps(s1) + 1;

    get_steplist(s1, s1 + end, steps1);
    get_epslist(s1, s1 + end, eps1);

    steps_expec[0] = 0;
    steps_expec[1] = 2;
    steps_expec[2] = 1;
    steps_expec[3] = itu;

    eps_expec[0] = 0.11;
    eps_expec[1] = 0.22 + c * 0.44;
    eps_expec[2] = 0.33;

    if (my_rank == 0) {
      register_test(
          9,
          "Test add_normal_step with exisiting step for smearing free block");
      print_test_header(9);

      test_index_eps_lists(9, steps1, steps_expec, end, eps1, eps_expec);

      printf("\n-------------------------------------------\n\n");
    }
  }

  { /* Test 10 */
    set_steps2zero(nsmx, s1);
    set_steps2zero(nsmx, s2);
    set_steplists2zero(nsmx, steps_expec, eps_expec);

    s1[0].iop = 0;
    s1[0].eps = 0.11;
    s1[1].iop = 2;
    s1[1].eps = 0.22;
    s1[2].iop = 1;
    s1[2].eps = 0.33;
    s1[3].iop = itu;

    s2[0].iop = 2;
    s2[0].eps = 0.44;

    c = 0.55;

    add_smeared_step(c, s2, s1);
    end = nfrc_steps(s1) + 1;

    get_steplist(s1, s1 + end, steps1);
    get_epslist(s1, s1 + end, eps1);

    steps_expec[0] = ismear;
    steps_expec[1] = 2;
    steps_expec[2] = iunsmear;
    steps_expec[3] = 0;
    steps_expec[4] = 2;
    steps_expec[5] = 1;
    steps_expec[6] = itu;

    eps_expec[1] = 0.44 * c;
    eps_expec[3] = 0.11;
    eps_expec[4] = 0.22;
    eps_expec[5] = 0.33;

    if (my_rank == 0) {
      register_test(10, "Test add_smearing_step to smearing free block");
      print_test_header(10);

      test_index_eps_lists(10, steps1, steps_expec, end, eps1, eps_expec);

      printf("\n-------------------------------------------\n\n");
    }
  }

  { /* Test 11 */
    set_steps2zero(nsmx, s1);
    set_steps2zero(nsmx, s2);
    set_steplists2zero(nsmx, steps_expec, eps_expec);

    s1[0].iop = ismear;
    s1[1].iop = 0;
    s1[1].eps = 0.11;
    s1[2].iop = 2;
    s1[2].eps = 0.22;
    s1[3].iop = iunsmear;
    s1[4].iop = 1;
    s1[4].eps = 0.33;
    s1[5].iop = itu;

    s2[0].iop = 3;
    s2[0].eps = 0.44;

    c = 0.55;

    add_normal_step(c, s2, s1);
    end = nfrc_steps(s1) + 1;

    get_steplist(s1, s1 + end, steps1);
    get_epslist(s1, s1 + end, eps1);

    steps_expec[0] = ismear;
    steps_expec[1] = 0;
    steps_expec[2] = 2;
    steps_expec[3] = iunsmear;
    steps_expec[4] = 1;
    steps_expec[5] = 3;
    steps_expec[6] = itu;

    eps_expec[1] = 0.11;
    eps_expec[2] = 0.22;
    eps_expec[4] = 0.33;
    eps_expec[5] = 0.44 * c;

    if (my_rank == 0) {
      register_test(11, "Test add_normal_step (new) to smeared block");
      print_test_header(11);

      test_index_eps_lists(11, steps1, steps_expec, end, eps1, eps_expec);

      printf("\n-------------------------------------------\n\n");
    }
  }

  { /* Test 12 */
    set_steps2zero(nsmx, s1);
    set_steps2zero(nsmx, s2);
    set_steplists2zero(nsmx, steps_expec, eps_expec);

    s1[0].iop = ismear;
    s1[1].iop = 0;
    s1[1].eps = 0.11;
    s1[2].iop = 2;
    s1[2].eps = 0.22;
    s1[3].iop = iunsmear;
    s1[4].iop = 1;
    s1[4].eps = 0.33;
    s1[5].iop = itu;

    s2[0].iop = 1;
    s2[0].eps = 0.44;

    c = 0.55;

    add_normal_step(c, s2, s1);
    end = nfrc_steps(s1) + 1;

    get_steplist(s1, s1 + end, steps1);
    get_epslist(s1, s1 + end, eps1);

    steps_expec[0] = ismear;
    steps_expec[1] = 0;
    steps_expec[2] = 2;
    steps_expec[3] = iunsmear;
    steps_expec[4] = 1;
    steps_expec[5] = itu;

    eps_expec[1] = 0.11;
    eps_expec[2] = 0.22;
    eps_expec[4] = 0.33 + 0.44 * c;

    if (my_rank == 0) {
      register_test(12, "Test add_normal_step (old) to smeared block");
      print_test_header(12);

      test_index_eps_lists(12, steps1, steps_expec, end, eps1, eps_expec);

      printf("\n-------------------------------------------\n\n");
    }
  }

  { /* Test 13 */
    set_steps2zero(nsmx, s1);
    set_steps2zero(nsmx, s2);
    set_steplists2zero(nsmx, steps_expec, eps_expec);

    s1[0].iop = ismear;
    s1[1].iop = 0;
    s1[1].eps = 0.11;
    s1[2].iop = 2;
    s1[2].eps = 0.22;
    s1[3].iop = iunsmear;
    s1[4].iop = 1;
    s1[4].eps = 0.33;
    s1[5].iop = itu;

    s2[0].iop = 1;
    s2[0].eps = 0.44;

    c = 0.55;

    add_smeared_step(c, s2, s1);
    end = nfrc_steps(s1) + 1;

    get_steplist(s1, s1 + end, steps1);
    get_epslist(s1, s1 + end, eps1);

    steps_expec[0] = ismear;
    steps_expec[1] = 0;
    steps_expec[2] = 2;
    steps_expec[3] = 1;
    steps_expec[4] = iunsmear;
    steps_expec[5] = 1;
    steps_expec[6] = itu;

    eps_expec[1] = 0.11;
    eps_expec[2] = 0.22;
    eps_expec[3] = 0.44 * c;
    eps_expec[5] = 0.33;

    if (my_rank == 0) {
      register_test(13, "Test add_smeared_step (new) to smeared block");
      print_test_header(13);

      test_index_eps_lists(13, steps1, steps_expec, end, eps1, eps_expec);

      printf("\n-------------------------------------------\n\n");
    }
  }

  { /* Test 14 */
    set_steps2zero(nsmx, s1);
    set_steps2zero(nsmx, s2);
    set_steplists2zero(nsmx, steps_expec, eps_expec);

    s1[0].iop = ismear;
    s1[1].iop = 0;
    s1[1].eps = 0.11;
    s1[2].iop = 2;
    s1[2].eps = 0.22;
    s1[3].iop = iunsmear;
    s1[4].iop = 1;
    s1[4].eps = 0.33;
    s1[5].iop = itu;

    s2[0].iop = 2;
    s2[0].eps = 0.44;

    c = 0.55;

    add_smeared_step(c, s2, s1);
    end = nfrc_steps(s1) + 1;

    get_steplist(s1, s1 + end, steps1);
    get_epslist(s1, s1 + end, eps1);

    steps_expec[0] = ismear;
    steps_expec[1] = 0;
    steps_expec[2] = 2;
    steps_expec[3] = iunsmear;
    steps_expec[4] = 1;
    steps_expec[5] = itu;

    eps_expec[1] = 0.11;
    eps_expec[2] = 0.22 + 0.44 * c;
    eps_expec[4] = 0.33;

    if (my_rank == 0) {
      register_test(14, "Test add_smeared_step (old) to smeared block");
      print_test_header(14);

      test_index_eps_lists(14, steps1, steps_expec, end, eps1, eps_expec);

      printf("\n-------------------------------------------\n\n");
    }
  }

  { /* Test 15 */
    set_steps2zero(nsmx, s1);
    set_steps2zero(nsmx, s2);
    set_steplists2zero(nsmx, steps_expec, eps_expec);

    s1[0].iop = 0;
    s1[0].eps = 0.11;
    s1[1].iop = 2;
    s1[1].eps = 0.22;
    s1[2].iop = itu;

    s2[0].iop = 1;
    s2[0].eps = 0.33;
    s2[1].iop = 2;
    s2[1].eps = 0.44;
    s2[2].iop = 3;
    s2[2].eps = 0.66;
    s2[3].iop = itu;

    c = 0.77;

    add_steps(nfrc_steps(s2), c, s2, s1);
    end = nfrc_steps(s1) + 1;

    get_steplist(s1, s1 + end, steps1);
    get_epslist(s1, s1 + end, eps1);

    steps_expec[0] = 0;
    steps_expec[1] = 2;
    steps_expec[2] = 1;
    steps_expec[3] = 3;
    steps_expec[4] = itu;

    eps_expec[0] = 0.11;
    eps_expec[1] = 0.22 + 0.44 * c;
    eps_expec[2] = 0.33 * c;
    eps_expec[3] = 0.66 * c;

    if (my_rank == 0) {
      register_test(15, "Adding two unsmeared blocks");
      print_test_header(15);

      test_index_eps_lists(15, steps1, steps_expec, end, eps1, eps_expec);

      printf("\n-------------------------------------------\n\n");
    }
  }

  { /* Test 16 */
    set_steps2zero(nsmx, s1);
    set_steps2zero(nsmx, s2);
    set_steplists2zero(nsmx, steps_expec, eps_expec);

    s1[0].iop = 0;
    s1[0].eps = 0.11;
    s1[1].iop = 2;
    s1[1].eps = 0.22;
    s1[2].iop = itu;

    s2[0].iop = ismear;
    s2[1].iop = 1;
    s2[1].eps = 0.33;
    s2[2].iop = 2;
    s2[2].eps = 0.44;
    s2[3].iop = iunsmear;
    s2[4].iop = 0;
    s2[4].eps = 0.66;
    s2[5].iop = 4;
    s2[5].eps = 0.88;
    s2[6].iop = itu;

    c = 0.77;

    add_steps(nfrc_steps(s2), c, s2, s1);
    end = nfrc_steps(s1) + 1;

    get_steplist(s1, s1 + end, steps1);
    get_epslist(s1, s1 + end, eps1);

    steps_expec[0] = ismear;
    steps_expec[1] = 1;
    steps_expec[2] = 2;
    steps_expec[3] = iunsmear;
    steps_expec[4] = 0;
    steps_expec[5] = 2;
    steps_expec[6] = 4;
    steps_expec[7] = itu;

    eps_expec[1] = 0.33 * c;
    eps_expec[2] = 0.44 * c;
    eps_expec[4] = 0.11 + c * 0.66;
    eps_expec[5] = 0.22;
    eps_expec[6] = 0.88 * c;

    if (my_rank == 0) {
      register_test(16, "Adding smeared block to unsmeared block");
      print_test_header(16);

      test_index_eps_lists(16, steps1, steps_expec, end, eps1, eps_expec);

      printf("\n-------------------------------------------\n\n");
    }
  }

  { /* Test 17 */
    set_steps2zero(nsmx, s1);
    set_steps2zero(nsmx, s2);
    set_steplists2zero(nsmx, steps_expec, eps_expec);

    s1[0].iop = ismear;
    s1[1].iop = 0;
    s1[1].eps = 0.11;
    s1[2].iop = 2;
    s1[2].eps = 0.22;
    s1[3].iop = iunsmear;
    s1[4].iop = 3;
    s1[4].eps = 0.33;
    s1[5].iop = itu;

    s2[0].iop = 2;
    s2[0].eps = 0.44;
    s2[1].iop = 4;
    s2[1].eps = 0.55;
    s2[2].iop = 3;
    s2[2].eps = 0.66;

    c = 0.77;

    add_steps(nfrc_steps(s2), c, s2, s1);
    end = nfrc_steps(s1) + 1;

    get_steplist(s1, s1 + end, steps1);
    get_epslist(s1, s1 + end, eps1);

    steps_expec[0] = ismear;
    steps_expec[1] = 0;
    steps_expec[2] = 2;
    steps_expec[3] = iunsmear;
    steps_expec[4] = 3;
    steps_expec[5] = 2;
    steps_expec[6] = 4;
    steps_expec[7] = itu;

    eps_expec[1] = 0.11;
    eps_expec[2] = 0.22;
    eps_expec[4] = 0.33 + c * 0.66;
    eps_expec[5] = 0.44 * c;
    eps_expec[6] = 0.55 * c;

    if (my_rank == 0) {
      register_test(17, "Adding unsmeared block to smeared block");
      print_test_header(17);

      test_index_eps_lists(17, steps1, steps_expec, end, eps1, eps_expec);

      printf("\n-------------------------------------------\n\n");
    }
  }

  { /* Test 18 */
    set_steps2zero(nsmx, s1);
    set_steps2zero(nsmx, s2);
    set_steplists2zero(nsmx, steps_expec, eps_expec);

    s1[0].iop = ismear;
    s1[1].iop = 0;
    s1[1].eps = 0.11;
    s1[2].iop = 2;
    s1[2].eps = 0.22;
    s1[3].iop = iunsmear;
    s1[4].iop = 0;
    s1[4].eps = 0.111;
    s1[5].iop = 3;
    s1[5].eps = 0.33;
    s1[6].iop = itu;

    s2[0].iop = ismear;
    s2[1].iop = 2;
    s2[1].eps = 0.44;
    s2[2].iop = 3;
    s2[2].eps = 0.55;
    s2[3].iop = 1;
    s2[3].eps = 0.77;
    s2[4].iop = iunsmear;
    s2[5].iop = 2;
    s2[5].eps = 0.222;
    s2[6].iop = 3;
    s2[6].eps = 0.333;
    s2[7].iop = 4;
    s2[7].eps = 0.444;
    s2[8].iop = itu;

    c = 0.666;

    add_steps(nfrc_steps(s2), c, s2, s1);
    end = nfrc_steps(s1) + 1;

    get_steplist(s1, s1 + end, steps1);
    get_epslist(s1, s1 + end, eps1);

    steps_expec[0] = ismear;
    steps_expec[1] = 0;
    steps_expec[2] = 2;
    steps_expec[3] = 3;
    steps_expec[4] = 1;
    steps_expec[5] = iunsmear;
    steps_expec[6] = 0;
    steps_expec[7] = 3;
    steps_expec[8] = 2;
    steps_expec[9] = 4;
    steps_expec[10] = itu;

    eps_expec[1] = 0.11;
    eps_expec[2] = 0.22 + 0.44 * c;
    eps_expec[3] = 0.55 * c;
    eps_expec[4] = 0.77 * c;
    eps_expec[6] = 0.111;
    eps_expec[7] = 0.33 + 0.333 * c;
    eps_expec[8] = 0.222 * c;
    eps_expec[9] = 0.444 * c;

    if (my_rank == 0) {
      register_test(18, "Adding smeared block to smeared block");
      print_test_header(18);

      test_index_eps_lists(18, steps1, steps_expec, end, eps1, eps_expec);

      printf("\n-------------------------------------------\n\n");
    }
  }

  { /* Test 19 */
    set_steps2zero(nsmx, mds);
    set_steplists2zero(nsmx, steps_expec, eps_expec);

    mds[0].iop = ismear;
    mds[1].iop = 0;
    mds[2].iop = 2;
    mds[3].iop = 3;
    mds[4].iop = 1;
    mds[5].iop = iunsmear;
    mds[6].iop = 0;
    mds[7].iop = 3;
    mds[8].iop = 2;
    mds[9].iop = 4;
    mds[10].iop = itu;

    mds[1].eps = 0.11;
    mds[2].eps = 0.22 + 0.44 * c;
    mds[3].eps = 0.55 * c;
    mds[4].eps = 0.77 * c;
    mds[6].eps = 0.111;
    mds[7].eps = 0.33 + 0.333 * c;
    mds[8].eps = 0.222 * c;
    mds[9].eps = 0.444 * c;

    end = nfrc_steps(mds) + 1;

    sort_forces();

    get_steplist(mds, mds + end, steps1);
    get_epslist(mds, mds + end, eps1);

    steps_expec[0] = ismear;
    steps_expec[1] = 2;
    steps_expec[2] = 0;
    steps_expec[3] = 1;
    steps_expec[4] = 3;
    steps_expec[5] = iunsmear;
    steps_expec[6] = 2;
    steps_expec[7] = 0;
    steps_expec[8] = 3;
    steps_expec[9] = 4;
    steps_expec[10] = itu;

    eps_expec[1] = 0.22 + 0.44 * c;
    eps_expec[2] = 0.11;
    eps_expec[3] = 0.77 * c;
    eps_expec[4] = 0.55 * c;
    eps_expec[6] = 0.222 * c;
    eps_expec[7] = 0.111;
    eps_expec[8] = 0.33 + 0.333 * c;
    eps_expec[9] = 0.444 * c;

    if (my_rank == 0) {
      register_test(19, "Sorting an integration block");
      print_test_header(19);

      test_index_eps_lists(19, steps1, steps_expec, end, eps1, eps_expec);

      printf("\n-------------------------------------------\n\n");
    }
  }

  if (my_rank == 0)
    report_test_results();

  MPI_Finalize();
  return 0;
}
