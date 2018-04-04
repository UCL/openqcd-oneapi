
/*******************************************************************************
 *
 * File test01_flags_and_states.c
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Tests of the configuration state bitmap
 *
 *******************************************************************************/

#define MAIN_PROGRAM

#include "flags.h"
#include "global.h"
#include "mpi.h"

#include <devel/testing_utilities/data_type_diffs.c>
#include <devel/testing_utilities/test_counter.c>
#include <devel/testing_utilities/verbose_queries.c>

int main(int argc, char *argv[])
{
  int my_rank;
  int is_verbose = 1;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  new_test_module();

  if (my_rank == 0) {
    printf("Tests of the configuration state bitmap\n");
    printf("------------------------------------------\n");

    printf("%dx%dx%dx%d lattice\n", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid\n", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n", L0, L1, L2, L3);
    printf("-------------------------------------------\n\n");
  }

  set_flags(UPDATED_UD);

  if (my_rank == 0) {
    register_test(1, "New clean UD, testing UD flags");
    print_test_header(1);

    test_flag_verbose(1, UD_IS_CLEAN, 1, is_verbose);
    test_flag_verbose(1, UD_IS_SMEARED, 0, is_verbose);
    test_flag_verbose(1, UD_PHASE_SET, 0, is_verbose);
    printf("-------------------------------------------\n\n");
  }

  if (my_rank == 0) {
    register_test(2, "New clean UD, testing buffer flags");
    print_test_header(2);

    test_flag_verbose(2, U_MATCH_UD, 0, is_verbose);
    test_flag_verbose(2, UDBUF_UP2DATE, 0, is_verbose);
    test_flag_verbose(2, BSTAP_UP2DATE, 0, is_verbose);
    test_flag_verbose(2, FTS_UP2DATE, 0, is_verbose);
    test_flag_verbose(2, SW_UP2DATE, 0, is_verbose);
    test_flag_verbose(2, SWD_UP2DATE, 0, is_verbose);
    test_flag_verbose(2, AW_UP2DATE, 0, is_verbose);
    test_flag_verbose(2, AWHAT_UP2DATE, 0, is_verbose);
    test_flag_verbose(2, SMEARED_UD_UP2DATE, 0, is_verbose);
    printf("-------------------------------------------\n\n");
  }

  set_flags(COPIED_BND_UD);

  if (my_rank == 0) {
    register_test(3, "Upadted UD boundaries");
    print_test_header(3);

    test_flag_verbose(3, UDBUF_UP2DATE, 1, is_verbose);
    printf("-------------------------------------------\n\n");
  }

  set_flags(SMEARED_UD);

  if (my_rank == 0) {
    register_test(4, "Smeared UD");
    print_test_header(4);

    test_flag_verbose(4, UD_IS_SMEARED, 1, is_verbose);
    test_flag_verbose(4, UD_PHASE_SET, 0, is_verbose);
    test_flag_verbose(4, SMEARED_UD_UP2DATE, 1, is_verbose);
    test_flag_verbose(4, UDBUF_UP2DATE, 1, is_verbose);
    printf("-------------------------------------------\n\n");
  }

  set_flags(UNSMEARED_UD);

  if (my_rank == 0) {
    register_test(5, "Unsmeared UD");
    print_test_header(5);

    test_flag_verbose(5, UD_IS_SMEARED, 0, is_verbose);
    test_flag_verbose(5, UD_PHASE_SET, 0, is_verbose);
    test_flag_verbose(5, SMEARED_UD_UP2DATE, 1, is_verbose);
    test_flag_verbose(5, UDBUF_UP2DATE, 1, is_verbose);
    printf("-------------------------------------------\n\n");
  }

  set_flags(SET_UD_PHASE);

  if (my_rank == 0) {
    register_test(6, "Set UD phase");
    print_test_header(6);

    test_flag_verbose(6, UD_IS_SMEARED, 0, is_verbose);
    test_flag_verbose(6, UD_PHASE_SET, 1, is_verbose);
    test_flag_verbose(6, SMEARED_UD_UP2DATE, 1, is_verbose);
    test_flag_verbose(6, UDBUF_UP2DATE, 1, is_verbose);
    printf("-------------------------------------------\n\n");
  }

  set_flags(SMEARED_UD);

  if (my_rank == 0) {
    register_test(7, "Smeared UD with phase applied");
    print_test_header(7);

    test_flag_verbose(7, UD_IS_SMEARED, 1, is_verbose);
    test_flag_verbose(7, UD_PHASE_SET, 1, is_verbose);
    test_flag_verbose(7, SMEARED_UD_UP2DATE, 1, is_verbose);
    test_flag_verbose(7, UDBUF_UP2DATE, 1, is_verbose);
    printf("-------------------------------------------\n\n");
  }

  set_flags(UNSET_UD_PHASE);

  if (my_rank == 0) {
    register_test(8, "Unset phase with smearing applied");
    print_test_header(8);

    test_flag_verbose(8, UD_IS_SMEARED, 1, is_verbose);
    test_flag_verbose(8, UD_PHASE_SET, 0, is_verbose);
    test_flag_verbose(8, SMEARED_UD_UP2DATE, 1, is_verbose);
    test_flag_verbose(8, UDBUF_UP2DATE, 1, is_verbose);
    printf("-------------------------------------------\n\n");
  }

  if (my_rank == 0) {
    report_test_results();
  }

  MPI_Finalize();
  return 0;
}
