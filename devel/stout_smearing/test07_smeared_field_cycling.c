
/*******************************************************************************
 *
 * File test07_smeared_field_cycling.c
 *
 * Author (2017, 2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Checks for how smear_fields() and unsmear_fields() behave
 *
 *******************************************************************************/

#define MAIN_PROGRAM

#include "archive.h"
#include "global.h"
#include "lattice.h"
#include "mpi.h"
#include "stout_smearing.h"
#include "uflds.h"

#include <devel/testing_utilities/data_type_diffs.c>
#include <devel/testing_utilities/test_counter.c>

int main(int argc, char *argv[])
{
  int my_rank, i, local_test, total_test;
  double theta[3] = {0.0, 0.0, 0.0};
  su3_dble **sfields;
  su3_dble *ud;
  su3_dble *ud_adresses[7];

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  new_test_module();

  if (my_rank == 0) {
    printf("Checks for how smear_fields() and unsmear_fields() behave\n");
    printf("-------------------------------------------\n\n");

    printf("%dx%dx%dx%d lattice, ", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid, ", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n", L0, L1, L2, L3);
    printf("\n-------------------------------------------\n\n");
  }

  set_bc_parms(3, 0., 0., 0., 0., NULL, NULL, theta);
  set_stout_smearing_parms(6, 0., 0.25, 1, 1);

  geometry();

  ud_adresses[6] = udfld();

  set_flags(UPDATED_UD);
  set_flags(COPIED_BND_UD);

  sfields = smeared_fields();

  set_flags(SMEARED_UD);
  set_flags(UNSMEARED_UD);

  for (i = 0; i < 6; ++i) {
    ud_adresses[i] = sfields[i];
  }

  smear_fields();

  ud = udfld();
  sfields = smeared_fields();

  if (my_rank == 0) {
    register_test(1, "Unsmeared -> smeared cycle test");
    print_test_header(1);

    printf("udfld()   {%p}, ud_adresses[5] {%p} (should be the same)\n",
           (void *)ud, (void *)ud_adresses[5]);
  }

  local_test = (ud != ud_adresses[5]);
  MPI_Reduce(&local_test, &total_test, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (my_rank == 0) {
    fail_test_if(1, total_test != 0);

    printf("sfield[0] {%p}, ud_adresses[6] {%p} (should be the same)\n",
           (void *)sfields[0], (void *)ud_adresses[6]);
  }

  local_test = (sfields[0] != ud_adresses[6]);
  MPI_Reduce(&local_test, &total_test, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (my_rank == 0) {
    fail_test_if(1, total_test != 0);
  }

  for (i = 0; i < 5; ++i) {
    local_test = (sfields[i + 1] != ud_adresses[i]);
    MPI_Reduce(&local_test, &total_test, 1, MPI_INT, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (my_rank == 0) {
      printf("sfield[%d] {%p}, ud_adresses[%d] {%p} (should be the same)\n",
             i + 1, (void *)sfields[i + 1], i, (void *)ud_adresses[i]);

      fail_test_if(1, total_test != 0);
    }
  }

  if (my_rank == 0) {
    printf("\n-------------------------------------------\n\n");
  }

  unsmear_fields();

  ud = udfld();
  sfields = smeared_fields();

  if (my_rank == 0) {
    register_test(2, "Smeared -> unsmeared cycle test");
    print_test_header(2);
  }

  local_test = (ud != ud_adresses[6]);
  MPI_Reduce(&local_test, &total_test, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (my_rank == 0) {
    printf("udfld()   {%p}, ud_adresses[6] {%p} (should be the same)\n",
           (void *)ud, (void *)ud_adresses[6]);

    fail_test_if(2, total_test != 0);
  }

  for (i = 0; i < 6; ++i) {
    local_test = (sfields[i] != ud_adresses[i]);
    MPI_Reduce(&local_test, &total_test, 1, MPI_INT, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (my_rank == 0) {
      printf("sfield[%d] {%p}, ud_adresses[%d] {%p} (should be the same)\n", i,
             (void *)sfields[i], i, (void *)ud_adresses[i]);

      fail_test_if(2, total_test != 0);
    }
  }

  if (my_rank == 0) {
    printf("\n-------------------------------------------\n\n");
  }

  if (my_rank == 0) {
    report_test_results();
  }

  MPI_Finalize();
  return 0;
}
