
/*******************************************************************************
 *
 * File test01_partial_communications.c
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Check of the partial communication routines
 *
 *******************************************************************************/

#define MAIN_PROGRAM

#include "field_com.h"
#include "global.h"
#include "lattice.h"
#include "mpi.h"
#include "random.h"
#include "su3fcts.h"
#include "uflds.h"

#include <tests/testing_utilities/data_type_diffs.c>
#include <tests/testing_utilities/test_counter.c>

static size_t one_offsets[4];
static size_t two_offsets[4];
static size_t num_type_one[4];
static size_t num_type_two[4];
static int init = 0;

static su3_dble *boundary_buffer(void)
{
  size_t nlinks;
  static su3_dble *buffer = NULL;

  if (buffer == NULL) {
    nlinks = 7 * BNDRY / 4;
    buffer = amalloc(nlinks * sizeof(*buffer), ALIGN);
  }

  return buffer;
}

static void reset_dirs(int *dirs)
{
  int ix;
  for (ix = 0; ix < 4; ++ix) {
    dirs[ix] = 1;
  }
}

static void set_dirs_bitmap(int *dirs, int bmap)
{
  int ix;
  for (ix = 0; ix < 4; ++ix) {
    dirs[ix] = (bmap >> ix) & 1;
  }
}

static void print_dirs(int const *dirs)
{
  printf("[%d %d %d %d] ", dirs[0], dirs[1], dirs[2], dirs[3]);
}

static void init_arrays(void)
{
  if (init == 1) {
    return;
  }

  num_type_one[0] = FACE0 / 2;
  num_type_one[1] = FACE1 / 2;
  num_type_one[2] = FACE2 / 2;
  num_type_one[3] = FACE3 / 2;

  one_offsets[0] = 0;
  one_offsets[1] = one_offsets[0] + num_type_one[0];
  one_offsets[2] = one_offsets[1] + num_type_one[1];
  one_offsets[3] = one_offsets[2] + num_type_one[2];

  num_type_two[0] = 3 * FACE0;
  num_type_two[1] = 3 * FACE1;
  num_type_two[2] = 3 * FACE2;
  num_type_two[3] = 3 * FACE3;

  two_offsets[0] = one_offsets[3] + num_type_one[3];
  two_offsets[1] = two_offsets[0] + num_type_two[0];
  two_offsets[2] = two_offsets[1] + num_type_two[1];
  two_offsets[3] = two_offsets[2] + num_type_two[2];

  init = 1;
}

static void diff_su3_dble_bndry(su3_dble const *x, su3_dble const *y,
                                int const *dirs, double *res)
{
  int mu, nu, ix;
  init_arrays();

  res[0] = 0.0;
  res[1] = 0.0;

  for (mu = 0; mu < 4; ++mu) {
    /* Type one link diffs */
    if (dirs[mu]) {
      for (ix = 0; ix < num_type_one[mu]; ++ix) {
        res[0] +=
            norm_diff_su3(x + one_offsets[mu] + ix, y + one_offsets[mu] + ix);
      }
    }

    /* Type two link diffs */
    for (ix = 0; ix < num_type_two[mu];) {
      for (nu = 0; nu < 4; ++nu) {
        if (nu == mu) {
          continue;
        }

        if (dirs[nu]) {
          res[1] +=
              norm_diff_su3(x + two_offsets[mu] + ix, y + two_offsets[mu] + ix);
        }

        ++ix;
      }
    }
  }
}

static void off_boundary_norm(su3_dble const *x, int const *dirs, double *res)
{
  int mu, nu, ix;
  init_arrays();

  res[0] = 0.0;
  res[1] = 0.0;

  for (mu = 0; mu < 4; ++mu) {
    /* Type one link diffs */
    if (!dirs[mu]) {
      for (ix = 0; ix < num_type_one[mu]; ++ix) {
        res[0] += norm_su3(x + one_offsets[mu] + ix);
      }
    }

    /* Type two link diffs */
    for (ix = 0; ix < num_type_two[mu];) {
      for (nu = 0; nu < 4; ++nu) {
        if (nu == mu) {
          continue;
        }

        if (!dirs[nu]) {
          res[1] += norm_su3(x + two_offsets[mu] + ix);
        }

        ++ix;
      }
    }
  }
}

int main(int argc, char *argv[])
{
  int num_links_bnd;
  int my_rank, bc, ix;
  int dirs[4];
  double theta[3];
  double diff[2], total_diff[2];

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (NPROC == 1) {
    printf("No need to test communication of a single proces\n");
    exit(0);
  }

  if (my_rank == 0) {
    printf("\n");
    printf("Tests of the partial communication routines\n");
    printf("-------------------------------------------\n\n");

    printf("%dx%dx%dx%d lattice,\n", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid,\n", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n\n", L0, L1, L2, L3);
  }

  bc = 3;

  theta[0] = 0.35;
  theta[1] = -1.25;
  theta[2] = 0.78;
  set_bc_parms(bc, 0.0, 0.0, 0.0, 0.0, NULL, NULL, theta);
  print_bc_parms(2);

  if (my_rank == 0) {
    printf("-------------------------------------------\n\n");
  }

  geometry();

  num_links_bnd = 7 * BNDRY / 4;
  reset_dirs(dirs);

  new_test_module();

  { /* Test 1 */

    if (my_rank == 0) {
      register_test(1, "Partial boundary copies su3_dble");
      print_test_header(1);
    }

    for (ix = 0; ix < 16; ++ix) {
      random_ud();
      set_dirs_bitmap(dirs, ix);

      copy_partial_boundary_su3_field(udfld(), dirs);
      cm3x3_assign(num_links_bnd, udfld() + 4 * VOLUME, boundary_buffer());

      copy_boundary_su3_field(udfld());
      diff_su3_dble_bndry(boundary_buffer(), udfld() + 4 * VOLUME, dirs, diff);

      total_diff[0] = 0.0;
      total_diff[1] = 0.0;
      MPI_Reduce(diff, total_diff, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

      if (my_rank == 0) {
        print_dirs(dirs);
        printf("Total diff: {%.2e, %.2e} (should be 0.0)\n", total_diff[0],
               total_diff[1]);
        fail_test_if(1, (total_diff[0] + total_diff[1]) > 1e-10);
      }
    }

    if (my_rank == 0) {
      printf("\n-------------------------------------------\n\n");
    }
  }

  { /* Test 2 */

    if (my_rank == 0) {
      register_test(2, "Testing that non-copies are left untouched");
      print_test_header(2);
    }

    for (ix = 0; ix < 16; ++ix) {
      random_ud();
      set_dirs_bitmap(dirs, ix);

      cm3x3_zero(num_links_bnd, udfld() + 4 * VOLUME);
      copy_partial_boundary_su3_field(udfld(), dirs);

      off_boundary_norm(udfld() + 4 * VOLUME, dirs, diff);

      total_diff[0] = 0.0;
      total_diff[1] = 0.0;
      MPI_Reduce(diff, total_diff, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

      if (my_rank == 0) {
        print_dirs(dirs);
        printf("Total norm: {%.2e, %.2e} (should be 0.0)\n", total_diff[0],
               total_diff[1]);
        fail_test_if(2, (total_diff[0] + total_diff[1]) > 1e-10);
      }
    }

    if (my_rank == 0) {
      printf("\n-------------------------------------------\n\n");
    }
  }

  if (my_rank == 0) {
    report_test_results();
  }

  MPI_Finalize();
  exit(0);
}
