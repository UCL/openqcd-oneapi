
/*******************************************************************************
 *
 * File time02_su3_alg_dble.c
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Timing of the communication routines for su3_alg_dble
 *
 *******************************************************************************/

#define MAIN_PROGRAM

#include "field_com.h"
#include "global.h"
#include "lattice.h"
#include "mpi.h"
#include "random.h"
#include "uflds.h"

static int init = 0;
static size_t n;
static su3_alg_dble *alg_field = NULL;

static void alloc_alg_field(void)
{
  int bc;

  error_root(sizeof(su3_alg_dble) != (8 * sizeof(double)), 1,
             "random_alg_field [time02_su3_alg_dble.c]",
             "The su3_alg_dble structures are not properly packed");

  bc = bc_type();
  n = 4 * VOLUME + 7 * (BNDRY / 4);

  if ((cpr[0] == (NPROC0 - 1)) && ((bc == 1) || (bc == 2))) {
    n += 3;
  }

  alg_field = amalloc(n * sizeof(*alg_field), ALIGN);
  error(alg_field == NULL, 1, "random_alg_field [time02_su3_alg_dble.c]",
        "Unable to allocate memory space for the su3_alg field");

  init = 1;
}

static void random_alg_field(void)
{
  double rs[8];
  su3_alg_dble *alg_ptr, *alg_max;

  if (init == 0) {
    alloc_alg_field();
  }

  alg_ptr = alg_field;
  alg_max = alg_ptr + n;

  for (; alg_ptr < alg_max; alg_ptr++) {
    gauss_dble(rs, 8);

    alg_ptr->c1 = rs[0];
    alg_ptr->c2 = rs[1];
    alg_ptr->c3 = rs[2];
    alg_ptr->c4 = rs[3];
    alg_ptr->c5 = rs[4];
    alg_ptr->c6 = rs[5];
    alg_ptr->c7 = rs[6];
    alg_ptr->c8 = rs[7];
  }
}

int main(int argc, char *argv[])
{
  int my_rank, count, nt;
  double wt1, wt2, wdt;
  FILE *flog = NULL;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (NPROC == 1) {
    printf("No need to time communication of a single proces\n");
    exit(0);
  }

  if (my_rank == 0) {
    flog = freopen("time02_su3_alg_dble.log", "w", stdout);

    printf("\n");
    printf("Timing of copy_boundary_su3_alg_field() and "
           "add_boundary_su3_alg_field()\n");
    printf("-------------------------------------------------------------------"
           "-----\n\n");

    printf("%dx%dx%dx%d lattice, ", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid, ", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n\n", L0, L1, L2, L3);

    printf("There are %d MPI processes\n", NPROC);

    if ((VOLUME * sizeof(double)) < (64 * 1024)) {
      printf("The local size of the su3_alg field is %d KB\n\n",
             (int)((32 * VOLUME * sizeof(double)) / (1024)));
    } else {
      printf("The local size of the su3_alg field is %d MB\n\n",
             (int)((32 * VOLUME * sizeof(double)) / (1024 * 1024)));
    }
  }

  geometry();
  start_ranlux(0, 12345);

  random_alg_field();

  nt = 2;
  wdt = 0.0;

  while (wdt < 10.0) {
    MPI_Barrier(MPI_COMM_WORLD);
    wt1 = MPI_Wtime();

    for (count = 0; count < nt; ++count) {
      copy_boundary_su3_alg_field(alg_field);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    wt2 = MPI_Wtime();

    wdt = wt2 - wt1;
    nt *= 2;
  }

  wdt = 1e6 * wdt / (2 * nt);

  if (my_rank == 0) {
    printf("Time per copy_boundary_su3_alg_field():\n");
    printf("%4.3f micro sec\n\n", wdt);
  }

  random_ud();

  nt = 2;
  wdt = 0.0;

  while (wdt < 10.0) {
    MPI_Barrier(MPI_COMM_WORLD);
    wt1 = MPI_Wtime();

    for (count = 0; count < nt; ++count) {
      add_boundary_su3_alg_field(alg_field);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    wt2 = MPI_Wtime();

    wdt = wt2 - wt1;
    nt *= 2;
  }

  wdt = 1e6 * wdt / (2 * nt);

  if (my_rank == 0) {
    printf("Time per add_boundary_su3_alg_field():\n");
    printf("%4.3f micro sec\n\n", wdt);
  }

  if (my_rank == 0) {
    fclose(flog);
  }

  MPI_Finalize();
  exit(0);
}
