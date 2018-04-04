
/*******************************************************************************
 *
 * File time01_su3_dble.c
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Timing of the communication routines for su3_dble
 *
 *******************************************************************************/

#define MAIN_PROGRAM

#include "field_com.h"
#include "global.h"
#include "lattice.h"
#include "mpi.h"
#include "random.h"
#include "uflds.h"

int main(int argc, char *argv[])
{
  int my_rank, bc, count, nt;
  double phi[2], phi_prime[2], theta[3];
  double wt1, wt2, wdt;
  FILE *flog = NULL;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (NPROC == 1) {
    printf("No need to time communication of a single proces\n");
    exit(0);
  }

  if (my_rank == 0) {
    flog = freopen("time01_su3_dble.log", "w", stdout);

    printf("\n");
    printf(
        "Timing of copy_boundary_su3_field() and add_boundary_su3_field() \n");
    printf(
        "----------------------------------------------------------------\n\n");

    printf("%dx%dx%dx%d lattice, ", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid, ", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n\n", L0, L1, L2, L3);

    printf("There are %d MPI processes\n", NPROC);

    if ((VOLUME * sizeof(double)) < (64 * 1024)) {
      printf("The local size of the gauge field is %d KB\n\n",
             (int)((72 * VOLUME * sizeof(double)) / (1024)));
    } else {
      printf("The local size of the gauge field is %d MB\n\n",
             (int)((72 * VOLUME * sizeof(double)) / (1024 * 1024)));
    }

    bc = find_opt(argc, argv, "-bc");

    if (bc != 0) {
      error_root(sscanf(argv[bc + 1], "%d", &bc) != 1, 1, "main [time2.c]",
                 "Syntax: time2 [-bc <type>]");
    }
  }

  MPI_Bcast(&bc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  phi[0] = 0.123;
  phi[1] = -0.534;
  phi_prime[0] = 0.912;
  phi_prime[1] = 0.078;
  theta[0] = 0.35;
  theta[1] = -1.25;
  theta[2] = 0.78;
  set_bc_parms(bc, 0.55, 0.78, 0.9012, 1.2034, phi, phi_prime, theta);
  print_bc_parms(2);

  geometry();
#ifndef SITERANDOM
  start_ranlux(0, 12345);
#else
  start_ranlux_site(0, 12345);
#endif

  random_ud();
  set_ud_phase();

  nt = 2;
  wdt = 0.0;

  while (wdt < 10.0) {
    MPI_Barrier(MPI_COMM_WORLD);
    wt1 = MPI_Wtime();

    for (count = 0; count < nt; ++count) {
      copy_boundary_su3_field(udfld());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    wt2 = MPI_Wtime();

    wdt = wt2 - wt1;
    nt *= 2;
  }

  wdt = 1e6 * wdt / nt;

  if (my_rank == 0) {
    printf("Time per copy_boundary_su3_field():\n");
    printf("%4.3f micro sec\n\n", wdt);
  }

  random_ud();
  set_ud_phase();

  nt = 2;
  wdt = 0.0;

  while (wdt < 10.0) {
    MPI_Barrier(MPI_COMM_WORLD);
    wt1 = MPI_Wtime();

    for (count = 0; count < nt; ++count) {
      add_boundary_su3_field(udfld());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    wt2 = MPI_Wtime();

    wdt = wt2 - wt1;
    nt *= 2;
  }

  wdt = 1e6 * wdt / nt;

  if (my_rank == 0) {
    printf("Time per add_boundary_su3_field():\n");
    printf("%4.3f micro sec\n\n", wdt);
  }

  if (my_rank == 0) {
    fclose(flog);
  }

  MPI_Finalize();
  exit(0);
}
