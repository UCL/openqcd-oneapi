
/*******************************************************************************
 *
 * File time3.c
 *
 * Copyright (C) 2011-2013, 2016 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Timing of Dw_blk() and Dwhat_blk().
 *
 *******************************************************************************/

#define OPENQCD_INTERNAL

#if !defined (STATIC_SIZES)
#error : This test cannot be compiled with dynamic lattice sizes
#endif

#include "dirac.h"
#include "global.h"
#include "lattice.h"
#include "linalg.h"
#include "mpi.h"
#include "random.h"
#include "sflds.h"
#include "sw_term.h"
#include "uflds.h"

int main(int argc, char *argv[])
{
  int my_rank, bc, count, nt;
  int n, nb, isw, bs[4];
  float mu;
  double phi[2], phi_prime[2], theta[3];
  double wt1, wt2, wdt;
  block_t *b;
  FILE *flog = NULL, *fin = NULL;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    flog = freopen("time3.log", "w", stdout);
    fin = freopen("check7.in", "r", stdin);

    printf("\n");
    printf("Timing of Dw_blk() and Dwhat_blk()\n");
    printf("----------------------------------\n\n");

    printf("%dx%dx%dx%d lattice, ", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid, ", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n\n", L0, L1, L2, L3);

    if (NPROC > 1) {
      printf("There are %d MPI processes\n", NPROC);
    } else {
      printf("There is 1 MPI process\n");
    }

    if ((VOLUME * sizeof(float)) < (64 * 1024)) {
      printf("The local size of the gauge field is %d KB\n",
             (int)((72 * VOLUME * sizeof(float)) / (1024)));
      printf("The local size of a quark field is %d KB\n",
             (int)((24 * VOLUME * sizeof(float)) / (1024)));
    } else {
      printf("The local size of the gauge field is %d MB\n",
             (int)((72 * VOLUME * sizeof(float)) / (1024 * 1024)));
      printf("The local size of a quark field is %d MB\n",
             (int)((24 * VOLUME * sizeof(float)) / (1024 * 1024)));
    }

#if (defined x64)
#if (defined AVX)
    printf("Using AVX instructions\n");
#else
    printf("Using SSE3 instructions and 16 xmm registers\n");
#endif
#if (defined P3)
    printf("Assuming SSE prefetch instructions fetch 32 bytes\n");
#elif (defined PM)
    printf("Assuming SSE prefetch instructions fetch 64 bytes\n");
#elif (defined P4)
    printf("Assuming SSE prefetch instructions fetch 128 bytes\n");
#else
    printf("SSE prefetch instructions are not used\n");
#endif
#endif
    printf("\n");

    read_line("bs", "%d %d %d %d", &bs[0], &bs[1], &bs[2], &bs[3]);
    fclose(fin);

    printf("bs = %d %d %d %d\n\n", bs[0], bs[1], bs[2], bs[3]);

    bc = find_opt(argc, argv, "-bc");

    if (bc != 0) {
      error_root(sscanf(argv[bc + 1], "%d", &bc) != 1, 1, "main [time3.c]",
                 "Syntax: time3 [-bc <type>]");
    }
  }

  MPI_Bcast(&bc, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (bc == 3) {
    set_ani_parms(1, 1.5, 4.3, 1.5, 0.9, 1.0, 1.0, 0.87, 1.23);
    print_ani_parms();
    set_lat_parms(5.5, 1.0, 0, NULL, 1.0);
  } else {
    set_no_ani_parms();
    set_lat_parms(5.5, 1.0, 0, NULL, 1.978);
  }

  print_lat_parms();

  MPI_Bcast(bs, 4, MPI_INT, 0, MPI_COMM_WORLD);
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

  set_sap_parms(bs, 0, 1, 1);
  alloc_bgr(SAP_BLOCKS);

  set_sw_parms(-0.0123);
  mu = 0.0785f;

  random_ud();
  set_ud_phase();
  sw_term(NO_PTS);
  assign_ud2ubgr(SAP_BLOCKS);
  assign_swd2swbgr(SAP_BLOCKS, NO_PTS);

  b = blk_list(SAP_BLOCKS, &nb, &isw);
  random_s((*b).vol, (*b).s[0], 1.0f);

  nt = (int)(2.0e6f / (double)(VOLUME));
  if (nt < 2) {
    nt = 2;
  }
  wdt = 0.0;

  while (wdt < 5.0) {
    MPI_Barrier(MPI_COMM_WORLD);
    wt1 = MPI_Wtime();
    for (count = 0; count < nt; count++) {
      for (n = 0; n < nb; n++) {
        Dw_blk(SAP_BLOCKS, n, mu, 0, 1);
        Dw_blk(SAP_BLOCKS, n, mu, 1, 2);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    wt2 = MPI_Wtime();

    wdt = wt2 - wt1;
    nt *= 2;
  }

  wdt = 1.0e6 * wdt / ((double)(nt) * (double)(VOLUME));

  if (my_rank == 0) {
    printf("Time per lattice point for Dw_blk():\n");
    printf("%4.3f micro sec (%d Mflops)\n\n", wdt, (int)(1920.0 / wdt));
  }

  nt = (int)(2.0e6f / (double)(VOLUME));
  if (nt < 2) {
    nt = 2;
  }
  wdt = 0.0;

  while (wdt < 5.0) {
    MPI_Barrier(MPI_COMM_WORLD);
    wt1 = MPI_Wtime();
    for (count = 0; count < nt; count++) {
      for (n = 0; n < nb; n++) {
        Dwhat_blk(SAP_BLOCKS, n, mu, 0, 1);
        Dwhat_blk(SAP_BLOCKS, n, mu, 1, 2);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    wt2 = MPI_Wtime();

    wdt = wt2 - wt1;
    nt *= 2;
  }

  wdt = 1.0e6 * wdt / ((double)(nt) * (double)(VOLUME));

  if (my_rank == 0) {
    printf("Time per lattice point for Dwhat_blk():\n");
    printf("%4.3f micro sec (%d Mflops)\n\n", wdt, (int)(1908.0 / wdt));
    fclose(flog);
  }

  MPI_Finalize();
  exit(0);
}
