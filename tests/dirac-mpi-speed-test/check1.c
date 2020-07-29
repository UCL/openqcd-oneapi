
/*******************************************************************************
 *
 * Single-precision Dw speedtest Felix Ziegler, FASTSUM (2020)
 *
 *
 * based on: tests/archive
 *
 *******************************************************************************/

#define OPENQCD_INTERNAL

#if !defined (STATIC_SIZES)
#error : This test cannot be compiled with dynamic lattice sizes
#endif

#include "archive.h"
#include "global.h"
#include "lattice.h"
#include "mpi.h"
#include "random.h"
#include "uflds.h"
#include "sflds.h"
#include "dirac.h"
#include "sw_term.h"

static double kappa[1] = {0.1};

static lat_parms_t my_lat_parms;

int main(int argc, char *argv[])
{

  int my_rank, bc, i;
  double phi[2], phi_prime[2], theta[3];
  char name[128];
  FILE *flog = NULL;

  /* single precision pointers */

  spinor_dble **psd;
  spinor ** ps;
  double wt1, wt2, wdt;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  /*sprintf(name, "/mnt/shared/sheff_fastsum/speed-test-dirac-C/check1-%d-%d-%d-%d.log", L0, L1, L2, L3);*/
  sprintf(name, "/mnt/shared/sheff_fastsum/speed-test-dirac-C/mpi-check1-%d-%d-%d-%d.log", L0, L1, L2, L3);
   
  if (my_rank == 0) {
    flog = freopen(name, "w", stdout);

    printf("\n");
    printf("Speedtest Dw in openQCD\n");
    printf("----------------------------------------\n\n");

    printf("%dx%dx%dx%d lattice, ", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid, ", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n\n", L0, L1, L2, L3);

    bc = find_opt(argc, argv, "-bc");

    if (bc != 0) {
      error_root(sscanf(argv[bc + 1], "%d", &bc) != 1, 1, "main [check1.c]",
                 "Syntax: check1 [-bc <type>]");

    }
  }

  set_no_ani_parms();

  set_lat_parms(6.0, 1.0, 1, kappa, 1.0);
  print_lat_parms();
  my_lat_parms = lat_parms();
  if(my_rank == 0)
  {
	  printf("m[0] = %.6lf\n", my_lat_parms.m0[0]);
  }

  MPI_Bcast(&bc, 1, MPI_INT, 0, MPI_COMM_WORLD);

  phi[0] = 0.0;
  phi[1] = 0.0;
  phi_prime[0] = 0.0;
  phi_prime[1] = 0.0;
  theta[0] = 0.0;
  theta[1] = 0.0;
  theta[2] = 0.0;
  set_bc_parms(bc, 1.0, 1.0, 1.0, 1.0, phi, phi_prime, theta);
  print_bc_parms(3);

  start_ranlux(0, 123456);
  geometry();

  random_ud();
  assign_ud2u();

   alloc_wsd(1);
   alloc_ws(2);

   psd = reserve_wsd(1);
   ps = reserve_ws(2);

   random_sd(VOLUME, psd[0], 1.0);
   assign_sd2s(VOLUME, psd[0], ps[0]);

   /* Execute Dirac operators on s, store result in r */

   /* *** 1.) Dw *** */

   MPI_Barrier(MPI_COMM_WORLD);
   wt1 = MPI_Wtime();

   for(i = 0; i < 100; i++)
   {
   	Dw(0.0f, ps[0], ps[1]);
   }	   

   MPI_Barrier(MPI_COMM_WORLD);
   wt2 = MPI_Wtime();
   wdt = wt2 - wt1;

   if(my_rank == 0)
   {
     printf("time for Dw = %.2e sec (total)\n", wdt);
   }

   /* *** 2.) Dwoe *** */

   MPI_Barrier(MPI_COMM_WORLD);
   wt1 = MPI_Wtime();

   for(i = 0; i < 100; i++)
   {
   	Dwoe(ps[0], ps[1]);
   }	   

   MPI_Barrier(MPI_COMM_WORLD);
   wt2 = MPI_Wtime();
   wdt = wt2 - wt1;

   if(my_rank == 0)
   {
     printf("time for Dwoe = %.2e sec (total)\n", wdt);
   }

   /* *** 3.) Dweo *** */

   MPI_Barrier(MPI_COMM_WORLD);
   wt1 = MPI_Wtime();

   for(i = 0; i < 100; i++)
   {
   	Dweo(ps[0], ps[1]);
   }	   

   MPI_Barrier(MPI_COMM_WORLD);
   wt2 = MPI_Wtime();
   wdt = wt2 - wt1;

   if(my_rank == 0)
   {
     printf("time for Dweo = %.2e sec (total)\n", wdt);
   }

   release_ws();
   release_wsd();

  if (my_rank == 0) {
    fclose(flog);
  }

  MPI_Finalize();
  exit(0);
}
