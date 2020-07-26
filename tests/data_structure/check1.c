
/*******************************************************************************
 *
 * Exporting CPU data Felix Ziegler, FASTSUM (2020)
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

  int my_rank, bc, nsize, ix;
  double phi[2], phi_prime[2], theta[3]; 

  char cnfg_dir[256], cnfg[NAME_SIZE];
  char piup_out[NAME_SIZE];
  char pidn_out[NAME_SIZE];
  char scnfg[NAME_SIZE];
  
  FILE *flog = NULL;
  FILE *fin = NULL;
  FILE *fout1 = NULL;
  FILE *fout2 = NULL;

  int * piup;
  int * pidn;


  /* double precision pointers */

  spinor_dble ** psd;
  spinor_dble * d_sp_aux;

  pauli_dble * md;

  /* single precision pointers */

  su3 * u;

  spinor ** ps;
  spinor * s_sp_aux;

  pauli * m;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    flog = freopen("check1.log", "w", stdout); 
    fin = freopen("check1.in", "r", stdin);

    printf("\n");
    printf("Checking data structures in openQCD\n");
    printf("----------------------------------------\n\n");

    printf("%dx%dx%dx%d lattice, ", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid, ", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n\n", L0, L1, L2, L3);

    bc = find_opt(argc, argv, "-bc");

    if (bc != 0) {
      error_root(sscanf(argv[bc + 1], "%d", &bc) != 1, 1, "main [check1.c]",
                 "Syntax: check1 [-bc <type>]");

    read_line("cnfg_dir", "%s\n", cnfg_dir);
    fclose(fin);

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

  /* Export links to binary data file */
  
  MPI_Bcast(cnfg_dir, NAME_SIZE, MPI_CHAR, 0, MPI_COMM_WORLD);

  check_dir_root(cnfg_dir);
  nsize = name_size("%s/piup-128-128-128-128", cnfg_dir);
  error_root(nsize >= NAME_SIZE, 1, "main [check1.c]",
             "cnfg_dir name is too long");
  nsize = name_size("%s/pidn-128-128-128-128", cnfg_dir);
  error_root(nsize >= NAME_SIZE, 1, "main [check1.c]",
             "cnfg_dir name is too long");

  if (my_rank == 0) {
    printf("Export random gauge field configurations to the file\n"
           "%s/u-cnfg-%d-%d-%d-%d.\n",
           cnfg_dir, L0, L1, L2, L3);
  }

  sprintf(cnfg, "%s/dp-u-%d-%d-%d-%d", cnfg_dir, L0, L1, L2, L3);
  mod_export_cnfg(cnfg);

  sprintf(cnfg, "%s/sp-u-%d-%d-%d-%d", cnfg_dir, L0, L1, L2, L3);
  assign_ud2u();
  u = ufld();

  if(my_rank == 0)
  { 
	  fout1 = fopen(cnfg, "wb");
	  for(ix = 0; ix < 4 * VOLUME; ix++, u+=1)
	  {
            fwrite((float*) u, sizeof(float), 18, fout1); 
	  }  
	  fclose(fout1);
  }

  /* write piup and pidn to binary data file */

  piup = iup[VOLUME / 2];
  pidn = idn[VOLUME / 2];

  sprintf(piup_out, "%s/piup-%d-%d-%d-%d", cnfg_dir, L0, L1, L2, L3);
  sprintf(pidn_out, "%s/pidn-%d-%d-%d-%d", cnfg_dir, L0, L1, L2, L3);

  if(my_rank == 0)
  {
	  fout1=fopen(piup_out, "wb");
	  fout2=fopen(pidn_out, "wb");
	  for(ix = 0; ix < VOLUME / 2; ix++, piup+=4, pidn+=4)
	  {
             fwrite(piup, sizeof(int), 4, fout1);
             fwrite(pidn, sizeof(int), 4, fout2);
	  }

	  fclose(fout1);
	  fclose(fout2);

  }

  /* write spinor source to binary data file */

   alloc_wsd(2);
   alloc_ws(2);

   ps = reserve_ws(2);
   psd = reserve_wsd(2); 

   random_sd(VOLUME, psd[0], 1.0);
   assign_sd2s(VOLUME, psd[0], ps[0]);

   sprintf(scnfg, "%s/dp-s-%d-%d-%d-%d", cnfg_dir, L0, L1, L2, L3);

   d_sp_aux = psd[0];

   if(my_rank == 0)
   {
	   fout1=fopen(scnfg, "wb");
	   for(ix = 0; ix < VOLUME; ix++, d_sp_aux++)
	   {
             fwrite((double *) d_sp_aux, sizeof(double), 24, fout1);
	   }
	   fclose(fout1);
   }
   
   sprintf(scnfg, "%s/sp-s-%d-%d-%d-%d", cnfg_dir, L0, L1, L2, L3);

   s_sp_aux = ps[0];

   if(my_rank == 0)
   {
	   fout1=fopen(scnfg, "wb");
	   for(ix = 0; ix < VOLUME; ix++, s_sp_aux++)
	   {
             fwrite((float *) s_sp_aux, sizeof(float), 24, fout1);
	   }
	   fclose(fout1);
   }


   /* Write spinor result to binary data file */

   Dw_dble(0.0, psd[0], psd[1]);

   d_sp_aux = psd[1];
   
   sprintf(scnfg, "%s/dp-r-%d-%d-%d-%d", cnfg_dir, L0, L1, L2, L3);

   if(my_rank == 0)
   {
	   fout1=fopen(scnfg, "wb");
	   for(ix = 0; ix < VOLUME; ix++, d_sp_aux++)
	   {
             fwrite((double *) d_sp_aux, sizeof(double), 24, fout1);
	   }
	   fclose(fout1);
   }

   Dw(0.0f, ps[0], ps[1]);

   sprintf(scnfg, "%s/sp-r-%d-%d-%d-%d", cnfg_dir, L0, L1, L2, L3);

   s_sp_aux = ps[1];

   if(my_rank == 0)
   {
	   fout1=fopen(scnfg, "wb");
	   for(ix = 0; ix < VOLUME; ix++, s_sp_aux++)
	   {
             fwrite((float *) s_sp_aux, sizeof(float), 24, fout1);
	   }
	   fclose(fout1);
   }

   release_ws();
   release_wsd();


   /* Exporting the Pauli term to binary data file */

   md = swdfld();

   sprintf(scnfg, "%s/dp-m-%d-%d-%d-%d", cnfg_dir, L0, L1, L2, L3);
   if(my_rank == 0)
   {
	   fout1 = fopen(scnfg, "wb");
	   for(ix = 0; ix < 2 * VOLUME; ix++, md++)
	   {
	      fwrite((double*) md, sizeof(double), 36, fout1);
	   }
	   fclose(fout1);
   }

   m = swfld();
   sprintf(scnfg, "%s/sp-m-%d-%d-%d-%d", cnfg_dir, L0, L1, L2, L3);
   if(my_rank == 0)
   {
	   fout1 = fopen(scnfg, "wb");
	   for(ix = 0; ix < 2 * VOLUME; ix++, m++)
	   {
	      fwrite((float*) m, sizeof(float), 36, fout1);
	   }
	   fclose(fout1);
   }


  if (my_rank == 0) {
    fclose(flog);
  }

  MPI_Finalize();
  exit(0);
}


/* BACKUP */

/*
   if(my_rank == 0)
   {
	   for(ix = 0; ix < 4; ix++)
	   {
	   printf("%.16e\t%.16e\n", (*sp_aux).c1.c1.re, (*sp_aux).c1.c1.im);
	   sp_aux++;
	   }
   }
*/
/*  if(my_rank == 0)
  {
	  printf("%d\t%d\t%d\n", 0, (*piup), (*pidn));
  }
  */ 
