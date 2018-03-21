
/*******************************************************************************
 *
 * File check5.c
 *
 * Copyright (C) 2011-2013, 2016 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Check and performance of the CG solver.
 *
 *******************************************************************************/

#define MAIN_PROGRAM

#include "archive.h"
#include "dirac.h"
#include "forces.h"
#include "global.h"
#include "lattice.h"
#include "linalg.h"
#include "linsolv.h"
#include "mpi.h"
#include "random.h"
#include "sflds.h"
#include "stout_smearing.h"
#include "sw_term.h"
#include "uflds.h"

static int my_rank, bc, first, last, step, nmx;
static double mu, m0, res;
static char cnfg_dir[NAME_SIZE], cnfg_file[NAME_SIZE], nbase[NAME_SIZE];

static void Dhatop_dble(spinor_dble *s, spinor_dble *r)
{
  Dwhat_dble(mu, s, r);
  mulg5_dble(VOLUME / 2, r);
  mu = -mu;
}

static void Dhatop(spinor *s, spinor *r)
{
  Dwhat((float)(mu), s, r);
  mulg5(VOLUME / 2, r);
  mu = -mu;
}

static void read_configurations_section(FILE *fin)
{
  if (my_rank == 0) {
    find_section("Configurations");
    read_line("name", "%s", nbase);
    read_line("cnfg_dir", "%s", cnfg_dir);
    read_line("first", "%d", &first);
    read_line("last", "%d", &last);
    read_line("step", "%d", &step);
  }

  MPI_Bcast(nbase, NAME_SIZE, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(cnfg_dir, NAME_SIZE, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(&first, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&last, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&step, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

static void read_anisotropy_section(FILE *fin)
{
  int has_tts, has_ani = 0;
  long section_pos;
  double nu, xi, cR, cT, us_gauge, ut_gauge;

  if (my_rank == 0) {
    section_pos = find_optional_section("Anisotropy parameters");

    if (section_pos == No_Section_Found) {
      has_ani = 0;
    } else {
      has_ani = 1;
      read_line("use_tts", "%d", &has_tts);
      read_line("nu", "%lf", &nu);
      read_line("xi", "%lf", &xi);
      read_line("cR", "%lf", &cR);
      read_line("cT", "%lf", &cT);
      read_optional_line("us_gauge", "%lf", &us_gauge, 1.0);
      read_optional_line("ut_gauge", "%lf", &ut_gauge, 1.0);
    }
  }

  MPI_Bcast(&has_ani, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (has_ani == 1) {
    MPI_Bcast(&has_tts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nu, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&xi, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cR, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cT, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&us_gauge, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ut_gauge, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    set_ani_parms(has_tts, nu, xi, cR, cT, us_gauge, ut_gauge, 1.0, 1.0);

  } else {
    set_no_ani_parms();
  }
}

static void read_lattice_section(FILE *fin)
{
  double kappa, csw;

  if (my_rank == 0) {
    find_section("Lattice parameters");
    read_line("kappa", "%lf", &kappa);
    read_line("csw", "%lf", &csw);
    read_line("mu", "%lf", &mu);
  }

  MPI_Bcast(&kappa, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&csw, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&mu, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  set_lat_parms(5.5, 1.0, 1, &kappa, csw);
  print_lat_parms();
}

static void read_smearing_section(FILE *fin)
{
  int n_smear, has_smearing = 0;
  long section_pos;
  static double rho_s, rho_t;

  if (my_rank == 0) {
    section_pos = find_optional_section("Smearing parameters");

    if (section_pos == No_Section_Found) {
      has_smearing = 0;
    } else {
      has_smearing = 1;
      read_line("n_smear", "%d", &n_smear);
      read_line("rho_t", "%lf", &rho_t);
      read_line("rho_s", "%lf", &rho_s);
    }
  }

  MPI_Bcast(&has_smearing, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (has_smearing == 1) {
    MPI_Bcast(&n_smear, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rho_t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rho_s, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    set_stout_smearing_parms(n_smear, rho_t, rho_s, 0, 1);
  } else {
    set_no_stout_smearing_parms();
  }
}

static void read_bc_section(FILE *fin)
{
  double cF, cF_prime;
  double phi[2], phi_prime[2], theta[3];

  if (my_rank == 0) {
    find_section("Boundary conditions");
    read_line("type", "%d", &bc);

    phi[0] = 0.0;
    phi[1] = 0.0;
    phi_prime[0] = 0.0;
    phi_prime[1] = 0.0;
    cF = 1.0;
    cF_prime = 1.0;

    if (bc == 1) {
      read_dprms("phi", 2, phi);
    }

    if ((bc == 1) || (bc == 2)) {
      read_dprms("phi'", 2, phi_prime);
    }

    if (bc != 3) {
      read_line("cF", "%lf", &cF);
    }

    if (bc == 2) {
      read_line("cF'", "%lf", &cF_prime);
    } else {
      cF_prime = cF;
    }

    read_dprms("theta", 3, theta);
  }

  MPI_Bcast(&bc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(phi, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(phi_prime, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cF, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cF_prime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(theta, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  set_bc_parms(bc, 1.0, 1.0, cF, cF_prime, phi, phi_prime, theta);
  print_bc_parms(3);
}

static void read_cg_section(FILE *fin)
{
  if (my_rank == 0) {
    find_section("CG");
    read_line("nmx", "%d", &nmx);
    read_line("res", "%lf", &res);
  }

  MPI_Bcast(&nmx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&res, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{
  int nsize, icnfg, status, ie;
  double rho, nrm, del;
  double wt1, wt2, wdt;
  complex_dble z;
  spinor **ws;
  spinor_dble **wsd, **psd;
  lat_parms_t lat;
  FILE *flog = NULL, *fin = NULL;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    flog = freopen("check5.log", "w", stdout);
    fin = freopen("check5.in", "r", stdin);

    printf("\n");
    printf("Check and performance of the CG solver\n");
    printf("--------------------------------------\n\n");

    printf("%dx%dx%dx%d lattice, ", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid, ", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n\n", L0, L1, L2, L3);
  }

  read_configurations_section(fin);
  read_anisotropy_section(fin);
  read_lattice_section(fin);
  read_smearing_section(fin);
  read_bc_section(fin);
  read_cg_section(fin);

  if (my_rank == 0) {
    fclose(fin);
  }

  start_ranlux(0, 1234);
  geometry();

  lat = lat_parms();

  m0 = lat.m0[0];
  set_sw_parms(m0);

  if (my_rank == 0) {
    printf("mu = %.6f\n\n", mu);

    printf("CG parameters:\n");
    printf("nmx = %d\n", nmx);
    printf("res = %.2e\n\n", res);

    printf("Configurations %sn%d -> %sn%d in steps of %d\n\n", nbase, first,
           nbase, last, step);
    fflush(flog);
  }

  alloc_ws(5);
  alloc_wsd(6);
  psd = reserve_wsd(3);

  error_root(((last - first) % step) != 0, 1, "main [check5.c]",
             "last-first is not a multiple of step");
  check_dir_root(cnfg_dir);
  nsize = name_size("%s/%sn%d", cnfg_dir, nbase, last);
  error_root(nsize >= NAME_SIZE, 1, "main [check5.c]",
             "configuration file name is too long");

  for (icnfg = first; icnfg <= last; icnfg += step) {
    sprintf(cnfg_file, "%s/%sn%d", cnfg_dir, nbase, icnfg);
    import_cnfg(cnfg_file);

    if (my_rank == 0) {
      printf("Configuration no %d\n\n", icnfg);
      fflush(flog);
    }

    set_ud_phase();
    smear_fields();
    random_sd(VOLUME, psd[0], 1.0);
    bnd_sd2zero(ALL_PTS, psd[0]);
    nrm = sqrt(norm_square_dble(VOLUME, 1, psd[0]));
    assign_sd2sd(VOLUME, psd[0], psd[2]);

    MPI_Barrier(MPI_COMM_WORLD);
    wt1 = MPI_Wtime();

    rho = tmcg(nmx, res, mu, psd[0], psd[1], &status);

    MPI_Barrier(MPI_COMM_WORLD);
    wt2 = MPI_Wtime();
    wdt = wt2 - wt1;

    z.re = -1.0;
    z.im = 0.0;
    mulc_spinor_add_dble(VOLUME, psd[2], psd[0], z);
    del = norm_square_dble(VOLUME, 1, psd[2]);
    error_root(del != 0.0, 1, "main [check5.c]",
               "Source field is not preserved");

    Dw_dble(mu, psd[1], psd[2]);
    mulg5_dble(VOLUME, psd[2]);
    Dw_dble(-mu, psd[2], psd[1]);
    mulg5_dble(VOLUME, psd[1]);
    mulc_spinor_add_dble(VOLUME, psd[1], psd[0], z);
    del = sqrt(norm_square_dble(VOLUME, 1, psd[1]));

    if (my_rank == 0) {
      printf("Solution w/o eo-preconditioning:\n");
      printf("status = %d\n", status);
      printf("rho   = %.2e, res   = %.2e\n", rho, res);
      printf("check = %.2e, check = %.2e\n", del, del / nrm);
      printf("time = %.2e sec (total)\n", wdt);
      if (status > 0) {
        printf("     = %.2e usec (per point and CG iteration)",
               (1.0e6 * wdt) / ((double)(status) * (double)(VOLUME)));
      }
      printf("\n\n");
      fflush(flog);
    }

    ws = reserve_ws(5);
    wsd = reserve_wsd(2);
    ie = sw_term(ODD_PTS);
    error_root(ie != 0, 1, "main [check5.c]",
               "Inversion of the SW term failed");
    assign_swd2sw();

    random_sd(VOLUME / 2, psd[0], 1.0);
    bnd_sd2zero(ALL_PTS, psd[0]);
    nrm = sqrt(norm_square_dble(VOLUME / 2, 1, psd[0]));
    assign_sd2sd(VOLUME / 2, psd[0], psd[2]);

    MPI_Barrier(MPI_COMM_WORLD);
    wt1 = MPI_Wtime();

    rho = cgne(VOLUME / 2, 1, Dhatop, Dhatop_dble, ws, wsd, nmx, res, psd[0],
               psd[1], &status);

    MPI_Barrier(MPI_COMM_WORLD);
    wt2 = MPI_Wtime();
    wdt = wt2 - wt1;

    z.re = -1.0;
    z.im = 0.0;
    mulc_spinor_add_dble(VOLUME / 2, psd[2], psd[0], z);
    del = norm_square_dble(VOLUME / 2, 1, psd[2]);
    error_root(del != 0.0, 1, "main [check5.c]",
               "Source field is not preserved");

    Dhatop_dble(psd[1], psd[2]);
    Dhatop_dble(psd[2], psd[1]);
    mulc_spinor_add_dble(VOLUME / 2, psd[1], psd[0], z);
    del = sqrt(norm_square_dble(VOLUME / 2, 1, psd[1]));

    if (my_rank == 0) {
      printf("Solution with eo-preconditioning:\n");
      printf("status = %d\n", status);
      printf("rho   = %.2e, res   = %.2e\n", rho, res);
      printf("check = %.2e, check = %.2e\n", del, del / nrm);
      printf("time = %.2e sec (total)\n", wdt);
      if (status > 0) {
        printf("     = %.2e usec (per point and CG iteration)",
               (1.0e6 * wdt) / ((double)(status) * (double)(VOLUME)));
      }
      printf("\n\n");
      fflush(flog);
    }

    unsmear_fields();

    release_wsd();
    release_ws();
  }

  if (my_rank == 0) {
    fclose(flog);
  }

  MPI_Finalize();
  exit(0);
}
