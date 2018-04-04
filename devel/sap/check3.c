
/*******************************************************************************
 *
 * File check3.c
 *
 * Copyright (C) 2011-2013, 2016 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Check and performance of the SAP+GCR solver.
 *
 *******************************************************************************/

#define MAIN_PROGRAM

#include "archive.h"
#include "dirac.h"
#include "global.h"
#include "lattice.h"
#include "linalg.h"
#include "mpi.h"
#include "random.h"
#include "sap.h"
#include "sflds.h"
#include "stout_smearing.h"
#include "uflds.h"

int my_rank, id, first, last, step;
int bs[4], nmr, ncy, nmx, nkv, eoflg, bc;
double mu, res, m0;
char cnfg_dir[NAME_SIZE], cnfg_file[NAME_SIZE], nbase[NAME_SIZE];

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
    print_ani_parms();
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
    read_line("eoflg", "%d", &eoflg);
  }

  MPI_Bcast(&kappa, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&csw, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&mu, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&eoflg, 1, MPI_INT, 0, MPI_COMM_WORLD);

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
    print_stout_smearing_parms();
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
  print_bc_parms(2);
}

static void read_gcr_section(FILE *fin)
{
  if (my_rank == 0) {
    find_section("GCR");
    read_line("nkv", "%d", &nkv);
    read_line("nmx", "%d", &nmx);
    read_line("res", "%lf", &res);
  }

  MPI_Bcast(&nkv, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nmx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&res, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

static void read_sap_section(FILE *fin)
{
  if (my_rank == 0) {
    find_section("SAP");
    read_iprms("bs", 4, bs);
    read_line("nmr", "%d", &nmr);
    read_line("ncy", "%d", &ncy);
  }

  MPI_Bcast(bs, 4, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nmr, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ncy, 1, MPI_INT, 0, MPI_COMM_WORLD);

  set_sap_parms(bs, 0, nmr, ncy);
}

int main(int argc, char *argv[])
{
  int isolv, nsize, icnfg, status;
  double rho, nrm, del;
  double wt1, wt2, wdt;
  spinor_dble **psd;
  lat_parms_t lat;
  sap_parms_t sap;
  tm_parms_t tm;
  FILE *flog = NULL, *fin = NULL;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    flog = freopen("check3.log", "w", stdout);
    fin = freopen("check3.in", "r", stdin);

    printf("\n");
    printf("Check and performance of the SAP+GCR solver\n");
    printf("-------------------------------------------\n\n");

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
  read_gcr_section(fin);
  read_sap_section(fin);

  if (my_rank == 0) {
    fclose(fin);
  }

  lat = lat_parms();
  sap = sap_parms();

  m0 = lat.m0[0];
  (void)set_sw_parms(m0);
  tm = set_tm_parms(eoflg);

  start_ranlux(0, 1234);
  geometry();
  alloc_ws(2 * nkv + 1);
  alloc_wsd(5);
  psd = reserve_wsd(3);

  if (my_rank == 0) {
    printf("mu = %.6f\n", mu);
    printf("eoflg = %d\n\n", tm.eoflg);

    printf("bs = (%d,%d,%d,%d)\n", sap.bs[0], sap.bs[1], sap.bs[2], sap.bs[3]);
    printf("nmr = %d\n", sap.nmr);
    printf("ncy = %d\n\n", sap.ncy);

    printf("nkv = %d\n", nkv);
    printf("nmx = %d\n", nmx);
    printf("res = %.2e\n\n", res);

    printf("Configurations %sn%d -> %sn%d in steps of %d\n\n", nbase, first,
           nbase, last, step);
    fflush(flog);
  }

  error_root(((last - first) % step) != 0, 1, "main [check3.c]",
             "last-first is not a multiple of step");

  nsize = name_size("%s/%sn%d", cnfg_dir, nbase, last);
  error_root(nsize >= NAME_SIZE, 1, "main [check3.c]",
             "cnfg_dir name is too long");

  for (icnfg = first; icnfg <= last; icnfg += step) {
    sprintf(cnfg_file, "%s/%sn%d", cnfg_dir, nbase, icnfg);
    import_cnfg(cnfg_file);

    if (my_rank == 0) {
      printf("Configuration no %d\n", icnfg);
      fflush(flog);
    }

    set_ud_phase();
    smear_fields();
    random_sd(VOLUME, psd[0], 1.0);
    bnd_sd2zero(ALL_PTS, psd[0]);
    nrm = sqrt(norm_square_dble(VOLUME, 1, psd[0]));

    for (isolv = 0; isolv < 2; isolv++) {
      assign_sd2sd(VOLUME, psd[0], psd[2]);
      set_sap_parms(bs, isolv, nmr, ncy);

      rho = sap_gcr(nkv, nmx, res, mu, psd[0], psd[1], &status);

      mulr_spinor_add_dble(VOLUME, psd[2], psd[0], -1.0);
      del = norm_square_dble(VOLUME, 1, psd[2]);
      error_root(del != 0.0, 1, "main [check3.c]",
                 "Source field is not preserved");

      Dw_dble(mu, psd[1], psd[2]);
      mulr_spinor_add_dble(VOLUME, psd[2], psd[0], -1.0);
      del = sqrt(norm_square_dble(VOLUME, 1, psd[2]));

      if (my_rank == 0) {
        printf("isolv = %d:\n", isolv);
        printf("status = %d\n", status);
        printf("rho   = %.2e, res   = %.2e\n", rho, res);
        printf("check = %.2e, check = %.2e\n", del, del / nrm);
      }

      assign_sd2sd(VOLUME, psd[0], psd[2]);

      MPI_Barrier(MPI_COMM_WORLD);
      wt1 = MPI_Wtime();

      rho = sap_gcr(nkv, nmx, res, mu, psd[2], psd[2], &status);

      MPI_Barrier(MPI_COMM_WORLD);
      wt2 = MPI_Wtime();
      wdt = wt2 - wt1;

      if (my_rank == 0) {
        printf("time = %.2e sec (total)\n", wdt);
        if (status > 0) {
          printf("     = %.2e usec (per point and GCR iteration)",
                 (1.0e6 * wdt) / ((double)(status) * (double)(VOLUME)));
        }
        printf("\n\n");
        fflush(flog);
      }

      mulr_spinor_add_dble(VOLUME, psd[2], psd[1], -1.0);
      del = norm_square_dble(VOLUME, 1, psd[2]);
      error_root(del != 0.0, 1, "main [check3.c]",
                 "Incorrect result when the input and "
                 "output fields coincide");

      unsmear_fields();
    }
  }

  if (my_rank == 0) {
    fclose(flog);
  }

  MPI_Finalize();
  exit(0);
}
