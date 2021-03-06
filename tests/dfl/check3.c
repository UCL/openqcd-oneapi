
/*******************************************************************************
 *
 * File check3.c
 *
 * Copyright (C) 2011-2013, 2016 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Check of the solver for the little Dirac equation.
 *
 *******************************************************************************/

#define OPENQCD_INTERNAL

#if !defined (STATIC_SIZES)
#error : This test cannot be compiled with dynamic lattice sizes
#endif

#include "archive.h"
#include "dfl.h"
#include "dirac.h"
#include "global.h"
#include "lattice.h"
#include "linalg.h"
#include "little.h"
#include "mpi.h"
#include "random.h"
#include "sap.h"
#include "sflds.h"
#include "stout_smearing.h"
#include "sw_term.h"
#include "uflds.h"
#include "vflds.h"

int my_rank, id, first, last, step;
int bs[4], Ns, nkv, nmx, eoflg, bc;
double kappa, csw, mu, cF, cF_prime;
int has_tts;
double nu, xi, cR, cT, us_gauge, ut_gauge, us_fermion, ut_fermion;
int n_smear;
double rho_t, rho_s;
double phi[2], phi_prime[2], theta[3], m0, res;
char cnfg_dir[NAME_SIZE], cnfg_file[NAME_SIZE], nbase[NAME_SIZE];

static void new_subspace(void)
{
  int nb, isw, ifail;
  int n, nmax, k, l;
  spinor **mds, **ws;
  sap_parms_t sp;

  blk_list(SAP_BLOCKS, &nb, &isw);

  if (nb == 0) {
    alloc_bgr(SAP_BLOCKS);
  }

  assign_ud2ubgr(SAP_BLOCKS);
  sw_term(NO_PTS);
  ifail = assign_swd2swbgr(SAP_BLOCKS, ODD_PTS);

  error(ifail != 0, 1, "new_subspace [check3.c]",
        "Inversion of the SW term was not safe");

  sp = sap_parms();
  nmax = 6;
  mds = reserve_ws(Ns);
  ws = reserve_ws(1);

  for (k = 0; k < Ns; k++) {
    random_s(VOLUME, mds[k], 1.0f);
    bnd_s2zero(ALL_PTS, mds[k]);
  }

  for (n = 0; n < nmax; n++) {
    for (k = 0; k < Ns; k++) {
      assign_s2s(VOLUME, mds[k], ws[0]);
      set_s2zero(VOLUME, mds[k]);

      for (l = 0; l < sp.ncy; l++) {
        sap(0.01f, 1, sp.nmr, mds[k], ws[0]);
      }
    }

    for (k = 0; k < Ns; k++) {
      for (l = 0; l < k; l++) {
        project(VOLUME, 1, mds[k], mds[l]);
      }

      normalize(VOLUME, 1, mds[k]);
    }
  }

  dfl_subspace(mds);

  release_ws();
  release_ws();
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
  long section_pos;

  if (my_rank == 0) {
    section_pos = find_optional_section("Anisotropy parameters");

    if (section_pos == No_Section_Found) {
      has_tts = 1;
      nu = 1.0;
      xi = 1.0;
      cR = 1.0;
      cT = 1.0;
      us_gauge = 1.0;
      ut_gauge = 1.0;
      us_fermion = 1.0;
      ut_fermion = 1.0;
    } else {
      read_line("use_tts", "%d", &has_tts);
      read_line("nu", "%lf", &nu);
      read_line("xi", "%lf", &xi);
      read_line("cR", "%lf", &cR);
      read_line("cT", "%lf", &cT);
      read_optional_line("us_gauge", "%lf", &us_gauge, 1.0);
      read_optional_line("ut_gauge", "%lf", &ut_gauge, 1.0);
      read_optional_line("us_fermion", "%lf", &us_fermion, 1.0);
      read_optional_line("ut_fermion", "%lf", &ut_fermion, 1.0);
    }
  }

  MPI_Bcast(&has_tts, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nu, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&xi, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cR, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cT, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&us_gauge, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ut_gauge, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&us_fermion, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ut_fermion, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

static void read_lattice_section(FILE *fin)
{
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
}

static void read_smearing_section(FILE *fin)
{
  long section_pos;

  if (my_rank == 0) {
    section_pos = find_optional_section("Smearing parameters");

    if (section_pos == No_Section_Found) {
      n_smear = 0;
      rho_t = 0.0;
      rho_s = 0.0;
    } else {
      read_line("n_smear", "%d", &n_smear);
      read_line("rho_t", "%lf", &rho_t);
      read_line("rho_s", "%lf", &rho_s);
    }
  }

  MPI_Bcast(&n_smear, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rho_t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rho_s, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

static void read_bc_section(FILE *fin)
{
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
}

static void read_dfl_section(FILE *fin)
{
  if (my_rank == 0) {
    find_section("DFL");
    read_line("bs", "%d %d %d %d", bs, bs + 1, bs + 2, bs + 3);
    read_line("Ns", "%d", &Ns);
  }

  MPI_Bcast(bs, 4, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Ns, 1, MPI_INT, 0, MPI_COMM_WORLD);
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

int main(int argc, char *argv[])
{
  int nsize, icnfg, status;
  int nv;
  double rho, nrm, del;
  double wt1, wt2, wdt;
  complex_dble **wvd, z;
  lat_parms_t lat;
  dfl_parms_t dfl;
  FILE *flog = NULL, *fin = NULL;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    flog = freopen("check3.log", "w", stdout);
    fin = freopen("check3.in", "r", stdin);

    printf("\n");
    printf("Check of the solver for the little Dirac equation\n");
    printf("-------------------------------------------------\n\n");

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
  read_dfl_section(fin);
  read_gcr_section(fin);

  if (my_rank == 0) {
    fclose(fin);
  }

  set_ani_parms(has_tts, nu, xi, cR, cT, us_gauge, ut_gauge, us_fermion,
                ut_fermion);
  print_ani_parms();

  lat = set_lat_parms(1.5, 1.0, 1, &kappa, csw);
  print_lat_parms();

  set_stout_smearing_parms(n_smear, rho_t, rho_s, 0, 1);
  print_stout_smearing_parms();

  set_bc_parms(bc, 1.0, 1.0, cF, cF_prime, phi, phi_prime, theta);
  print_bc_parms(3);

  set_sap_parms(bs, 1, 4, 5);
  m0 = lat.m0[0];
  set_sw_parms(m0);
  set_tm_parms(eoflg);
  dfl = set_dfl_parms(bs, Ns);
  nv = Ns * VOLUME / (bs[0] * bs[1] * bs[2] * bs[3]);

  start_ranlux(0, 1234);
  geometry();

  alloc_ws(Ns + 1);
  alloc_wv(2 * nkv + 1);
  alloc_wvd(7);
  wvd = reserve_wvd(4);

  if (my_rank == 0) {
    printf("mu = %.6f\n", mu);
    printf("eoflg = %d\n\n", eoflg);

    printf("bs = (%d,%d,%d,%d)\n", dfl.bs[0], dfl.bs[1], dfl.bs[2], dfl.bs[3]);
    printf("Ns = %d\n\n", dfl.Ns);

    printf("nkv = %d\n", nkv);
    printf("nmx = %d\n", nmx);
    printf("res = %.2e\n\n", res);

    printf("Configurations %sn%d -> %sn%d in steps of %d\n\n", nbase, first,
           nbase, last, step);
    fflush(flog);
  }

  error_root(((last - first) % step) != 0, 1, "main [check3.c]",
             "last-first is not a multiple of step");
  check_dir_root(cnfg_dir);

  nsize = name_size("%s/%sn%d", cnfg_dir, nbase, last);
  error_root(nsize >= NAME_SIZE, 1, "main [check3.c]",
             "cnfg_dir name is too long");

  for (icnfg = first; icnfg <= last; icnfg += step) {
    sprintf(cnfg_file, "%s/%sn%d", cnfg_dir, nbase, icnfg);
    import_cnfg(cnfg_file);
    set_ud_phase();
    smear_fields();

    if (my_rank == 0) {
      printf("Configuration no %d\n", icnfg);
      fflush(flog);
    }

    new_subspace();
    random_vd(nv, wvd[0], 1.0);
    nrm = sqrt(vnorm_square_dble(nv, 1, wvd[0]));
    assign_vd2vd(nv, wvd[0], wvd[2]);
    set_Awhat(mu);

    MPI_Barrier(MPI_COMM_WORLD);
    wt1 = MPI_Wtime();

    rho = ltl_gcr(nkv, nmx, res, mu, wvd[0], wvd[1], &status);

    MPI_Barrier(MPI_COMM_WORLD);
    wt2 = MPI_Wtime();
    wdt = wt2 - wt1;

    z.re = -1.0;
    z.im = 0.0;
    mulc_vadd_dble(nv, wvd[2], wvd[0], z);
    del = vnorm_square_dble(nv, 1, wvd[2]);
    error_root(del != 0.0, 1, "main [check3.c]",
               "Source field is not preserved");

    set_Aw(mu);
    set_Awhat(mu);
    Aw_dble(wvd[1], wvd[2]);
    mulc_vadd_dble(nv, wvd[2], wvd[0], z);
    Aweeinv_dble(wvd[2], wvd[3]);
    assign_vd2vd(nv / 2, wvd[3], wvd[2]);
    del = sqrt(vnorm_square_dble(nv, 1, wvd[2]));

    if (my_rank == 0) {
      printf("status = %d\n", status);
      printf("rho   = %.2e, res   = %.2e\n", rho, res);
      printf("check = %.2e, check = %.2e\n", del, del / nrm);
      printf("time = %.2e sec (total)\n", wdt);
      if (status > 0) {
        printf("     = %.2e usec (per point and GCR iteration)",
               (1.0e6 * wdt) / ((double)(status) * (double)(VOLUME)));
      }
      printf("\n\n");
      fflush(flog);
    }

    ltl_gcr(nkv, nmx, res, mu, wvd[0], wvd[0], &status);
    mulc_vadd_dble(nv, wvd[0], wvd[1], z);
    del = vnorm_square_dble(nv, 1, wvd[0]);
    error_root(del != 0.0, 1, "main [check3.c]",
               "Incorrect result when the input and output fields coincide");

    unsmear_fields();
  }

  if (my_rank == 0) {
    fclose(flog);
  }

  MPI_Finalize();
  exit(0);
}
