
/*******************************************************************************
 *
 * File qcd1.c
 *
 * Copyright (C) 2011-2013, 2016 Martin Luescher, Isabel Campos
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * HMC simulation program for QCD with Wilson quarks.
 *
 * Syntax: qcd1 -i <filename> [-noloc] [-noexp] [-rmold] [-noms]
 *                            [-c <filename> [-a [-norng]]]
 *
 * For usage instructions see the file README.qcd1.
 *
 *******************************************************************************/

#define OPENQCD_INTERNAL

#include "archive.h"
#include "flags.h"
#include "forces.h"
#include "global.h"
#include "lattice.h"
#include "mpi.h"
#include "random.h"
#include "stout_smearing.h"
#include "su3fcts.h"
#include "tcharge.h"
#include "uflds.h"
#include "update.h"
#include "utils.h"
#include "version.h"
#include "wflow.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N0 (NPROC0 * L0)
#define N1 (NPROC1 * L1)
#define N2 (NPROC2 * L2)
#define N3 (NPROC3 * L3)

typedef struct
{
  int nt, iac;
  double dH, avpl, avpl_smeared;
  double average_t_link, average_s_link;
  double average_stout_t_link, average_stout_s_link;
  complex ploop, smeared_ploop;
} dat_t;

static struct
{
  int dn, nn, tmax;
  double eps;
} file_head;

static struct
{
  int nt;
  double **Wsl, **Ysl, **Qsl;
} data;

static int my_rank, noloc, noexp, rmold, noms, norng;
static int scnfg, append, endian, loose_append;
static int level, seed;
static int cmd_seed = -1;
static int nth, ntr, dtr_log, dtr_ms, dtr_cnfg;
static int ipgrd[2], flint;
static double *Wact, *Yact, *Qtop;
static double npl, volume;

static char line[2 * NAME_SIZE + 32];
static char log_dir[NAME_SIZE], dat_dir[NAME_SIZE];
static char loc_dir[NAME_SIZE], cnfg_dir[NAME_SIZE];
static char log_file[2 * NAME_SIZE + 16], log_save[2 * NAME_SIZE + 32];
static char par_file[2 * NAME_SIZE + 16], par_save[2 * NAME_SIZE + 32];
static char dat_file[2 * NAME_SIZE + 16], dat_save[2 * NAME_SIZE + 32];
static char msdat_file[2 * NAME_SIZE + 16], msdat_save[2 * NAME_SIZE + 32];
static char rng_file[2 * NAME_SIZE + 16], rng_save[2 * NAME_SIZE + 32];
static char cnfg_file[2 * NAME_SIZE + 64], end_file[2 * NAME_SIZE + 32];
static char nbase[NAME_SIZE], cnfg[NAME_SIZE + 16];
static FILE *fin = NULL, *flog = NULL, *fdat = NULL, *fend = NULL;

static hmc_parms_t hmc;

static int write_dat(int n, dat_t *ndat)
{
  int i, iw, ic;
  stdint_t istd[2];
  double dstd[2];

  ic = 0;

  for (i = 0; i < n; i++) {
    istd[0] = (stdint_t)((*ndat).nt);
    istd[1] = (stdint_t)((*ndat).iac);

    dstd[0] = (*ndat).dH;
    dstd[1] = (*ndat).avpl;

    if (endian == openqcd_utils__BIG_ENDIAN) {
      bswap_int(2, istd);
      bswap_double(2, dstd);
    }

    iw = fwrite(istd, sizeof(stdint_t), 2, fdat);
    iw += fwrite(dstd, sizeof(double), 2, fdat);

    if (iw != 4) {
      return ic;
    }

    ic += 1;
    ndat += 1;
  }

  return ic;
}

static int read_dat(int n, dat_t *ndat)
{
  int i, ir, ic;
  stdint_t istd[2];
  double dstd[2];

  ic = 0;

  for (i = 0; i < n; i++) {
    ir = fread(istd, sizeof(stdint_t), 2, fdat);
    ir += fread(dstd, sizeof(double), 2, fdat);

    if (ir != 4) {
      return ic;
    }

    if (endian == openqcd_utils__BIG_ENDIAN) {
      bswap_int(2, istd);
      bswap_double(2, dstd);
    }

    (*ndat).nt = (int)(istd[0]);
    (*ndat).iac = (int)(istd[1]);

    (*ndat).dH = dstd[0];
    (*ndat).avpl = dstd[1];

    ic += 1;
    ndat += 1;
  }

  return ic;
}

static void alloc_data(void)
{
  int nn, tmax;
  int in;
  double **pp, *p;

  nn = file_head.nn;
  tmax = file_head.tmax;

  pp = amalloc(3 * (nn + 1) * sizeof(*pp), 3);
  p = amalloc(3 * (nn + 1) * (tmax + 1) * sizeof(*p), 4);

  error((pp == NULL) || (p == NULL), 1, "alloc_data [qcd1.c]",
        "Unable to allocate data arrays");

  data.Wsl = pp;
  data.Ysl = pp + nn + 1;
  data.Qsl = pp + 2 * (nn + 1);

  for (in = 0; in < (3 * (nn + 1)); in++) {
    *pp = p;
    pp += 1;
    p += tmax;
  }

  Wact = p;
  p += nn + 1;
  Yact = p;
  p += nn + 1;
  Qtop = p;
}

static void write_file_head(void)
{
  int iw;
  stdint_t istd[3];
  double dstd[1];

  istd[0] = (stdint_t)(file_head.dn);
  istd[1] = (stdint_t)(file_head.nn);
  istd[2] = (stdint_t)(file_head.tmax);
  dstd[0] = file_head.eps;

  if (endian == openqcd_utils__BIG_ENDIAN) {
    bswap_int(3, istd);
    bswap_double(1, dstd);
  }

  iw = fwrite(istd, sizeof(stdint_t), 3, fdat);
  iw += fwrite(dstd, sizeof(double), 1, fdat);

  error_root(iw != 4, 1, "write_file_head [qcd1.c]", "Incorrect write count");
}

static void check_file_head(void)
{
  int ir;
  stdint_t istd[3];
  double dstd[1];

  ir = fread(istd, sizeof(stdint_t), 3, fdat);
  ir += fread(dstd, sizeof(double), 1, fdat);

  error_root(ir != 4, 1, "check_file_head [qcd1.c]", "Incorrect read count");

  if (endian == openqcd_utils__BIG_ENDIAN) {
    bswap_int(3, istd);
    bswap_double(1, dstd);
  }

  error_root(
      ((int)(istd[0]) != file_head.dn) || ((int)(istd[1]) != file_head.nn) ||
          ((int)(istd[2]) != file_head.tmax) || (dstd[0] != file_head.eps),
      1, "check_file_head [qcd1.c]", "Unexpected value of dn,nn,tmax or eps");
}

static void write_data(void)
{
  int iw, nn, tmax;
  int in, t;
  stdint_t istd[1];
  double dstd[1];

  istd[0] = (stdint_t)(data.nt);

  if (endian == openqcd_utils__BIG_ENDIAN) {
    bswap_int(1, istd);
  }

  iw = fwrite(istd, sizeof(stdint_t), 1, fdat);

  nn = file_head.nn;
  tmax = file_head.tmax;

  for (in = 0; in <= nn; in++) {
    for (t = 0; t < tmax; t++) {
      dstd[0] = data.Wsl[in][t];

      if (endian == openqcd_utils__BIG_ENDIAN) {
        bswap_double(1, dstd);
      }

      iw += fwrite(dstd, sizeof(double), 1, fdat);
    }
  }

  for (in = 0; in <= nn; in++) {
    for (t = 0; t < tmax; t++) {
      dstd[0] = data.Ysl[in][t];

      if (endian == openqcd_utils__BIG_ENDIAN) {
        bswap_double(1, dstd);
      }

      iw += fwrite(dstd, sizeof(double), 1, fdat);
    }
  }

  for (in = 0; in <= nn; in++) {
    for (t = 0; t < tmax; t++) {
      dstd[0] = data.Qsl[in][t];

      if (endian == openqcd_utils__BIG_ENDIAN) {
        bswap_double(1, dstd);
      }

      iw += fwrite(dstd, sizeof(double), 1, fdat);
    }
  }

  error_root(iw != (1 + 3 * (nn + 1) * tmax), 1, "write_data [qcd1.c]",
             "Incorrect write count");
}

static int read_data(void)
{
  int ir, nn, tmax;
  int in, t;
  stdint_t istd[1];
  double dstd[1];

  ir = fread(istd, sizeof(stdint_t), 1, fdat);

  if (ir != 1) {
    return 0;
  }

  if (endian == openqcd_utils__BIG_ENDIAN) {
    bswap_int(1, istd);
  }

  data.nt = (int)(istd[0]);

  nn = file_head.nn;
  tmax = file_head.tmax;

  for (in = 0; in <= nn; in++) {
    for (t = 0; t < tmax; t++) {
      ir += fread(dstd, sizeof(double), 1, fdat);

      if (endian == openqcd_utils__BIG_ENDIAN) {
        bswap_double(1, dstd);
      }

      data.Wsl[in][t] = dstd[0];
    }
  }

  for (in = 0; in <= nn; in++) {
    for (t = 0; t < tmax; t++) {
      ir += fread(dstd, sizeof(double), 1, fdat);

      if (endian == openqcd_utils__BIG_ENDIAN) {
        bswap_double(1, dstd);
      }

      data.Ysl[in][t] = dstd[0];
    }
  }

  for (in = 0; in <= nn; in++) {
    for (t = 0; t < tmax; t++) {
      ir += fread(dstd, sizeof(double), 1, fdat);

      if (endian == openqcd_utils__BIG_ENDIAN) {
        bswap_double(1, dstd);
      }

      data.Qsl[in][t] = dstd[0];
    }
  }

  error_root(ir != (1 + 3 * (nn + 1) * tmax), 1, "read_data [qcd1.c]",
             "Read error or incomplete data record");

  return 1;
}

static void read_dirs(void)
{
  if (my_rank == 0) {
    find_section("Run name");
    read_line("name", "%s", nbase);

    find_section("Directories");
    read_line("log_dir", "%s", log_dir);
    read_line("dat_dir", "%s", dat_dir);
    if (noloc == 0) {
      read_line("loc_dir", "%s", loc_dir);
    } else {
      loc_dir[0] = '\0';
    }
    if ((noexp == 0) || ((scnfg) && (cnfg[strlen(cnfg) - 1] != '*'))) {
      read_line("cnfg_dir", "%s", cnfg_dir);
    } else {
      cnfg_dir[0] = '\0';
    }
  }

  mpc_bcast_c(nbase, NAME_SIZE);
  mpc_bcast_c(log_dir, NAME_SIZE);
  mpc_bcast_c(dat_dir, NAME_SIZE);
  mpc_bcast_c(loc_dir, NAME_SIZE);
  mpc_bcast_c(cnfg_dir, NAME_SIZE);
}

static void setup_files(void)
{
  if (noloc == 0) {
    check_dir(loc_dir);
  }

  if (noexp == 0) {
    check_dir_root(cnfg_dir);
  }

  check_dir_root(log_dir);
  check_dir_root(dat_dir);
  error_root(name_size("%s/%s.log~", log_dir, nbase) >= NAME_SIZE, 1,
             "setup_files [qcd1.c]", "log_dir name is too long");
  error_root(name_size("%s/%s.ms.dat~", dat_dir, nbase) >= NAME_SIZE, 1,
             "setup_files [qcd1.c]", "dat_dir name is too long");

  sprintf(log_file, "%s/%s.log", log_dir, nbase);
  sprintf(par_file, "%s/%s.par", dat_dir, nbase);
  sprintf(dat_file, "%s/%s.dat", dat_dir, nbase);
  sprintf(msdat_file, "%s/%s.ms.dat", dat_dir, nbase);
  sprintf(rng_file, "%s/%s.rng", dat_dir, nbase);
  sprintf(end_file, "%s/%s.end", log_dir, nbase);
  sprintf(log_save, "%s~", log_file);
  sprintf(par_save, "%s~", par_file);
  sprintf(dat_save, "%s~", dat_file);
  sprintf(msdat_save, "%s~", msdat_file);
  sprintf(rng_save, "%s~", rng_file);
}

#if !defined(STATIC_SIZES)
static void read_lattize_sizes(void)
{
  int local_lattice_sizes[4], mpi_layout[4], block_layout[4];

  if (my_rank == 0) {
    find_section("Lattice sizes");
    read_iprms("number_of_processes", 4, mpi_layout);
    read_iprms("local_lattice_sizes", 4, local_lattice_sizes);
    read_iprms("number_of_blocks", 4, block_layout);
  }

  mpc_bcast_i(mpi_layout, 4);
  mpc_bcast_i(local_lattice_sizes, 4);
  mpc_bcast_i(block_layout, 4);

  set_lattice_sizes(mpi_layout, local_lattice_sizes, block_layout);
}
#endif

static void read_lat_parms(void)
{
  int nk;
  double beta, c0, csw, *kappa;

  if (my_rank == 0) {
    find_section("Lattice parameters");
    read_line("beta", "%lf", &beta);
    read_line("c0", "%lf", &c0);
    nk = count_tokens("kappa");
    read_line("csw", "%lf", &csw);
  }

  mpc_bcast_d(&beta, 1);
  mpc_bcast_d(&c0, 1);
  mpc_bcast_i(&nk, 1);
  mpc_bcast_d(&csw, 1);

  if (nk > 0) {
    kappa = malloc(nk * sizeof(*kappa));
    error(kappa == NULL, 1, "read_lat_parms [qcd1.c]",
          "Unable to allocate parameter array");
    if (my_rank == 0) {
      read_dprms("kappa", nk, kappa);
    }
    mpc_bcast_d(kappa, nk);
  } else {
    kappa = NULL;
  }

  set_lat_parms(beta, c0, nk, kappa, csw);

  if (nk > 0) {
    free(kappa);
  }

  if (append) {
    check_lat_parms(fdat);
  } else {
    write_lat_parms(fdat);
  }
}

static void read_bc_parms(void)
{
  int bc;
  double cG, cG_prime, cF, cF_prime;
  double phi[2], phi_prime[2], theta[3];

  if (my_rank == 0) {
    find_section("Boundary conditions");
    read_line("type", "%d", &bc);

    phi[0] = 0.0;
    phi[1] = 0.0;
    phi_prime[0] = 0.0;
    phi_prime[1] = 0.0;
    cG = 1.0;
    cG_prime = 1.0;
    cF = 1.0;
    cF_prime = 1.0;

    if (bc == 1) {
      read_dprms("phi", 2, phi);
    }

    if ((bc == 1) || (bc == 2)) {
      read_dprms("phi'", 2, phi_prime);
    }

    if (bc != 3) {
      read_line("cG", "%lf", &cG);
      read_line("cF", "%lf", &cF);
    }

    if (bc == 2) {
      read_line("cG'", "%lf", &cG_prime);
      read_line("cF'", "%lf", &cF_prime);
    }

    theta[0] = 0.0;
    theta[1] = 0.0;
    theta[2] = 0.0;

    read_optional_dprms("theta", 3, theta);
  }

  mpc_bcast_i(&bc, 1);
  mpc_bcast_d(phi, 2);
  mpc_bcast_d(phi_prime, 2);
  mpc_bcast_d(&cG, 1);
  mpc_bcast_d(&cG_prime, 1);
  mpc_bcast_d(&cF, 1);
  mpc_bcast_d(&cF_prime, 1);
  mpc_bcast_d(theta, 3);

  set_bc_parms(bc, cG, cG_prime, cF, cF_prime, phi, phi_prime, theta);

  if (append) {
    check_bc_parms(fdat);
  } else {
    write_bc_parms(fdat);
  }
}

static void read_schedule(void)
{
  int ie, ir, iw;
  stdint_t istd[3];

  if (my_rank == 0) {
    find_section("MD trajectories");
    read_line("nth", "%d", &nth);
    read_line("ntr", "%d", &ntr);
    read_line("dtr_log", "%d", &dtr_log);
    if (noms == 0) {
      read_line("dtr_ms", "%d", &dtr_ms);
    } else {
      dtr_ms = 0;
    }
    read_line("dtr_cnfg", "%d", &dtr_cnfg);

    error_root((append != 0) && (nth != 0), 1, "read_schedule [qcd1.c]",
               "Continuation run: nth must be equal to zero");

    ie = 0;
    ie |= (nth < 0);
    ie |= (ntr < 1);
    ie |= (dtr_log < 1);
    ie |= (dtr_log > dtr_cnfg);
    ie |= ((dtr_cnfg % dtr_log) != 0);
    ie |= ((nth % dtr_cnfg) != 0);
    ie |= ((ntr % dtr_cnfg) != 0);

    if (noms == 0) {
      ie |= (dtr_ms < dtr_log);
      ie |= (dtr_ms > dtr_cnfg);
      ie |= ((dtr_ms % dtr_log) != 0);
      ie |= ((dtr_cnfg % dtr_ms) != 0);
    }

    error_root(ie != 0, 1, "read_schedule [qcd1.c]",
               "Improper value of nth,ntr,dtr_log,dtr_ms or dtr_cnfg");
  }

  mpc_bcast_i(&nth, 1);
  mpc_bcast_i(&ntr, 1);
  mpc_bcast_i(&dtr_log, 1);
  mpc_bcast_i(&dtr_ms, 1);
  mpc_bcast_i(&dtr_cnfg, 1);

  if (my_rank == 0) {
    if (append) {
      ir = fread(istd, sizeof(stdint_t), 3, fdat);
      error_root(ir != 3, 1, "read_schedule [qcd1.c]", "Incorrect read count");

      /* Skip compatability check on a loose append */
      if (loose_append) {
        return;
      }

      if (endian == openqcd_utils__BIG_ENDIAN) {
        bswap_int(3, istd);
      }

      ie = 0;
      ie |= (istd[0] != (stdint_t)(dtr_log));
      ie |= (istd[1] != (stdint_t)(dtr_ms));
      ie |= (istd[2] != (stdint_t)(dtr_cnfg));

      error_root(ie != 0, 1, "read_schedule [qcd1.c]",
                 "Parameters do not match previous run");
    } else {
      istd[0] = (stdint_t)(dtr_log);
      istd[1] = (stdint_t)(dtr_ms);
      istd[2] = (stdint_t)(dtr_cnfg);

      if (endian == openqcd_utils__BIG_ENDIAN) {
        bswap_int(3, istd);
      }

      iw = fwrite(istd, sizeof(stdint_t), 3, fdat);
      error_root(iw != 3, 1, "read_schedule [qcd1.c]", "Incorrect write count");
    }
  }
}

static void read_actions(void)
{
  int i, k, l, nact, *iact;
  int npf, nlv, nmu;
  double tau, *mu;
  action_parms_t ap;
  rat_parms_t rp;

  if (my_rank == 0) {
    find_section("HMC parameters");
    nact = count_tokens("actions");
    read_line("npf", "%d", &npf);
    read_line("nlv", "%d", &nlv);
    read_line("tau", "%lf", &tau);
  }

  mpc_bcast_i(&nact, 1);
  mpc_bcast_i(&npf, 1);
  mpc_bcast_i(&nlv, 1);
  mpc_bcast_d(&tau, 1);

  if (nact > 0) {
    iact = malloc(nact * sizeof(*iact));
    error(iact == NULL, 1, "read_actions [qcd1.c]",
          "Unable to allocate temporary array");
    if (my_rank == 0) {
      read_iprms("actions", nact, iact);
    }
    mpc_bcast_i(iact, nact);
  } else {
    iact = NULL;
  }

  nmu = 0;

  for (i = 0; i < nact; i++) {
    k = iact[i];
    ap = action_parms(k);

    if (ap.action == ACTIONS) {
      read_action_parms(k);
    }

    ap = action_parms(k);

    if ((ap.action == ACF_RAT) || (ap.action == ACF_RAT_SDET)) {
      l = ap.irat[0];
      rp = rat_parms(l);

      if (rp.degree == 0) {
        read_rat_parms(l);
      }
    } else if ((nmu == 0) &&
               ((ap.action == ACF_TM1) || (ap.action == ACF_TM1_EO) ||
                (ap.action == ACF_TM1_EO_SDET) || (ap.action == ACF_TM2) ||
                (ap.action == ACF_TM2_EO))) {
      if (my_rank == 0) {
        find_section("HMC parameters");
        nmu = count_tokens("mu");
      }

      mpc_bcast_i(&nmu, 1);
    }
  }

  if (nmu > 0) {
    mu = malloc(nmu * sizeof(*mu));
    error(mu == NULL, 1, "read_actions [qcd1.c]",
          "Unable to allocate temporary array");

    if (my_rank == 0) {
      find_section("HMC parameters");
      read_dprms("mu", nmu, mu);
    }

    mpc_bcast_d(mu, nmu);
  } else {
    mu = NULL;
  }

  hmc = set_hmc_parms(nact, iact, npf, nmu, mu, nlv, tau);

  if (nact > 0) {
    free(iact);
  }
  if (nmu > 0) {
    free(mu);
  }

  if (append) {
    check_hmc_parms(fdat);

    if (loose_append) {
      leniently_check_action_parms(fdat);
    } else {
      check_action_parms(fdat);
    }

  } else {
    write_hmc_parms(fdat);
    write_action_parms(fdat);
  }
}

static void read_smearing(void)
{
  long section_pos;
  int has_smearing = 0;
  int n_smear, smear_gauge, smear_fermion;
  double rho_t, rho_s;

  if (my_rank == 0) {
    section_pos = find_optional_section("Smearing parameters");

    if (section_pos == No_Section_Found) {
      has_smearing = 0;
    } else {
      has_smearing = 1;
      read_line("n_smear", "%d", &n_smear);
      read_line("rho_t", "%lf", &rho_t);
      read_line("rho_s", "%lf", &rho_s);
      read_line("gauge", "%d", &smear_gauge);
      read_line("fermion", "%d", &smear_fermion);
    }
  }

  mpc_bcast_i(&has_smearing, 1);

  if (has_smearing == 1) {
    mpc_bcast_i(&n_smear, 1);
    mpc_bcast_d(&rho_t, 1);
    mpc_bcast_d(&rho_s, 1);
    mpc_bcast_i(&smear_gauge, 1);
    mpc_bcast_i(&smear_fermion, 1);

    set_stout_smearing_parms(n_smear, rho_t, rho_s, smear_gauge, smear_fermion);
  } else {
    set_no_stout_smearing_parms();
  }

  if (append) {
    check_stout_smearing_parms(fdat);
  } else {
    write_stout_smearing_parms(fdat);
  }
}

static void read_integrator(void)
{
  int nlv, i, j, k, l;
  mdint_parms_t mdp;
  force_parms_t fp;
  rat_parms_t rp;

  nlv = hmc.nlv;

  for (i = 0; i < nlv; i++) {
    read_mdint_parms(i);
    mdp = mdint_parms(i);

    for (j = 0; j < mdp.nfr; j++) {
      k = mdp.ifr[j];
      fp = force_parms(k);

      if (fp.force == FORCES) {
        read_force_parms2(k);
      }

      fp = force_parms(k);

      if ((fp.force == FRF_RAT) || (fp.force == FRF_RAT_SDET)) {
        l = fp.irat[0];
        rp = rat_parms(l);

        if (rp.degree == 0) {
          read_rat_parms(l);
        }
      }
    }
  }

  if (append) {
    check_rat_parms(fdat);
    check_mdint_parms(fdat);

    if (loose_append) {
      leniently_check_force_parms(fdat);
    } else {
      check_force_parms(fdat);
    }
  } else {
    write_rat_parms(fdat);
    write_mdint_parms(fdat);
    write_force_parms(fdat);
  }
}

static void read_sap_parms(void)
{
  int bs[4];

  if (my_rank == 0) {
    find_section("SAP");
    read_line("bs", "%d %d %d %d", bs, bs + 1, bs + 2, bs + 3);
  }

  mpc_bcast_i(bs, 4);
  set_sap_parms(bs, 1, 4, 5);

  if (loose_append) {
    leniently_check_sap_parms(fdat);
  } else if (append) {
    check_sap_parms(fdat);
  } else {
    write_sap_parms(fdat);
  }
}

static void read_dfl_parms(void)
{
  int bs[4], Ns;
  int ninv, nmr, ncy, nkv, nmx, nsm;
  double kappa, mu, res, dtau;

  if (my_rank == 0) {
    find_section("Deflation subspace");
    read_line("bs", "%d %d %d %d", bs, bs + 1, bs + 2, bs + 3);
    read_line("Ns", "%d", &Ns);
  }

  mpc_bcast_i(bs, 4);
  mpc_bcast_i(&Ns, 1);
  set_dfl_parms(bs, Ns);

  if (my_rank == 0) {
    find_section("Deflation subspace generation");
    read_line("kappa", "%lf", &kappa);
    read_line("mu", "%lf", &mu);
    read_line("ninv", "%d", &ninv);
    read_line("nmr", "%d", &nmr);
    read_line("ncy", "%d", &ncy);
  }

  mpc_bcast_d(&kappa, 1);
  mpc_bcast_d(&mu, 1);
  mpc_bcast_i(&ninv, 1);
  mpc_bcast_i(&nmr, 1);
  mpc_bcast_i(&ncy, 1);
  set_dfl_gen_parms(kappa, mu, ninv, nmr, ncy);

  if (my_rank == 0) {
    find_section("Deflation projection");
    read_line("nkv", "%d", &nkv);
    read_line("nmx", "%d", &nmx);
    read_line("res", "%lf", &res);
  }

  mpc_bcast_i(&nkv, 1);
  mpc_bcast_i(&nmx, 1);
  mpc_bcast_d(&res, 1);
  set_dfl_pro_parms(nkv, nmx, res);

  if (my_rank == 0) {
    find_section("Deflation update scheme");
    read_line("dtau", "%lf", &dtau);
    read_line("nsm", "%d", &nsm);
  }

  mpc_bcast_d(&dtau, 1);
  mpc_bcast_i(&nsm, 1);
  set_dfl_upd_parms(dtau, nsm);

  if (loose_append) {
    leniently_check_dfl_parms(fdat);
  } else if (append) {
    check_dfl_parms(fdat);
  } else {
    write_dfl_parms(fdat);
  }
}

static void read_solvers(void)
{
  int nact, *iact, nlv;
  int nfr, *ifr, i, j, k;
  int nsp, isap, idfl;
  mdint_parms_t mdp;
  action_parms_t ap;
  force_parms_t fp;
  solver_parms_t sp;

  nact = hmc.nact;
  iact = hmc.iact;
  nlv = hmc.nlv;

  isap = 0;
  idfl = 0;

  for (i = 0; i < nact; i++) {
    ap = action_parms(iact[i]);

    if ((ap.action == ACF_TM1) || (ap.action == ACF_TM1_EO) ||
        (ap.action == ACF_TM1_EO_SDET) || (ap.action == ACF_TM2) ||
        (ap.action == ACF_TM2_EO) || (ap.action == ACF_RAT) ||
        (ap.action == ACF_RAT_SDET)) {
      if ((ap.action == ACF_TM2) || (ap.action == ACF_TM2_EO)) {
        nsp = 2;
      } else {
        nsp = 1;
      }

      for (k = 0; k < nsp; k++) {
        j = ap.isp[k];
        sp = solver_parms(j);

        if (sp.solver == SOLVERS) {
          read_solver_parms(j);
          sp = solver_parms(j);

          if (sp.solver == SAP_GCR) {
            isap = 1;
          } else if (sp.solver == DFL_SAP_GCR) {
            isap = 1;
            idfl = 1;
          }
        }
      }
    }
  }

  for (i = 0; i < nlv; i++) {
    mdp = mdint_parms(i);
    nfr = mdp.nfr;
    ifr = mdp.ifr;

    for (j = 0; j < nfr; j++) {
      fp = force_parms(ifr[j]);

      if ((fp.force == FRF_TM1) || (fp.force == FRF_TM1_EO) ||
          (fp.force == FRF_TM1_EO_SDET) || (fp.force == FRF_TM2) ||
          (fp.force == FRF_TM2_EO) || (fp.force == FRF_RAT) ||
          (fp.force == FRF_RAT_SDET)) {
        k = fp.isp[0];
        sp = solver_parms(k);

        if (sp.solver == SOLVERS) {
          read_solver_parms(k);
          sp = solver_parms(k);

          if (sp.solver == SAP_GCR) {
            isap = 1;
          } else if (sp.solver == DFL_SAP_GCR) {
            isap = 1;
            idfl = 1;
          }
        }
      }
    }
  }

  if (loose_append) {
    leniently_check_solver_parms(fdat);
  } else if (append) {
    check_solver_parms(fdat);
  } else {
    write_solver_parms(fdat);
  }

  if (isap) {
    read_sap_parms();
  }

  if (idfl) {
    read_dfl_parms();
  }
}

static void read_wflow_parms(void)
{
  int nstep, dnms, ie, ir, iw;
  stdint_t istd[3];
  double eps, dstd[1];

  if (my_rank == 0) {
    if (append) {
      ir = fread(istd, sizeof(stdint_t), 1, fdat);
      error_root(ir != 1, 1, "read_wflow_parms [qcd1.c]",
                 "Incorrect read count");

      if (endian == openqcd_utils__BIG_ENDIAN) {
        bswap_int(1, istd);
      }

      error_root(istd[0] != (stdint_t)(noms == 0), 1,
                 "read_wflow_parms [qcd1.c]",
                 "Attempt to mix measurement with other runs");
    } else {
      istd[0] = (stdint_t)(noms == 0);

      if (endian == openqcd_utils__BIG_ENDIAN) {
        bswap_int(1, istd);
      }

      iw = fwrite(istd, sizeof(stdint_t), 1, fdat);
      error_root(iw != 1, 1, "read_wflow_parms [qcd1.c]",
                 "Incorrect write count");
    }

    if (noms == 0) {
      find_section("Wilson flow");
      read_line("integrator", "%s", line);
      read_line("eps", "%lf", &eps);
      read_line("nstep", "%d", &nstep);
      read_line("dnms", "%d", &dnms);

      if (strcmp(line, "EULER") == 0) {
        flint = 0;
      } else if (strcmp(line, "RK2") == 0) {
        flint = 1;
      } else if (strcmp(line, "RK3") == 0) {
        flint = 2;
      } else {
        error_root(1, 1, "read_wflow_parms [qcd1.c]", "Unknown integrator");
      }

      error_root((dnms < 1) || (nstep < dnms) || ((nstep % dnms) != 0), 1,
                 "read_wflow_parms [qcd1.c]",
                 "nstep must be a multiple of dnms");
    } else {
      flint = 0;
      eps = 0.0;
      nstep = 1;
      dnms = 1;
    }
  }

  mpc_bcast_i(&flint, 1);
  mpc_bcast_d(&eps, 1);
  mpc_bcast_i(&nstep, 1);
  mpc_bcast_i(&dnms, 1);

  file_head.dn = dnms;
  file_head.nn = nstep / dnms;
  file_head.tmax = N0;
  file_head.eps = eps;

  if ((my_rank == 0) && (noms == 0)) {
    if (append) {
      ir = fread(istd, sizeof(stdint_t), 3, fdat);
      ir += fread(dstd, sizeof(double), 1, fdat);
      error_root(ir != 4, 1, "read_wflow_parms [qcd1.c]",
                 "Incorrect read count");

      if (endian == openqcd_utils__BIG_ENDIAN) {
        bswap_int(3, istd);
        bswap_double(1, dstd);
      }

      ie = 0;
      ie |= (istd[0] != (stdint_t)(flint));
      ie |= (istd[1] != (stdint_t)(nstep));
      ie |= (istd[2] != (stdint_t)(dnms));
      ie |= (dstd[0] != eps);

      error_root(ie != 0, 1, "read_wflow_parms [qcd1.c]",
                 "Parameters do not match previous run");
    } else {
      istd[0] = (stdint_t)(flint);
      istd[1] = (stdint_t)(nstep);
      istd[2] = (stdint_t)(dnms);
      dstd[0] = eps;

      if (endian == openqcd_utils__BIG_ENDIAN) {
        bswap_int(3, istd);
        bswap_double(1, dstd);
      }

      iw = fwrite(istd, sizeof(stdint_t), 3, fdat);
      iw += fwrite(dstd, sizeof(double), 1, fdat);
      error_root(iw != 4, 1, "read_wflow_parms [qcd1.c]",
                 "Incorrect write count");
    }
  }
}

static void read_ani_parms(void)
{
  int has_ani, has_tts;
  long section_pos;
  double nu, xi, cR, cT, us_gauge, ut_gauge, us_fermion, ut_fermion;

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
      read_optional_line("us_fermion", "%lf", &us_fermion, 1.0);
      read_optional_line("ut_fermion", "%lf", &ut_fermion, 1.0);
    }
  }

  mpc_bcast_i(&has_ani, 1);

  if (has_ani == 1) {
    mpc_bcast_i(&has_tts, 1);
    mpc_bcast_d(&nu, 1);
    mpc_bcast_d(&xi, 1);
    mpc_bcast_d(&cR, 1);
    mpc_bcast_d(&cT, 1);
    mpc_bcast_d(&us_gauge, 1);
    mpc_bcast_d(&ut_gauge, 1);
    mpc_bcast_d(&us_fermion, 1);
    mpc_bcast_d(&ut_fermion, 1);

    set_ani_parms(has_tts, nu, xi, cR, cT, us_gauge, ut_gauge, us_fermion,
                  ut_fermion);
  } else {
    set_no_ani_parms();
  }

  if (append) {
    check_ani_parms(fdat);
  } else {
    write_ani_parms(fdat);
  }
}

static void read_infile(int argc, char *argv[])
{
  int ifile, iseed, new_rng;

  if (my_rank == 0) {
    flog = freopen("STARTUP_ERROR", "w", stdout);

    ifile = find_opt(argc, argv, "-i");
    noloc = find_opt(argc, argv, "-noloc");
    noexp = find_opt(argc, argv, "-noexp");
    rmold = find_opt(argc, argv, "-rmold");
    noms = find_opt(argc, argv, "-noms");
    scnfg = find_opt(argc, argv, "-c");
    append = find_opt(argc, argv, "-a");
    loose_append = find_opt(argc, argv, "-A");
    norng = find_opt(argc, argv, "-norng");
    iseed = find_opt(argc, argv, "-seed");
    endian = endianness();

    append = (append != 0) || (loose_append != 0);
    new_rng = (iseed != 0) || (norng != 0);

    error_root(
        (ifile == 0) || (ifile == (argc - 1)) ||
            ((append != 0) && (scnfg == 0)) || (iseed == (argc - 1)) ||
            ((loose_append != 0) && (new_rng == 0)),
        1, "read_infile [qcd1.c]",
        "Syntax: qcd1 -i <filename> [-noloc] [-noexp] "
        "[-rmold] [-noms] [-c [<filename>] [-norng] [-(a/A)]] [-seed <seed>]");

    error_root(endian == openqcd_utils__UNKNOWN_ENDIAN, 1,
               "read_infile [qcd1.c]", "Machine has unknown endianness");

    error_root((noexp) && (noloc), 1, "read_infile [qcd1.c]",
               "The concurrent use of -noloc and -noexp is not permitted");

    if ((scnfg != 0) && (scnfg < (argc - 1))) {
      strncpy(cnfg, argv[scnfg + 1], NAME_SIZE - 1);
      cnfg[NAME_SIZE - 1] = '\0';

      if (cnfg[0] == '-') {
        cnfg[0] = '\0';
      }
    } else {
      cnfg[0] = '\0';
    }

    if (iseed) {
      cmd_seed = (int)strtol(argv[iseed + 1], NULL, 10);
    }

    fin = freopen(argv[ifile + 1], "r", stdin);
    error_root(fin == NULL, 1, "read_infile [qcd1.c]",
               "Unable to open input file");
  }

  mpc_bcast_i(&noloc, 1);
  mpc_bcast_i(&noexp, 1);
  mpc_bcast_i(&rmold, 1);
  mpc_bcast_i(&noms, 1);
  mpc_bcast_i(&scnfg, 1);
  mpc_bcast_i(&append, 1);
  mpc_bcast_i(&loose_append, 1);
  mpc_bcast_i(&norng, 1);
  mpc_bcast_i(&endian, 1);
  mpc_bcast_i(&cmd_seed, 1);
  mpc_bcast_c(cnfg, NAME_SIZE);

  if (my_rank == 0) {
    find_section("Random number generator");
    read_line("level", "%d", &level);
    read_line("seed", "%d", &seed);
  }

  mpc_bcast_i(&level, 1);
  mpc_bcast_i(&seed, 1);

  if (cmd_seed >= 0) {
    seed = cmd_seed;
  }

  read_dirs();
  setup_files();

  if (my_rank == 0) {
    if (append) {
      fdat = fopen(par_file, "rb");
    } else {
      fdat = fopen(par_file, "wb");
    }

    error_root(fdat == NULL, 1, "read_infile [qcd1.c]",
               "Unable to open parameter file");
  }

#if !defined(STATIC_SIZES)
  read_lattize_sizes();
#endif

  read_ani_parms();
  read_smearing();
  read_lat_parms();
  read_bc_parms();
  read_schedule();
  read_actions();
  read_integrator();
  read_solvers();
  read_wflow_parms();

  if (my_rank == 0) {
    fclose(fin);
    fclose(fdat);

    if (append == 0) {
      copy_file(par_file, par_save);
    }
  }
}

static void check_old_log(int ic, int *nl, int *icnfg, int new_rng)
{
  int ir, isv;
  int np[4], bp[4];

  fend = fopen(log_file, "r");
  error_root(fend == NULL, 1, "check_old_log [qcd1.c]",
             "Unable to open log file");
  (*nl) = 0;
  (*icnfg) = 0;
  ir = 1;
  isv = 0;

  while (fgets(line, NAME_SIZE, fend) != NULL) {
    if (strstr(line, "process grid") != NULL) {
      ir &= (sscanf(line, "%dx%dx%dx%d process grid, %dx%dx%dx%d", np, np + 1,
                    np + 2, np + 3, bp, bp + 1, bp + 2, bp + 3) == 8);

      ipgrd[0] = ((np[0] != NPROC0) || (np[1] != NPROC1) || (np[2] != NPROC2) ||
                  (np[3] != NPROC3));
      ipgrd[1] = ((bp[0] != NPROC0_BLK) || (bp[1] != NPROC1_BLK) ||
                  (bp[2] != NPROC2_BLK) || (bp[3] != NPROC3_BLK));
    } else if (strstr(line, "Trajectory no") != NULL) {
      ir &= (sscanf(line, "Trajectory no %d", nl) == 1);
      isv = 0;
    } else if (strstr(line, "Configuration no") != NULL) {
      ir &= (sscanf(line, "Configuration no %d", icnfg) == 1);
      isv = 1;
    }
  }

  fclose(fend);

  error_root(ir != 1, 1, "check_old_log [qcd1.c]", "Incorrect read count");

  if (new_rng == 0) {
    error_root((ic >= 0) && (ic != (*icnfg)), 1, "check_old_log [qcd1.c]",
               "Continuation run:\n"
               "Initial configuration is not the last one of the previous run");

    error_root(isv == 0, 1, "check_old_log [qcd1.c]",
               "Continuation run:\n"
               "The log file extends beyond the last configuration save");
  } else if (ic >= 0) {
    error_root(ic > (*icnfg), 1, "check_old_log [qcd1.c]",
               "Continuing from a config later than the last one computed");
  }
}

static void check_old_dat(int nl)
{
  int nt;
  dat_t ndat;

  fdat = fopen(dat_file, "rb");
  error_root(fdat == NULL, 1, "check_old_dat [qcd1.c]",
             "Unable to open data file");
  nt = 0;

  while (read_dat(1, &ndat) == 1) {
    nt = ndat.nt;
  }

  fclose(fdat);

  error_root(nt != nl, 1, "check_old_dat [qcd1.c]",
             "Continuation run: Incomplete or too many data records");
}

static void check_old_msdat(int nl)
{
  int ic, ir, nt, pnt, dnt;

  fdat = fopen(msdat_file, "rb");
  error_root(fdat == NULL, 1, "check_old_msdat [qcd1.c]",
             "Unable to open data file");

  check_file_head();

  nt = 0;
  dnt = 0;
  pnt = 0;

  for (ic = 0;; ic++) {
    ir = read_data();

    if (ir == 0) {
      error_root(ic == 0, 1, "check_old_msdat [qcd1.c]",
                 "No data records found");
      break;
    }

    nt = data.nt;

    if (ic == 1) {
      dnt = nt - pnt;
      error_root(dnt < 1, 1, "check_old_msdat [qcd1.c]",
                 "Incorrect trajectory separation");
    } else if (ic > 1) {
      error_root(nt != (pnt + dnt), 1, "check_old_msdat [qcd1.c]",
                 "Trajectory sequence is not equally spaced");
    }

    pnt = nt;
  }

  fclose(fdat);

  error_root((nt != nl) || ((ic > 1) && (dnt != dtr_ms)), 1,
             "check_old_msdat [qcd1.c]",
             "Last trajectory numbers "
             "or the trajectory separations do not match");
}

static void check_files(int *nl, int *icnfg)
{
  int icmax, ic;

  ipgrd[0] = 0;
  ipgrd[1] = 0;

  if (my_rank == 0) {
    if (noloc) {
      error_root(cnfg[strlen(cnfg) - 1] == '*', 1, "check_files [qcd1.c]",
                 "Attempt to read an "
                 "imported configuration when -noloc is set");
    }

    if (append) {

      ic = -1;
      if (cnfg[0] != '\0') {
        error_root(strstr(cnfg, nbase) != cnfg, 1, "check_files [qcd1.c]",
                   "Continuation run:\n"
                   "Run name does not match the previous one");
        error_root(sscanf(cnfg + strlen(nbase), "n%d", &ic) != 1, 1,
                   "check_files [qcd1.c]",
                   "Continuation run:\n"
                   "Unable to read configuration number from file name");
      }

      check_old_log(ic, nl, icnfg, (norng != 0) || (cmd_seed >= 0));

      if (ic == -1) {
        ic = *icnfg;
        sprintf(cnfg, "%sn%d", nbase, ic);
      }

      /* Change the trajectory number to be inline with the start cfg */
      if (ic != *icnfg) {
        *nl = ic * dtr_cnfg;
      }

      if ((loose_append == 0) && (ic == *icnfg)) {
        check_old_dat(*nl);
      }

      if ((noms == 0) && (loose_append == 0) && (ic == *icnfg)) {
        check_old_msdat(*nl);
      }

      (*icnfg) = ic + 1;
    } else {
      fin = fopen(log_file, "r");
      fdat = fopen(dat_file, "rb");

      if (noms == 0) {
        fend = fopen(msdat_file, "rb");
      } else {
        fend = NULL;
      }

      error_root((fin != NULL) || (fdat != NULL) || (fend != NULL), 1,
                 "check_files [qcd1.c]",
                 "Attempt to overwrite old *.log or *.dat file");

      if (noms == 0) {
        fdat = fopen(msdat_file, "wb");
        error_root(fdat == NULL, 1, "check_files [qcd1.c]",
                   "Unable to open measurement data file");
        write_file_head();
        fclose(fdat);
      }

      (*nl) = 0;
      (*icnfg) = 1;
    }

    icmax = (*icnfg) + (ntr - nth) / dtr_cnfg;

    if (noloc == 0) {
      error_root(name_size("%s/%sn%d_%d", loc_dir, nbase, icmax, NPROC - 1) >=
                     NAME_SIZE,
                 1, "check_files [qcd1.c]", "loc_dir name is too long");
    }

    if (noexp == 0) {
      error_root(name_size("%s/%sn%d", cnfg_dir, nbase, icmax) >= NAME_SIZE, 1,
                 "check_files [qcd1.c]", "cnfg_dir name is too long");
    }

    if (scnfg) {
      if (cnfg[strlen(cnfg) - 1] == '*') {
        error_root(name_size("%s/%s%d", loc_dir, cnfg, NPROC - 1) >= NAME_SIZE,
                   1, "check_files [qcd1.c]", "loc_dir name is too long");
      } else {
        error_root(name_size("%s/%s", cnfg_dir, cnfg) >= NAME_SIZE, 1,
                   "check_files [qcd1.c]", "cnfg_dir name is too long");
      }
    }
  }

  mpc_bcast_i(nl, 1);
  mpc_bcast_i(icnfg, 1);
}

static void init_ud(void)
{
  char *p;

  if (scnfg) {
    if (cnfg[strlen(cnfg) - 1] != '*') {
      sprintf(cnfg_file, "%s/%s", cnfg_dir, cnfg);
      import_cnfg(cnfg_file);
    } else {
      sprintf(line, "%s/%s", loc_dir, cnfg);
      p = line + strlen(line) - 1;
      p[0] = '\0';
      sprintf(cnfg_file, "%s_%d", line, my_rank);
      read_cnfg(cnfg_file);
    }
  } else {
    random_ud();
  }
}

#ifndef SITERANDOM
static void init_rng(int icnfg)
{
  int ic;

  if ((cmd_seed >= 0) || (append == 0)) {
    start_ranlux(level, seed);
  } else {
    if (cnfg[strlen(cnfg) - 1] != '*') {
      if (norng) {
        start_ranlux(level, seed ^ (icnfg - 1));
      } else {
        ic = import_ranlux(rng_file);
        error_root(ic != (icnfg - 1), 1, "init_rng [qcd1.c]",
                   "Configuration number mismatch (*.rng file)");
      }
    }
  }
}
#else
static void init_rng(int icnfg)
{
  start_ranlux_site(level, seed);
}
#endif

#ifdef dirac_counters
static void init_dirac_counters(void)
{
  Dw_dble_counter = 0;
  Dwee_dble_counter = 0;
  Dwoo_dble_counter = 0;
  Dwoe_dble_counter = 0;
  Dweo_dble_counter = 0;
  Dwhat_dble_counter = 0;
  Dw_blk_dble_counter = 0;
  Dwee_blk_dble_counter = 0;
  Dwoo_blk_dble_counter = 0;
  Dwoe_blk_dble_counter = 0;
  Dweo_blk_dble_counter = 0;
  Dwhat_blk_dble_counter = 0;

  Dw_counter = 0;
  Dwee_counter = 0;
  Dwoo_counter = 0;
  Dwoe_counter = 0;
  Dweo_counter = 0;
  Dwhat_counter = 0;
  Dw_blk_counter = 0;
  Dwee_blk_counter = 0;
  Dwoo_blk_counter = 0;
  Dwoe_blk_counter = 0;
  Dweo_blk_counter = 0;
  Dwhat_blk_counter = 0;
}
#endif

static void store_ud(su3_dble *usv)
{
  su3_dble *udb;

  udb = udfld();
  cm3x3_assign(4 * VOLUME, udb, usv);
}

static void recall_ud(su3_dble *usv)
{
  su3_dble *udb;

  udb = udfld();
  cm3x3_assign(4 * VOLUME, usv, udb);
  set_flags(UPDATED_UD);
}

static void set_data(int nt)
{
  int in, dn, nn;
  double eps;

  data.nt = nt;
  dn = file_head.dn;
  nn = file_head.nn;
  eps = file_head.eps;

  for (in = 0; in < nn; in++) {
    Wact[in] = plaq_action_slices(data.Wsl[in]);
    Yact[in] = ym_action_slices(data.Ysl[in]);
    Qtop[in] = tcharge_slices(data.Qsl[in]);

    if (flint == 0) {
      fwd_euler(dn, eps);
    } else if (flint == 1) {
      fwd_rk2(dn, eps);
    } else {
      fwd_rk3(dn, eps);
    }
  }

  Wact[in] = plaq_action_slices(data.Wsl[in]);
  Yact[in] = ym_action_slices(data.Ysl[in]);
  Qtop[in] = tcharge_slices(data.Qsl[in]);
}

static void print_info(int icnfg)
{
  int isap, idfl, n;
  long ip;

  if (my_rank == 0) {
    ip = ftell(flog);
    fclose(flog);

    if (ip == 0L) {
      remove("STARTUP_ERROR");
    }

    if (append) {
      flog = freopen(log_file, "a", stdout);
    } else {
      flog = freopen(log_file, "w", stdout);
    }

    error_root(flog == NULL, 1, "print_info [qcd1.c]",
               "Unable to open log file");

    if (append) {
      printf("Continuation run (%s), start from configuration %s\n\n",
             loose_append ? "-A" : "-a", cnfg);
    } else {
      printf("\n");
      printf("Simulation of QCD with Wilson quarks\n");
      printf("------------------------------------\n\n");

      if (scnfg) {
        printf("New run, start from configuration %s\n\n", cnfg);
      } else {
        printf("New run, start from random configuration\n\n");
      }

      printf("Using the HMC algorithm\n");
      printf("Program major version: %s\n", openQCD_RELEASE);
      printf("Program build date: %s\n", build_date);
      printf("Program git SHA: %s\n", build_git_sha);
      printf("Program user CFLAGS: %s\n", build_user_cflags);
    }

    if (endian == openqcd_utils__LITTLE_ENDIAN) {
      printf("The machine is little endian\n");
    } else {
      printf("The machine is big endian\n");
    }
    if (noloc) {
      printf("The local disks are not used\n");
    }
    if (noexp) {
      printf("The generated configurations are not exported\n");
    }
    if (rmold) {
      printf("Old configurations are deleted\n");
    }
    printf("\n");

    if ((ipgrd[0] != 0) && (ipgrd[1] != 0)) {
      printf("Process grid and process block size changed:\n");
    } else if (ipgrd[0] != 0) {
      printf("Process grid changed:\n");
    } else if (ipgrd[1] != 0) {
      printf("Process block size changed:\n");
    }

    if ((append == 0) || (ipgrd[0] != 0) || (ipgrd[1] != 0)) {
      printf("%dx%dx%dx%d lattice, ", N0, N1, N2, N3);
      printf("%dx%dx%dx%d local lattice\n", L0, L1, L2, L3);
      printf("%dx%dx%dx%d process grid, ", NPROC0, NPROC1, NPROC2, NPROC3);
      printf("%dx%dx%dx%d process block size\n\n", NPROC0_BLK, NPROC1_BLK,
             NPROC2_BLK, NPROC3_BLK);
    }

    if (append == 0) {
      print_lat_parms();
      print_ani_parms();
      print_stout_smearing_parms();
      print_bc_parms(3);
    }

    printf("Random number generator:\n");

    if (cmd_seed >= 0) {
      printf("Using seed from command line\n");
      printf("level = %d, seed = %d\n\n", level, seed);
    } else if (append) {
      if (cnfg[strlen(cnfg) - 1] != '*') {
        if (norng) {
          printf("level = %d, seed = %d, effective seed = %d\n\n", level, seed,
                 seed ^ (icnfg - 1));
        } else {
          printf("State of ranlxs and ranlxd reset to the\n");
          printf("last exported state\n\n");
        }
      } else {
        printf("State of ranlxs and ranlxd read from\n");
        printf("initial field-configuration file\n\n");
      }
    } else {
      printf("level = %d, seed = %d\n\n", level, seed);
    }

    if (append) {
      printf("Trajectories:\n");
      printf("ntr = %d\n\n", ntr);
    } else {
      print_hmc_parms();

      printf("Trajectories:\n");
      printf("nth = %d, ntr = %d\n", nth, ntr);

      if (noms) {
        printf("dtr_log = %d, dtr_cnfg = %d\n\n", dtr_log, dtr_cnfg);
        printf("Wilson flow observables are not measured\n\n");
      } else {
        printf("dtr_log = %d, dtr_ms = %d, dtr_cnfg = %d\n\n", dtr_log, dtr_ms,
               dtr_cnfg);
        printf("Online measurement of Wilson flow observables\n\n");
      }
    }

    if (append == 0 || loose_append) {
      print_action_parms();
    }

    if (append == 0) {
      print_rat_parms();
      print_mdint_parms();
      print_force_parms2();
    }

    if (append == 0 || loose_append) {
      print_solver_parms(&isap, &idfl);

      if (isap) {
        print_sap_parms(0);
      }

      if (idfl) {
        print_dfl_parms(1);
      }
    }

    if ((append == 0) && (noms == 0)) {
      printf("Wilson flow:\n");
      if (flint == 0) {
        printf("Euler integrator\n");
      } else if (flint == 1) {
        printf("2nd order RK integrator\n");
      } else {
        printf("3rd order RK integrator\n");
      }
      n = fdigits(file_head.eps);
      printf("eps = %.*f\n", IMAX(n, 1), file_head.eps);
      printf("nstep = %d\n", file_head.dn * file_head.nn);
      printf("dnms = %d\n\n", file_head.dn);
    }

    fflush(flog);
  }
}

static void print_log(dat_t *ndat)
{
  double average_link, average_stout_link;
  stout_smearing_params_t smear_params;

  if (my_rank == 0) {
    printf("Trajectory no %d\n", (*ndat).nt);
    printf("dH = %+.2e, ", (*ndat).dH);
    printf("iac = %d\n", (*ndat).iac);

    smear_params = stout_smearing_parms();

    if (smear_params.num_smear > 1) {
      printf("Average plaquette (thin)  = %.15f\n", (*ndat).avpl);
      printf("Average plaquette (stout) = %.15f\n", (*ndat).avpl_smeared);
    } else {
      printf("Average plaquette = %.15f\n", (*ndat).avpl);
    }

    if (bc_type() == 3) {
      if ((smear_params.num_smear > 1) &&
          not_equal_d(smear_params.rho_temporal, 0.0)) {
        printf("Average temporal link (thin)  = %.15f\n",
               (*ndat).average_t_link);
        printf("Average temporal link (stout) = %.15f\n",
               (*ndat).average_stout_t_link);
      } else {
        printf("Average temporal link = %.15f\n", (*ndat).average_t_link);
      }

      average_link = (3 * ndat->average_s_link + ndat->average_t_link) / 4;

      if (smear_params.num_smear > 1) {
        average_stout_link =
            (3 * ndat->average_stout_s_link + ndat->average_stout_t_link) / 4;

        printf("Average spatial link (thin)  = %.15f\n",
               (*ndat).average_s_link);
        printf("Average spatial link (stout) = %.15f\n",
               (*ndat).average_stout_s_link);
        printf("Average link (thin)  = %.15f\n", average_link);
        printf("Average link (stout) = %.15f\n", average_stout_link);
      } else {
        printf("Average spatial link = %.15f\n", (*ndat).average_s_link);
        printf("Average link = %.15f\n", average_link);
      }

      printf("Polyakov loop = %.15f %.15f\n", (*ndat).ploop.re,
             (*ndat).ploop.im);

      if (stout_smearing_parms().num_smear > 0) {
        printf("Polyakov loop (stout) = %.15f %.15f\n",
               (*ndat).smeared_ploop.re, (*ndat).smeared_ploop.im);
      }
    }

    print_all_avgstat();
  }
}

static void save_dat(int n, double siac, double wtcyc, double wtall,
                     dat_t *ndat)
{
  int iw;

  if (my_rank == 0) {
    fdat = fopen(dat_file, "ab");
    error_root(fdat == NULL, 1, "save_dat [qcd1.c]",
               "Unable to open data file");

    iw = write_dat(1, ndat);
    error_root(iw != 1, 1, "save_dat [qcd1.c]", "Incorrect write count");
    fclose(fdat);

    printf("Acceptance rate = %1.2f\n", siac / (double)(n + 1));
    printf("Time per trajectory = %.2e sec (average = %.2e sec)\n\n",
           wtcyc / (double)(dtr_log), wtall / (double)(n + 1));
    fflush(flog);
  }
}

static void save_msdat(int n, double wtms, double wtmsall)
{
  int nms, in, dn, nn, din;
  double eps;

  if (my_rank == 0) {
    fdat = fopen(msdat_file, "ab");
    error_root(fdat == NULL, 1, "save_msdat [qcd1.c]",
               "Unable to open data file");
    write_data();
    fclose(fdat);

    nms = (n + 1 - nth) / dtr_ms + (nth > 0);
    dn = file_head.dn;
    nn = file_head.nn;
    eps = file_head.eps;

    din = nn / 10;
    if (din < 1) {
      din = 1;
    }

    printf("Measurement run:\n\n");

    for (in = 0; in <= nn; in += din) {
      printf("n = %3d, t = %.2e, Wact = %.6e, Yact = %.6e, Q = % .2e\n",
             in * dn, eps * (double)(in * dn), Wact[in], Yact[in], Qtop[in]);
    }

    printf("\n");
    printf("Configuration fully processed in %.2e sec ", wtms);
    printf("(average = %.2e sec)\n", wtmsall / (double)(nms));
    printf("Measured data saved\n\n");
    fflush(flog);
  }
}

static void save_cnfg(int icnfg)
{
  int ie;

  ie = query_flags(UD_PHASE_SET);
  if (ie == 0) {
    ie = check_bc(0.0) ^ 0x1;
  }
  error_root(ie != 0, 1, "save_cnfg [qcd1.c]",
             "Phase-modified field or unexpected boundary values");

  if (noloc == 0) {
    sprintf(cnfg_file, "%s/%sn%d_%d", loc_dir, nbase, icnfg, my_rank);
    write_cnfg(cnfg_file);
  }

  if (noexp == 0) {
    sprintf(cnfg_file, "%s/%sn%d", cnfg_dir, nbase, icnfg);
    export_cnfg(cnfg_file);
  }

  if (my_rank == 0) {
    if ((noloc == 0) && (noexp == 0)) {
      printf("Configuration no %d saved on the local disks "
             "and exported\n\n",
             icnfg);
    } else if (noloc == 0) {
      printf("Configuration no %d saved on the local disks\n\n", icnfg);
    } else if (noexp == 0) {
      printf("Configuration no %d exported\n\n", icnfg);
    }
  }
}

static void check_endflag(int *iend)
{
  if (my_rank == 0) {
    fend = fopen(end_file, "r");

    if (fend != NULL) {
      fclose(fend);
      remove(end_file);
      (*iend) = 1;
      printf("End flag set, run stopped\n\n");
    } else {
      (*iend) = 0;
    }
  }

  mpc_bcast_i(iend, 1);
}

static void remove_cnfg(int icnfg)
{
  if ((rmold) && (icnfg >= 1)) {
    if (noloc == 0) {
      sprintf(cnfg_file, "%s/%sn%d_%d", loc_dir, nbase, icnfg, my_rank);
      remove(cnfg_file);
    }

    if ((noexp == 0) && (my_rank == 0)) {
      sprintf(cnfg_file, "%s/%sn%d", cnfg_dir, nbase, icnfg);
      remove(cnfg_file);
    }
  }
}

static dat_t compute_log_values(double const *act0, double const *act1,
                                int nact, int iac, int nt)
{
  int i;
  double w0[7], w1[7];
  dat_t ndat;

  w0[0] = 0.0;
  for (i = 0; i <= hmc.nact; i++) {
    w0[0] += (act1[i] - act0[i]);
  }

  w0[1] = plaq_wsum_dble(0) / npl;

  if (bc_type() == 3) {
    w0[2] = temporal_link_sum(0) / volume;
    w0[3] = spatial_link_sum(0) / (3 * volume);
  }

  if (stout_smearing_parms().num_smear > 0) {
    smear_fields();
    w0[4] = plaq_wsum_dble(0) / npl;

    if (bc_type() == 3) {
      w0[5] = temporal_link_sum(0) / volume;
      w0[6] = spatial_link_sum(0) / (3 * volume);
    }
    unsmear_fields();
  }

  mpc_gsum_d(w0, w1, 7);

  ndat.nt = nt;
  ndat.iac = iac;
  ndat.dH = w1[0];

  ndat.avpl = w1[1];
  ndat.average_t_link = w1[2];
  ndat.average_s_link = w1[3];

  ndat.avpl_smeared = w1[4];
  ndat.average_stout_t_link = w1[5];
  ndat.average_stout_s_link = w1[6];

  if (bc_type() == 3) {
    ndat.ploop = polyakov_loop();

    if (stout_smearing_parms().num_smear > 0) {
      smear_fields();
      ndat.smeared_ploop = polyakov_loop();
      unsmear_fields();
    }
  }

  return ndat;
}

int main(int argc, char *argv[])
{
  int nl, icnfg;
  int nwud, nws, nwsd, nwv, nwvd;
  int n, iend, iac;
  double *act0, *act1, siac;
  double wt1, wt2, wtcyc, wtall, wtms, wtmsall;
  su3_dble **usv;
  dat_t ndat;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  read_infile(argc, argv);

  if (noms == 0) {
    alloc_data();
  }

  check_files(&nl, &icnfg);
  geometry();

  hmc_wsize(&nwud, &nws, &nwsd, &nwv, &nwvd);

  alloc_wud(nwud);
  alloc_ws(nws);
  alloc_wsd(nwsd);
  alloc_wv(nwv);
  alloc_wvd(nwvd);

  if ((noms == 0) && (flint)) {
    alloc_wfd(1);
  }

  act0 = malloc(2 * (hmc.nact + 1) * sizeof(*act0));
  act1 = act0 + hmc.nact + 1;
  error(act0 == NULL, 1, "main [qcd1.c]", "Unable to allocate action arrays");

  print_info(icnfg);
  hmc_sanity_check();
  set_mdsteps();
  setup_counters();
  setup_chrono();
  init_rng(icnfg);
  init_ud();

  if (bc_type() == 0) {
    npl = (double)(6 * (N0 - 1) * N1) * (double)(N2 * N3);
  } else {
    npl = (double)(6 * N0 * N1) * (double)(N2 * N3);
  }

  volume = (double)(N0 * N1) * (double)(N2 * N3);

  iend = 0;
  siac = 0.0;
  wtcyc = 0.0;
  wtall = 0.0;
  wtms = 0.0;
  wtmsall = 0.0;

#ifdef dirac_counters
  init_dirac_counters();
#endif

  for (n = 0; (iend == 0) && (n < ntr); n++) {
    MPI_Barrier(MPI_COMM_WORLD);
    wt1 = MPI_Wtime();

    iac = run_hmc(act0, act1);

    MPI_Barrier(MPI_COMM_WORLD);
    wt2 = MPI_Wtime();

    siac += (double)(iac);
    wtcyc += (wt2 - wt1);

    if (((ntr - n - 1) % dtr_log) == 0) {
      ndat = compute_log_values(act0, act1, hmc.nact, iac, nl + n + 1);
      print_log(&ndat);
      wtall += wtcyc;
      save_dat(n, siac, wtcyc, wtall, &ndat);
      wtcyc = 0.0;

      if ((noms == 0) && ((n + 1) >= nth) && (((ntr - n - 1) % dtr_ms) == 0)) {
        MPI_Barrier(MPI_COMM_WORLD);
        wt1 = MPI_Wtime();

        usv = reserve_wud(1);
        store_ud(usv[0]);
        set_data(nl + n + 1);
        recall_ud(usv[0]);
        release_wud();

        MPI_Barrier(MPI_COMM_WORLD);
        wt2 = MPI_Wtime();

        wtms = wt2 - wt1;
        wtmsall += wtms;
        save_msdat(n, wtms, wtmsall);
      }
    }

    if (((n + 1) >= nth) && (((ntr - n - 1) % dtr_cnfg) == 0)) {
      save_cnfg(icnfg);
      export_ranlux(icnfg, rng_file);
      check_endflag(&iend);

      if (my_rank == 0) {
        fflush(flog);
        copy_file(log_file, log_save);
        copy_file(dat_file, dat_save);
        if (noms == 0) {
          copy_file(msdat_file, msdat_save);
        }
        copy_file(rng_file, rng_save);
      }

      remove_cnfg(icnfg - 1);
      icnfg += 1;
    }
  }

  if (my_rank == 0) {
#ifdef dirac_counters
    if (Dw_counter > 0) {
      fprintf(flog, " Dw called %d times \n", Dw_counter);
    }
    if (Dwee_counter > 0) {
      fprintf(flog, " Dwee called %d times \n", Dwee_counter);
    }
    if (Dwoo_counter > 0) {
      fprintf(flog, " Dwoo called %d times \n", Dwoo_counter);
    }
    if (Dwoe_counter > 0) {
      fprintf(flog, " Dwoe called %d times \n", Dwoe_counter);
    }
    if (Dweo_counter > 0) {
      fprintf(flog, " Dweo called %d times \n", Dweo_counter);
    }
    if (Dwhat_counter > 0) {
      fprintf(flog, " Dwhat called %d times \n", Dwhat_counter);
    }
    if (Dw_blk_counter > 0) {
      fprintf(flog, " Dw_blk called %d times \n", Dw_blk_counter);
    }
    if (Dwee_blk_counter > 0) {
      fprintf(flog, " Dwee_blk called %d times \n", Dwee_blk_counter);
    }
    if (Dwoo_blk_counter > 0) {
      fprintf(flog, " Dwoo_blk called %d times \n", Dwoo_blk_counter);
    }
    if (Dwoe_blk_counter > 0) {
      fprintf(flog, " Dwoe_blk called %d times \n", Dwoe_blk_counter);
    }
    if (Dweo_blk_counter > 0) {
      fprintf(flog, " Dweo_blk called %d times \n", Dweo_blk_counter);
    }
    if (Dwhat_blk_counter > 0) {
      fprintf(flog, " Dwhat_blk called %d times \n", Dwhat_blk_counter);
    }

    if (Dw_dble_counter > 0) {
      fprintf(flog, " Dw_dble called %d times \n", Dw_dble_counter);
    }
    if (Dwee_dble_counter > 0) {
      fprintf(flog, " Dwee_dble called %d times \n", Dwee_dble_counter);
    }
    if (Dwoo_dble_counter > 0) {
      fprintf(flog, " Dwoo_dble called %d times \n", Dwoo_dble_counter);
    }
    if (Dwoe_dble_counter > 0) {
      fprintf(flog, " Dwoe_dble called %d times \n", Dwoe_dble_counter);
    }
    if (Dweo_dble_counter > 0) {
      fprintf(flog, " Dweo_dble called %d times \n", Dweo_dble_counter);
    }
    if (Dwhat_dble_counter > 0) {
      fprintf(flog, " Dwhat_dble called %d times \n", Dwhat_dble_counter);
    }
    if (Dw_blk_dble_counter > 0) {
      fprintf(flog, " Dw_blk_dble called %d times \n", Dw_blk_dble_counter);
    }
    if (Dwee_blk_dble_counter > 0) {
      fprintf(flog, " Dwee_blk_dble called %d times \n", Dwee_blk_dble_counter);
    }
    if (Dwoo_blk_dble_counter > 0) {
      fprintf(flog, " Dwoo_blk_dble called %d times \n", Dwoo_blk_dble_counter);
    }
    if (Dwoe_blk_dble_counter > 0) {
      fprintf(flog, " Dwoe_blk_dble called %d times \n", Dwoe_blk_dble_counter);
    }
    if (Dweo_blk_dble_counter > 0) {
      fprintf(flog, " Dweo_blk_dble called %d times \n", Dweo_blk_dble_counter);
    }
    if (Dwhat_blk_dble_counter > 0) {
      fprintf(flog, " Dwhat_blk_dble called %d times \n",
              Dwhat_blk_dble_counter);
    }
#endif
    fprintf(flog, "Memory footprint at program end:\n");
    fprintf(flog, "   Current memory usage per proc: %8.3f MB\n",
            amem_use_mb());
    fprintf(flog, "   Maximal memory usage per proc: %8.3f MB\n",
            amem_max_mb());

    fclose(flog);
  }

  MPI_Finalize();
  exit(0);
}
