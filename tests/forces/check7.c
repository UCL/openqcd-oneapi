
/*******************************************************************************
 *
 * File check7.c
 *
 * Copyright (C) 2011-2013, 2016 Stefan Schaefer, Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Check of force2() and action2().
 *
 *******************************************************************************/

#define OPENQCD_INTERNAL

#if !defined (STATIC_SIZES)
#error : This test cannot be compiled with dynamic lattice sizes
#endif

#include "dfl.h"
#include "forces.h"
#include "global.h"
#include "lattice.h"
#include "linalg.h"
#include "mdflds.h"
#include "mpi.h"
#include "random.h"
#include "su3fcts.h"
#include "uflds.h"

#define N0 (NPROC0 * L0)

static void rot_ud(double eps)
{
  int bc, ix, t, ifc;
  su3_dble *u;
  su3_alg_dble *mom;
  mdflds_t *mdfs;

  bc = bc_type();
  mdfs = mdflds();
  mom = (*mdfs).mom;
  u = udfld();

  for (ix = (VOLUME / 2); ix < VOLUME; ix++) {
    t = global_time(ix);

    if (t == 0) {
      expXsu3(eps, mom, u);
      mom += 1;
      u += 1;

      if (bc != 0) {
        expXsu3(eps, mom, u);
      }
      mom += 1;
      u += 1;

      for (ifc = 2; ifc < 8; ifc++) {
        if (bc != 1) {
          expXsu3(eps, mom, u);
        }
        mom += 1;
        u += 1;
      }
    } else if (t == (N0 - 1)) {
      if (bc != 0) {
        expXsu3(eps, mom, u);
      }
      mom += 1;
      u += 1;

      for (ifc = 1; ifc < 8; ifc++) {
        expXsu3(eps, mom, u);
        mom += 1;
        u += 1;
      }
    } else {
      for (ifc = 0; ifc < 8; ifc++) {
        expXsu3(eps, mom, u);
        mom += 1;
        u += 1;
      }
    }
  }

  set_flags(UPDATED_UD);
}

static int is_frc_zero(su3_alg_dble *f)
{
  int ie;

  ie = 1;
  ie &= is_equal_d((*f).c1, 0.0);
  ie &= is_equal_d((*f).c2, 0.0);
  ie &= is_equal_d((*f).c3, 0.0);
  ie &= is_equal_d((*f).c4, 0.0);
  ie &= is_equal_d((*f).c5, 0.0);
  ie &= is_equal_d((*f).c6, 0.0);
  ie &= is_equal_d((*f).c7, 0.0);
  ie &= is_equal_d((*f).c8, 0.0);

  return ie;
}

static void check_bnd_frc(void)
{
  int bc, ix, t, ifc, ie;
  su3_alg_dble *frc;
  mdflds_t *mdfs;

  bc = bc_type();
  mdfs = mdflds();
  frc = (*mdfs).frc;
  ie = 0;

  for (ix = (VOLUME / 2); ix < VOLUME; ix++) {
    t = global_time(ix);

    if ((t == 0) && (bc == 0)) {
      ie |= is_frc_zero(frc);
      frc += 1;

      ie |= (is_frc_zero(frc) ^ 0x1);
      frc += 1;

      for (ifc = 2; ifc < 8; ifc++) {
        ie |= is_frc_zero(frc);
        frc += 1;
      }
    } else if ((t == 0) && (bc == 1)) {
      ie |= is_frc_zero(frc);
      frc += 1;

      ie |= is_frc_zero(frc);
      frc += 1;

      for (ifc = 2; ifc < 8; ifc++) {
        ie |= (is_frc_zero(frc) ^ 0x1);
        frc += 1;
      }
    } else if ((t == (N0 - 1)) && (bc == 0)) {
      ie |= (is_frc_zero(frc) ^ 0x1);
      frc += 1;

      for (ifc = 1; ifc < 8; ifc++) {
        ie |= is_frc_zero(frc);
        frc += 1;
      }
    } else {
      for (ifc = 0; ifc < 8; ifc++) {
        ie |= is_frc_zero(frc);
        frc += 1;
      }
    }
  }

  error(ie != 0, 1, "check_bnd_frc [check7.c]",
        "Force field vanishes on an incorrect set of links");
}

static double dSdt(double mu0, double mu1, int ipf, int isp, int *status)
{
  mdflds_t *mdfs;

  set_frc2zero();
  force2(mu0, mu1, ipf, isp, 0, 1.2345, status);
  check_bnd_frc();
  mdfs = mdflds();

  return scalar_prod_alg(4 * VOLUME, 0, (*mdfs).mom, (*mdfs).frc);
}

int main(int argc, char *argv[])
{
  int my_rank, bc, isp, status[6], mnkv;
  int bs[4], Ns, nmx, nkv, nmr, ncy, ninv;
  int isap, idfl;
  double chi[2], chi_prime[2], theta[3];
  double kappa, mu, mu0, mu1, res;
  double eps, act0, act1, dact, dsdt;
  double dev_act[2], dev_frc, sig_loss, rdmy;
  solver_parms_t sp;
  FILE *flog = NULL, *fin = NULL;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    flog = freopen("check7.log", "w", stdout);
    fin = freopen("check6.in", "r", stdin);

    printf("\n");
    printf("Check of force2() and action2()\n");
    printf("-------------------------------\n\n");

    printf("%dx%dx%dx%d lattice, ", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid, ", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n\n", L0, L1, L2, L3);

    bc = find_opt(argc, argv, "-bc");

    if (bc != 0) {
      error_root(sscanf(argv[bc + 1], "%d", &bc) != 1, 1, "main [check7.c]",
                 "Syntax: check7 [-bc <type>]");
    }
  }

  MPI_Bcast(&bc, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (bc == 3) {
    set_ani_parms(1, 1.5, 4.3, 1.5, 0.9, 1.0, 1.0, 0.87, 1.23);
    print_ani_parms();
    set_lat_parms(5.5, 1.0, 0, NULL, 1.0);
  } else {
    set_no_ani_parms();
    set_lat_parms(5.5, 1.0, 0, NULL, 1.782);
  }

  print_lat_parms();

  chi[0] = 0.123;
  chi[1] = -0.534;
  chi_prime[0] = 0.912;
  chi_prime[1] = 0.078;
  theta[0] = 0.38;
  theta[1] = -1.25;
  theta[2] = 0.54;
  set_bc_parms(bc, 1.0, 1.0, 0.953, 1.203, chi, chi_prime, theta);
  print_bc_parms(2);

  if (my_rank == 0) {
    find_section("SAP");
    read_iprms("bs", 4, bs);
  }

  MPI_Bcast(bs, 4, MPI_INT, 0, MPI_COMM_WORLD);
  set_sap_parms(bs, 1, 4, 5);

  if (my_rank == 0) {
    find_section("Deflation subspace");
    read_iprms("bs", 4, bs);
    read_line("Ns", "%d", &Ns);
  }

  MPI_Bcast(bs, 4, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Ns, 1, MPI_INT, 0, MPI_COMM_WORLD);
  set_dfl_parms(bs, Ns);

  if (my_rank == 0) {
    find_section("Deflation subspace generation");
    read_line("kappa", "%lf", &kappa);
    read_line("mu", "%lf", &mu);
    read_line("ninv", "%d", &ninv);
    read_line("nmr", "%d", &nmr);
    read_line("ncy", "%d", &ncy);
  }

  MPI_Bcast(&kappa, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&mu, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ninv, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nmr, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ncy, 1, MPI_INT, 0, MPI_COMM_WORLD);
  set_dfl_gen_parms(kappa, mu, ninv, nmr, ncy);

  if (my_rank == 0) {
    find_section("Deflation projection");
    read_line("nkv", "%d", &nkv);
    read_line("nmx", "%d", &nmx);
    read_line("res", "%lf", &res);
  }

  MPI_Bcast(&nkv, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nmx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&res, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  set_dfl_pro_parms(nkv, nmx, res);

  set_hmc_parms(0, NULL, 1, 0, NULL, 1, 1.0);
  mnkv = 0;

  for (isp = 0; isp < 3; isp++) {
    read_solver_parms(isp);
    sp = solver_parms(isp);

    if (sp.nkv > mnkv) {
      mnkv = sp.nkv;
    }
  }

  if (my_rank == 0) {
    fclose(fin);
  }

  print_solver_parms(&isap, &idfl);
  print_sap_parms(1);
  print_dfl_parms(0);

  start_ranlux(0, 1245);
  geometry();

  set_sw_parms(-0.0123);
  mnkv = 2 * mnkv + 2;
  if (mnkv < (Ns + 2)) {
    mnkv = Ns + 2;
  }
  if (mnkv < 5) {
    mnkv = 5;
  }

  alloc_ws(mnkv);
  alloc_wsd(6);
  alloc_wv(2 * nkv + 2);
  alloc_wvd(4);

  for (isp = 0; isp < 3; isp++) {
    if (isp == 0) {
      mu0 = 1.0;
      mu1 = 1.5;
      eps = 1.0e-4;
    } else if (isp == 1) {
      mu0 = 0.1;
      mu1 = 0.25;
      eps = 2.0e-4;
    } else {
      mu0 = 0.01;
      mu1 = 0.02;
      eps = 3.0e-4;
    }

    random_ud();
    set_ud_phase();
    random_mom();

    if (isp == 2) {
      dfl_modes(status);
      error_root(status[0] < 0, 1, "main [check7.c]", "dfl_modes failed");
    }

    status[0] = 0;
    status[1] = 0;

    act0 = setpf2(mu0, mu1, 0, isp, 0, status);
    error_root((status[0] < 0) || (status[1] < 0), 1, "main [check7.c]",
               "setpf2 failed (isp,mu0,mu1=%d,%.2e,%.2e)", isp, mu0, mu1);
    act1 = action2(mu0, mu1, 0, isp, 0, status);
    error_root((status[0] < 0) || (status[1] < 0), 1, "main [check7.c]",
               "action2 failed (isp,mu0,mu1=%d,%.2e,%.2e)", isp, mu0, mu1);

    rdmy = fabs(act1 - act0);
    MPI_Reduce(&rdmy, dev_act, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    rdmy = act1 - act0;
    MPI_Reduce(&rdmy, dev_act + 1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(dev_act, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    dsdt = dSdt(mu0, mu1, 0, isp, status);

    if (my_rank == 0) {
      printf("Solver number %d, mu0 = %.2e, mu1 = %.2e\n", isp, mu0, mu1);

      if (isp == 0) {
        printf("Status = %d\n", status[0]);
      } else if (isp == 1) {
        printf("Status = %d,%d\n", status[0], status[1]);
      } else {
        printf("Status = (%d,%d,%d),(%d,%d,%d)\n", status[0], status[1],
               status[2], status[3], status[4], status[5]);
      }

      printf("Absolute action difference |setpf2-action2| = %.1e,",
             fabs(dev_act[1]));
      printf(" %.1e (local)\n", dev_act[0]);
      fflush(flog);
    }

    rot_ud(eps);
    act0 = 2.0 * action2(mu0, mu1, 0, isp, 0, status) / 3.0;
    rot_ud(-eps);

    rot_ud(-eps);
    act1 = 2.0 * action2(mu0, mu1, 0, isp, 0, status) / 3.0;
    rot_ud(eps);

    rot_ud(2.0 * eps);
    act0 -= action2(mu0, mu1, 0, isp, 0, status) / 12.0;
    rot_ud(-2.0 * eps);

    rot_ud(-2.0 * eps);
    act1 -= action2(mu0, mu1, 0, isp, 0, status) / 12.0;
    rot_ud(2.0 * eps);

    dact = 1.2345 * (act0 - act1) / eps;
    dev_frc = dsdt - dact;
    sig_loss = -log10(fabs(1.0 - act0 / act1));

    rdmy = dsdt;
    MPI_Reduce(&rdmy, &dsdt, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dsdt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    rdmy = dev_frc;
    MPI_Reduce(&rdmy, &dev_frc, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dev_frc, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    rdmy = sig_loss;
    MPI_Reduce(&rdmy, &sig_loss, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sig_loss, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
      printf("Relative deviation of dS/dt = %.2e ", fabs(dev_frc / dsdt));
      printf("[significance loss = %d digits]\n\n", (int)(sig_loss));
      fflush(flog);
    }
  }

  if (my_rank == 0) {
    fclose(flog);
  }

  MPI_Finalize();
  exit(0);
}
