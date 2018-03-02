
/*******************************************************************************
 *
 * File dfl_parms.c
 *
 * Copyright (C) 2009, 2010, 2011, 2013 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Deflation parameters
 *
 * The externally accessible functions are
 *
 *   dfl_parms_t set_dfl_parms(int *bs,int Ns)
 *     Sets the parameters of the deflation subspace. The parameters are
 *
 *       bs[4]          Sizes of the blocks in DFL_BLOCKS block grid.
 *
 *       Ns             Number of deflation modes per block (must be
 *                      even and non-zero).
 *
 *     The return value is a structure that contains the above parameters.
 *     Note that these parameters can only be set once.
 *
 *   dfl_parms_t dfl_parms(void)
 *     Returns the parameters currently set for the deflation subspace.
 *
 *   dfl_pro_parms_t set_dfl_pro_parms(int nkv,int nmx,double res)
 *     Sets the parameters used when applying the deflation projection in the
 *     deflated solver program dfl_sap_gcr(). The parameters are
 *
 *       nkv            Maximal number of Krylov vectors to be used by the
 *                      solver for the little Dirac equation before a restart.
 *
 *       nmx            Maximal total number of Krylov vectors generated by
 *                      the solver for the little Dirac equation.
 *
 *       res            Required relative residue when solving the little
 *                      Dirac equation.
 *
 *     The return value is a structure that contains the above parameters.
 *
 *   dfl_pro_parms_t dfl_pro_parms(void)
 *     Returns the parameters currently set for the deflation projectors in
 *     the deflated solver program dfl_sap_gcr().
 *
 *   dfl_gen_parms_t set_dfl_gen_parms(double kappa,double mu,
 *                                     int ninv,int nmr,int ncy,
 *                                     int nkv,int nmx,double res)
 *     Sets the parameters of the inverse iteration procedure that generates
 *     the deflation subspace. The parameters are
 *
 *       kappa          Hopping parameter of the Dirac operator.
 *
 *       mu             Twisted mass parameter.
 *
 *       ninv           Total number of inverse iteration steps (ninv>=4).
 *
 *       nmr            Number of block minimal residual iterations to be
 *                      used when the SAP smoother is applied.
 *
 *       ncy            Number of SAP cycles per inverse iteration.
 *
 *     The return value is a structure that contains the above parameters and
 *     the bare mass m0 that corresponds to the hopping parameter kappa.
 *
 *   dfl_gen_parms_t dfl_gen_parms(void)
 *     Returns the parameters currently set for the generation of the deflation
 *     subspace plus the corresponding bare mass m0.
 *
 *   dfl_upd_parms_t set_dfl_upd_parms(double dtau,int nsm)
 *     Sets the parameters of the deflation subspace update scheme. The
 *     parameters are
 *
 *       dtau           Molecular-dynamics time separation between
 *                      updates of the deflation subspace.
 *
 *       nsm            Number of deflated smoothing interations to be
 *                      applied when the subspace is updated.
 *
 *     The return value is a structure that contains the above parameters.
 *
 *   dfl_upd_parms_t dfl_upd_parms(void)
 *     Returns the parameters currently set for the deflation subspace
 *     update scheme.
 *
 *   void print_dfl_parms(int ipr)
 *     Prints the parameters of the deflation subspace, the projectors, the
 *     subspace generation algorithm and the update scheme to stdout on MPI
 *     process 0. The update scheme is omitted if ipr=0.
 *
 *   void write_dfl_parms(FILE *fdat)
 *     Writes the parameters of the deflation subspace, the projectors, the
 *     subspace generation algorithm and the update scheme to the file fdat
 *     on MPI process 0.
 *
 *   void check_dfl_parms(FILE *fdat)
 *     Compares the parameters of the deflation subspace, the projectors the
 *     subspace generation algorithm and the update scheme with the values
 *     stored on the file fdat on MPI process 0, assuming the latter were
 *     written to the file by the program write_dfl_parms() (mismatches of
 *     maximal solver iteration numbers are not considered to be an error).
 *
 * Notes:
 *
 * To ensure the consistency of the data base, the parameters must be set
 * simultaneously on all processes. The types dfl_parms_t, ... are defined
 * in the file flags.h.
 *
 *******************************************************************************/

#define DFL_PARMS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "utils.h"
#include "flags.h"
#include "global.h"

static dfl_parms_t dfl = {{0, 0, 0, 0}, 0};
static dfl_pro_parms_t dfl_pro = {0, 0, 1.0};
static dfl_gen_parms_t dfl_gen = {0, 0, 0, 0.0, DBL_MAX, 0.0};
static dfl_upd_parms_t dfl_upd = {0, 0.0};

static void check_block_size(int *bs)
{
  int n0, n1, n2, n3;

  error_root((bs[0] < 4) || (bs[1] < 4) || (bs[2] < 4) || (bs[3] < 4) ||
                 (bs[0] > L0) || (bs[1] > L1) || (bs[2] > L2) || (bs[3] > L3),
             1, "check_block_size [dfl_parms.c]",
             "Block sizes are out of range");

  error_root((bs[0] % 2) || (bs[1] % 2) || (bs[2] % 2) || (bs[3] % 2), 1,
             "check_block_size [dfl_parms.c]", "Block sizes must be even");

  error_root((L0 % bs[0]) || (L1 % bs[1]) || (L2 % bs[2]) || (L3 % bs[3]), 1,
             "check_block_size [dfl_parms.c]",
             "Blocks do not divide the local lattice");

  n0 = L0 / bs[0];
  n1 = L1 / bs[1];
  n2 = L2 / bs[2];
  n3 = L3 / bs[3];

  error_root(((NPROC0 * n0) % 2) || ((NPROC1 * n1) % 2) ||
                 ((NPROC2 * n2) % 2) || ((NPROC3 * n3) % 2),
             1, "check_block_size [dfl_parms.c]",
             "There must be an even number of blocks in each direction");

  error_root((n0 * n1 * n2 * n3) % 2, 1, "check_block_size [dfl_parms.c]",
             "The number of blocks in the local lattice must be even");
}

dfl_parms_t set_dfl_parms(int *bs, int Ns)
{
  int iprms[5];

  if (NPROC > 1) {
    iprms[0] = bs[0];
    iprms[1] = bs[1];
    iprms[2] = bs[2];
    iprms[3] = bs[3];
    iprms[4] = Ns;

    mpc_bcast_i(iprms, 5);

    error((iprms[0] != bs[0]) || (iprms[1] != bs[1]) || (iprms[2] != bs[2]) ||
              (iprms[3] != bs[3]) || (iprms[4] != Ns),
          1, "set_dfl_parms [dfl_parms.c]", "Parameters are not global");
  }

  error_root((dfl.Ns > 0) && ((bs[0] != dfl.bs[0]) || (bs[1] != dfl.bs[1]) ||
                              (bs[2] != dfl.bs[2]) || (bs[3] != dfl.bs[3]) ||
                              (Ns != dfl.Ns)),
             1, "set_dfl_parms [dfl_parms.c]",
             "bs[4] and Ns may be set only once");

  check_block_size(bs);
  error_root((Ns < 2) || (Ns & 0x1), 1, "set_dfl_parms [dfl_parms.c]",
             "Improper value of Ns");

  dfl.bs[0] = bs[0];
  dfl.bs[1] = bs[1];
  dfl.bs[2] = bs[2];
  dfl.bs[3] = bs[3];
  dfl.Ns = Ns;

  return dfl;
}

dfl_parms_t dfl_parms(void) { return dfl; }

dfl_pro_parms_t set_dfl_pro_parms(int nkv, int nmx, double res)
{
  int iprms[2];
  double dprms[1];

  if (NPROC > 1) {
    iprms[0] = nkv;
    iprms[1] = nmx;

    dprms[0] = res;

    mpc_bcast_i(iprms, 2);
    mpc_bcast_d(dprms, 1);

    error((iprms[0] != nkv) || (iprms[1] != nmx) || (dprms[0] != res), 1,
          "set_dfl_pro_parms [dfl_parms.c]", "Parameters are not global");
  }

  error_root((nkv < 1) || (nmx < 1) || (res <= DBL_EPSILON), 1,
             "set_dfl_pro_parms [dfl_parms.c]", "Improper parameter values");

  dfl_pro.nkv = nkv;
  dfl_pro.nmx = nmx;
  dfl_pro.res = res;

  return dfl_pro;
}

dfl_pro_parms_t dfl_pro_parms(void) { return dfl_pro; }

dfl_gen_parms_t set_dfl_gen_parms(double kappa, double mu, int ninv, int nmr,
                                  int ncy)
{
  int iprms[3];
  double dprms[2];
  ani_params_t ani;

  if (NPROC > 1) {
    iprms[0] = ninv;
    iprms[1] = nmr;
    iprms[2] = ncy;

    dprms[0] = kappa;
    dprms[1] = mu;

    mpc_bcast_i(iprms, 3);
    mpc_bcast_d(dprms, 2);

    error((iprms[0] != ninv) || (iprms[1] != nmr) || (iprms[2] != ncy) ||
              (dprms[0] != kappa) || (dprms[1] != mu),
          1, "set_dfl_gen_parms [dfl_parms.c]", "Parameters are not global");
  }

  error_root((ninv < 4) || (nmr < 1) || (ncy < 1) || (kappa < 0.0), 1,
             "set_dfl_gen_parms [dfl_parms.c]", "Parameters are out of range");

  dfl_gen.ninv = ninv;
  dfl_gen.nmr = nmr;
  dfl_gen.ncy = ncy;

  dfl_gen.kappa = kappa;
  dfl_gen.mu = mu;

  ani = ani_parms();

  error(ani.xi == 0.0, 1, "set_dfl_gen_parms [dfl_parms.c]",
        "Anisotropic parameters not set");

  if (kappa != 0.0)
    dfl_gen.m0 = 1.0 / (2.0 * kappa) - 1.0 - 3.0 * ani.nu / ani.xi;
  else
    dfl_gen.m0 = DBL_MAX;

  return dfl_gen;
}

dfl_gen_parms_t dfl_gen_parms(void) { return dfl_gen; }

dfl_upd_parms_t set_dfl_upd_parms(double dtau, int nsm)
{
  int iprms[1];
  double dprms[1];

  if (NPROC > 1) {
    iprms[0] = nsm;
    dprms[0] = dtau;

    mpc_bcast_i(iprms, 1);
    mpc_bcast_d(dprms, 1);

    error((iprms[0] != nsm) || (dprms[0] != dtau), 1,
          "set_dfl_upd_parms [dfl_parms.c]", "Parameters are not global");
  }

  error_root((dtau < 0.0) || (nsm < 0), 1, "set_dfl_upd_parms [dfl_parms.c]",
             "Improper parameter values");

  dfl_upd.dtau = dtau;
  dfl_upd.nsm = nsm;

  return dfl_upd;
}

dfl_upd_parms_t dfl_upd_parms(void) { return dfl_upd; }

void print_dfl_parms(int ipr)
{
  int my_rank, n;

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    printf("Deflation subspace:\n");
    printf("bs = %d %d %d %d\n", dfl.bs[0], dfl.bs[1], dfl.bs[2], dfl.bs[3]);
    printf("Ns = %d\n\n", dfl.Ns);

    printf("Deflation projection:\n");
    printf("nkv = %d\n", dfl_pro.nkv);
    printf("nmx = %d\n", dfl_pro.nmx);
    printf("res = %.1e\n\n", dfl_pro.res);

    printf("Deflation subspace generation:\n");
    n = fdigits(dfl_gen.kappa);
    printf("kappa = %.*f\n", IMAX(n, 6), dfl_gen.kappa);
    n = fdigits(dfl_gen.mu);
    printf("mu = %.*f\n", IMAX(n, 1), dfl_gen.mu);
    printf("ninv = %d\n", dfl_gen.ninv);
    printf("nmr = %d\n", dfl_gen.nmr);
    printf("ncy = %d\n\n", dfl_gen.ncy);

    if (ipr) {
      printf("Deflation subspace update scheme:\n");
      n = fdigits(dfl_upd.dtau);
      printf("dtau = %.*f\n", IMAX(n, 1), dfl_upd.dtau);
      printf("nsm = %d\n\n", dfl_upd.nsm);
    }
  }
}

void write_dfl_parms(FILE *fdat)
{
  int my_rank, endian;
  int i, iw;
  stdint_t istd[11];
  double dstd[4];

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  endian = endianness();

  if (my_rank == 0) {
    for (i = 0; i < 4; i++)
      istd[i] = (stdint_t)(dfl.bs[i]);

    istd[4] = (stdint_t)(dfl.Ns);
    istd[5] = (stdint_t)(dfl_pro.nkv);
    istd[6] = (stdint_t)(dfl_pro.nmx);
    istd[7] = (stdint_t)(dfl_gen.ninv);
    istd[8] = (stdint_t)(dfl_gen.nmr);
    istd[9] = (stdint_t)(dfl_gen.ncy);
    istd[10] = (stdint_t)(dfl_upd.nsm);

    dstd[0] = dfl_pro.res;
    dstd[1] = dfl_gen.kappa;
    dstd[2] = dfl_gen.mu;
    dstd[3] = dfl_upd.dtau;

    if (endian == BIG_ENDIAN) {
      bswap_int(11, istd);
      bswap_double(4, dstd);
    }

    iw = fwrite(istd, sizeof(stdint_t), 11, fdat);
    iw += fwrite(dstd, sizeof(double), 4, fdat);
    error_root(iw != 15, 1, "write_dfl_parms [dfl_parms.c]",
               "Incorrect write count");
  }
}

void check_dfl_parms(FILE *fdat)
{
  int my_rank, endian;
  int i, ir, ie;
  stdint_t istd[11];
  double dstd[4];

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  endian = endianness();

  if (my_rank == 0) {
    ir = fread(istd, sizeof(stdint_t), 11, fdat);
    ir += fread(dstd, sizeof(double), 4, fdat);
    error_root(ir != 15, 1, "check_dfl_parms [dfl_parms.c]",
               "Incorrect read count");

    if (endian == BIG_ENDIAN) {
      bswap_int(11, istd);
      bswap_double(4, dstd);
    }

    ie = 0;

    for (i = 0; i < 4; i++)
      ie |= (istd[i] != (stdint_t)(dfl.bs[i]));

    ie |= (istd[4] != (stdint_t)(dfl.Ns));
    ie |= (istd[5] != (stdint_t)(dfl_pro.nkv));
    ie |= (istd[7] != (stdint_t)(dfl_gen.ninv));
    ie |= (istd[8] != (stdint_t)(dfl_gen.nmr));
    ie |= (istd[9] != (stdint_t)(dfl_gen.ncy));
    ie |= (istd[10] != (stdint_t)(dfl_upd.nsm));

    ie |= (dstd[0] != dfl_pro.res);
    ie |= (dstd[1] != dfl_gen.kappa);
    ie |= (dstd[2] != dfl_gen.mu);
    ie |= (dstd[3] != dfl_upd.dtau);

    error_root(ie != 0, 1, "check_dfl_parms [dfl_parms.c]",
               "Parameters do not match");
  }
}
