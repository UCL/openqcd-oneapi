
/*******************************************************************************
 *
 * File dfl_modes.c
 *
 * Copyright (C) 2007, 2011-2013 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Computation of the global modes used to construct the deflation subspace.
 *
 * The externally accessible functions are
 *
 *   void dfl_modes(int *status)
 *     Computes the basis vectors of the deflation subspace by applying a
 *     smoothing procedure to a set of random fields. The subspace is then
 *     initialized by calling the program dfl_subspace().
 *
 *   void dfl_update(int nsm, int *status)
 *     Updates the deflation subspace by applying nsm deflated smoothing
 *     steps to the global fields from which the current subspace was built.
 *
 *   void dfl_modes2(int *status)
 *     Calls the program dfl_modes() and, if status[0]=-3 is returned,
 *     call dfl_modes() a second time. The array status must have at least
 *     two elements that report the status values returned in the two calls.
 *     Normally dfl_modes() is called only once in which case status[1]=0.
 *
 *   void dfl_update2(int nsm, int *status)
 *     Calls the program dfl_update() and, if status[0]=-3 is returned,
 *     call dfl_modes(). The array status must have at least two elements
 *     that report the status values returned in the two calls. Normally
 *     dfl_modes is not called in which case status[1]=0.
 *
 * Notes:
 *
 * The deflation subspace is generated by inverse iteration and projection
 * to the basis fields in the DFL_BLOCKS grid. See
 *
 *  M. Luescher: "Local coherence and deflation of the low quark modes
 *                in lattice QCD", JHEP 0707 (2007) 081, and
 *
 *               "Deflation acceleration of lattice QCD simulations",
 *               JHEP 0712 (2007) 011.
 *
 * The following parameter-setting programs must have been called before
 * launching the programs in this module:
 *
 *  set_lat_parms()        SW improvement coefficient.
 *
 *  set_bc_parms()         Boundary conditions and associated improvement
 *                         coefficients.
 *
 *  set_sap_parms()        Block size of the SAP block grid.
 *
 *  set_dfl_parms()        Parameters of the deflation subspace.
 *
 *  set_dfl_pro_parms()    Deflation projection parameters.
 *
 *  set_dfl_gen_parms()    Subspace generation parameters.
 *
 * See doc/parms.pdf and the relevant files in the modules/flags directory
 * for further explanations. The update program moreover assumes that the
 * current deflation subspace was previously initialized using dfl_modes().
 *
 * Each inverse iteration step consists of the application of a few cycles
 * of the Schwarz alternating procedure (see sap/sap.c). After the first
 * three iterations, the fields are deflated, using the current deflation
 * subspace before the SAP is applied. The parameters used in this process
 * are
 *
 *  kappa          Hopping parameter of the Dirac operator.
 *
 *  mu             Twisted mass parameter.
 *
 *  ninv           Total number of inverse iteration steps (ninv>=4).
 *
 *  nmr            Number of block minimal residual iterations to be
 *                 used when the SAP smoother is applied.
 *
 *  ncy            Number of SAP cycles per inverse iteration.
 *
 * All these are set by set_dfl_gen_parms(). Additionally, the values of
 * parameters
 *
 *  nkv            Maximal number of Krylov vectors to be used by the
 *                 solver for the little Dirac equation before a restart.
 *
 *  nmx            Maximal total number of Krylov vectors generated by
 *                 the solver for the little Dirac equation.
 *
 *  res            Required relative residue when solving the little
 *                 Dirac equation.
 *
 * are set by set_dfl_pro_parms().
 *
 * On exit the argument status[0] reports the average solver iteration numbers
 * that were required for the solution of the little Dirac equation. A negative
 * value indicates that the program failed (-1: the solver did not converge, -2:
 * the inversion of the SW term was not safe, -3: the inversion of the diagonal
 * part of the little Dirac operator was not safe). In all these cases, the
 * deflation subspace is initialized with the fields that were computed before
 * the failure occured.
 *
 * The programs dfl_modes2() and dfl_update2() can be used in place of the
 * programs dfl_modes() and dfl_update(), respectively, if some protection
 * against the rare cases, where the little Dirac operator turns out to be
 * accidentally ill-conditioned, is desired.
 *
 * The programs in this module perform global operations and must be called
 * simultaneously on all MPI processes. The required workspaces are
 *
 *  spinor              Ns+2        (Ns: number of deflation modes per block)
 *  complex             2*nkv+2
 *  complex_dble        4
 *
 * (see utils/wspace.c)
 *
 * Some debugging output is printed to stdout on process 0 if DFL_MODES_DBG is
 * defined at compilation time.
 *
 *******************************************************************************/

#define DFL_MODES_C

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
#include "sw_term.h"
#include "uflds.h"
#include "vflds.h"

typedef union
{
  spinor s;
  float r[24];
} spin_t;

static int my_rank, eoflg;
static int Ns = 0, nv, nrn;
static double m0;
static complex_dble *cs1, *cs2;
static dfl_pro_parms_t dpr;
static dfl_gen_parms_t dgn;

#ifdef DFL_MODES_DBG

static void print_res(spinor **mds)
{
  int k;
  double r;
  spinor **ws;

  ws = reserve_ws(1);

  for (k = 0; k < Ns; k++) {
    Dw((float)(dgn.mu), mds[k], ws[0]);
    r = (double)(norm_square(VOLUME, 1, ws[0]) /
                 norm_square(VOLUME, 1, mds[k]));

    if (my_rank == 0) {
      printf("k = %2d, |Dw*psi|/|psi| = %.1e\n", k, sqrt(r));
    }
  }

  release_ws();
}

#endif

static int set_frame(void)
{
  int nb, isw, ifail;
  int *bs, swu, swe, swo;
  sw_parms_t sw;
  tm_parms_t tm;
  dfl_parms_t dfl;

  if (Ns == 0) {
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    dfl = dfl_parms();
    error_root(dfl.Ns == 0, 1, "set_frame [dfl_modes.c]",
               "Parameters of the deflation subspace are not set");

    Ns = dfl.Ns;
    bs = dfl.bs;
    nv = Ns * VOLUME / (bs[0] * bs[1] * bs[2] * bs[3]);
    nrn = 0;

    cs1 = amalloc(2 * Ns * sizeof(*cs1), ALIGN);
    error(cs1 == NULL, 1, "set_frame [dfl_modes.c]",
          "Unable to allocate auxiliary arrays");
    cs2 = cs1 + Ns;

    blk_list(SAP_BLOCKS, &nb, &isw);

    if (nb == 0) {
      alloc_bgr(SAP_BLOCKS);
    }
  }

  dpr = dfl_pro_parms();
  error_root(dpr.nkv == 0, 1, "set_frame [dfl_modes.c]",
             "Deflation projection parameters are not set");

  dgn = dfl_gen_parms();
  error_root(dgn.ninv == 0, 1, "set_frame [dfl_modes.c]",
             "Deflation subspace generation parameters are not set");

  if (query_flags(U_MATCH_UD) != 1) {
    assign_ud2u();
  }

  if (query_grid_flags(SAP_BLOCKS, UBGR_MATCH_UD) != 1) {
    assign_ud2ubgr(SAP_BLOCKS);
  }

  sw = sw_parms();
  m0 = sw.m0;
  set_sw_parms(dgn.m0);
  sw_term(NO_PTS);

  if ((query_flags(SW_UP2DATE) != 1) || (query_flags(SW_E_INVERTED) == 1) ||
      (query_flags(SW_O_INVERTED) == 1)) {
    assign_swd2sw();
  }

  ifail = 0;
  swu = query_grid_flags(SAP_BLOCKS, SW_UP2DATE);
  swe = query_grid_flags(SAP_BLOCKS, SW_E_INVERTED);
  swo = query_grid_flags(SAP_BLOCKS, SW_O_INVERTED);

  if ((swu != 1) || (swe == 1) || (swo != 1)) {
    ifail = assign_swd2swbgr(SAP_BLOCKS, ODD_PTS);
  }

  tm = tm_parms();
  eoflg = tm.eoflg;
  if (eoflg != 1) {
    set_tm_parms(1);
  }

  return ifail;
}

static void random_fields(spinor **mds)
{
  int k, l;
  spin_t *s, *sm;
#ifdef SITERANDOM
  int ix;
#endif

  for (k = 0; k < Ns; k++) {
    s = (spin_t *)(mds[k]);
    sm = s + VOLUME;
#ifdef SITERANDOM
    ix = 0;
#endif

    for (; s < sm; s++) {
#ifdef SITERANDOM
      ranlxs_site((*s).r, 24, ix++);
#else
      ranlxs((*s).r, 24);
#endif

      for (l = 0; l < 24; l++) {
        (*s).r[l] -= 0.5f;
      }
    }

    bnd_s2zero(ALL_PTS, mds[k]);
  }

  nrn = 0;
}

static void renormalize_fields(spinor **mds)
{
  int k, l;

  for (k = 0; k < Ns; k++) {
    for (l = 0; l < k; l++) {
      project(VOLUME, 1, mds[k], mds[l]);
    }

    normalize(VOLUME, 1, mds[k]);
  }

  nrn = 0;
}

static void sum_vprod(int n)
{
  int k;

  if (NPROC > 1) {
    MPI_Reduce(cs1, cs2, 2 * n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(cs2, 2 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    for (k = 0; k < n; k++) {
      cs2[k].re = cs1[k].re;
      cs2[k].im = cs1[k].im;
    }
  }
}

static void restore_fields(spinor **mds)
{
  int k, l;
  complex **vs, z;

  vs = vflds() + Ns;

  for (k = 0; k < Ns; k++) {
    if (k > 0) {
      for (l = 0; l < k; l++) {
        z = vprod(nv, 0, vs[l], vs[k]);
        cs1[l].re = (double)(z.re);
        cs1[l].im = (double)(z.im);
      }

      sum_vprod(k);

      for (l = 0; l < k; l++) {
        z.re = (float)(-cs2[l].re);
        z.im = (float)(-cs2[l].im);
        mulc_vadd(nv, vs[k], vs[l], z);
      }
    }

    vnormalize(nv, 1, vs[k]);
    dfl_v2s(vs[k], mds[k]);
  }

  nrn = 0;
}

#ifdef DFL_MODES_DBG

static void smooth_fields(int ncy, spinor **mds)
{
  int k, l;
  double r0, r1;
  spinor **ws;

  ws = reserve_ws(1);
  r0 = 1.0;
  r1 = 1.0;

  for (k = 0; k < Ns; k++) {
    assign_s2s(VOLUME, mds[k], ws[0]);
    set_s2zero(VOLUME, mds[k]);
    r0 = (double)(norm_square(VOLUME, 1, ws[0]));

    for (l = 0; l < ncy; l++) {
      sap((float)(dgn.mu), 1, dgn.nmr, mds[k], ws[0]);
    }

    r1 = (double)(norm_square(VOLUME, 1, ws[0]));
    r0 = sqrt(r0);
    r1 = sqrt(r1);

    if (my_rank == 0) {
      printf("SAP smoothing of mode no %2d: "
             "residue %.1e -> %.1e, ratio = %.1e\n",
             k, r0, r1, r1 / r0);
    }
  }

  release_ws();
}

static void dfl_smooth_fields(spinor **mds, int *status)
{
  int k, l, stat;
  double r0, r1;
  complex **vs, **wv;
  complex_dble **wvd;
  spinor **ws;

  vs = vflds() + Ns;
  wv = reserve_wv(1);
  wvd = reserve_wvd(1);
  ws = reserve_ws(2);
  r0 = 1.0;
  r1 = 1.0;

  for (k = 0; k < Ns; k++) {
    assign_v2vd(nv, vs[k], wvd[0]);
    ltl_gcr(dpr.nkv, dpr.nmx, dpr.res, dgn.mu, wvd[0], wvd[0], &stat);
    assign_vd2v(nv, wvd[0], wv[0]);

    dfl_v2s(wv[0], ws[1]);
    Dw((float)(dgn.mu), ws[1], ws[0]);
    mulr_spinor_add(VOLUME, mds[k], ws[0], -1.0f);
    assign_s2s(VOLUME, mds[k], ws[0]);
    set_s2zero(VOLUME, mds[k]);
    r0 = (double)(norm_square(VOLUME, 1, ws[0]));

    for (l = 0; l < dgn.ncy; l++) {
      sap((float)(dgn.mu), 1, dgn.nmr, mds[k], ws[0]);
    }

    r1 = (double)(norm_square(VOLUME, 1, ws[0]));
    r0 = sqrt(r0);
    r1 = sqrt(r1);

    if (my_rank == 0) {
      printf("Deflated SAP smoothing of mode no %2d: "
             "status = %d, residue %.1e -> %.1e, ratio = %.1e\n",
             k, stat, r0, r1, r1 / r0);
    }

    mulr_spinor_add(VOLUME, mds[k], ws[1], 1.0f);

    if (status[0] >= 0) {
      if (stat >= 0) {
        status[0] += stat;
      } else {
        status[0] = stat;
      }
    }
  }

  release_ws();
  release_wvd();
  release_wv();
}

#else

static void smooth_fields(int ncy, spinor **mds)
{
  int k, l;
  spinor **ws;

  ws = reserve_ws(1);

  for (k = 0; k < Ns; k++) {
    assign_s2s(VOLUME, mds[k], ws[0]);
    set_s2zero(VOLUME, mds[k]);

    for (l = 0; l < ncy; l++) {
      sap((float)(dgn.mu), 1, dgn.nmr, mds[k], ws[0]);
    }
  }

  release_ws();
}

static void dfl_smooth_fields(spinor **mds, int *status)
{
  int k, l, stat;
  complex **vs, **wv;
  complex_dble **wvd;
  spinor **ws;

  vs = vflds() + Ns;
  wv = reserve_wv(1);
  wvd = reserve_wvd(1);
  ws = reserve_ws(2);

  for (k = 0; k < Ns; k++) {
    assign_v2vd(nv, vs[k], wvd[0]);
    ltl_gcr(dpr.nkv, dpr.nmx, dpr.res, dgn.mu, wvd[0], wvd[0], &stat);
    assign_vd2v(nv, wvd[0], wv[0]);

    dfl_v2s(wv[0], ws[1]);
    Dw((float)(dgn.mu), ws[1], ws[0]);
    mulr_spinor_add(VOLUME, mds[k], ws[0], -1.0f);
    assign_s2s(VOLUME, mds[k], ws[0]);
    set_s2zero(VOLUME, mds[k]);

    for (l = 0; l < dgn.ncy; l++) {
      sap((float)(dgn.mu), 1, dgn.nmr, mds[k], ws[0]);
    }

    mulr_spinor_add(VOLUME, mds[k], ws[1], 1.0f);

    if (status[0] >= 0) {
      if (stat >= 0) {
        status[0] += stat;
      } else {
        status[0] = stat;
      }
    }
  }

  release_ws();
  release_wvd();
  release_wv();
}

#endif

void dfl_modes(int *status)
{
  int n, ifail;
  spinor **mds;

  status[0] = 0;
  ifail = set_frame();
  mds = reserve_ws(Ns);
  random_fields(mds);

#ifdef DFL_MODES_DBG
  if (my_rank == 0) {
    printf("Progress report [program dfl_modes]:\n\n");
    printf("Ns = %d, ninv = %d, nmr = %d, ncy = %d\n", Ns, dgn.ninv, dgn.nmr,
           dgn.ncy);
    printf("nkv = %d, nmx = %d, res = %.1e, ifail = %d\n\n", dpr.nkv, dpr.nmx,
           dpr.res, ifail);
  }
#endif

  if (ifail) {
    dfl_subspace(mds);
    status[0] = -2;
  } else {
    for (n = 0; n < 3; n++) {
      smooth_fields(n + 1, mds);

#ifdef DFL_MODES_DBG
      print_res(mds);
#endif
    }

    for (; n < dgn.ninv; n++) {
      if (nrn > 3) {
        renormalize_fields(mds);
      }

      dfl_subspace(mds);
      ifail = set_Awhat(dgn.mu);

      if (ifail) {
        status[0] = -3;
        break;
      } else {
        dfl_smooth_fields(mds, status);
        nrn += 1;

        if (status[0] < 0) {
          break;
        }

#ifdef DFL_MODES_DBG
        print_res(mds);
#endif
      }
    }

    if (status[0] >= 0) {
      dfl_subspace(mds);
      n = Ns * (dgn.ninv - 3);
      status[0] = (status[0] + n / 2) / n;
    }
  }

  release_ws();
  set_sw_parms(m0);
  if (eoflg != 1) {
    set_tm_parms(eoflg);
  }

#ifdef DFL_MODES_DBG
  if (my_rank == 0) {
    printf("status = %d\n", status[0]);
    printf("dfl_modes: all done\n\n");
    fflush(stdout);
  }
#endif
}

void dfl_update(int nsm, int *status)
{
  int n, ifail, iprms[1];
  spinor **mds;

  if (NPROC > 1) {
    iprms[0] = nsm;

    MPI_Bcast(iprms, 1, MPI_INT, 0, MPI_COMM_WORLD);

    error(iprms[0] != nsm, 1, "dfl_update [dfl_modes.c]",
          "Parameters are not global");
  }

  status[0] = 0;
  ifail = set_frame();
  mds = reserve_ws(Ns);
  restore_fields(mds);

#ifdef DFL_MODES_DBG
  if (my_rank == 0) {
    printf("Progress report [program dfl_update]:\n\n");
    printf("nsm = %d\n", nsm);
    printf("Ns = %d, ninv = %d, nmr = %d, ncy = %d\n", Ns, dgn.ninv, dgn.nmr,
           dgn.ncy);
    printf("nkv = %d, nmx = %d, res = %.1e, ifail = %d\n\n", dpr.nkv, dpr.nmx,
           dpr.res, ifail);
  }
#endif

  if (ifail) {
    status[0] = -2;
  } else {
    for (n = 0; n < nsm; n++) {
      ifail = set_Awhat(dgn.mu);

      if (ifail) {
        status[0] = -3;
        break;
      } else {
        dfl_smooth_fields(mds, status);
        nrn += 1;

        if (status[0] < 0) {
          break;
        }

        if ((nrn > 3) && (n < (nsm - 1))) {
          renormalize_fields(mds);
        }

        dfl_subspace(mds);

#ifdef DFL_MODES_DBG
        print_res(mds);
#endif
      }
    }
  }

  if (status[0] > 0) {
    n = Ns * nsm;
    status[0] = (status[0] + n / 2) / n;
  }

  release_ws();
  set_sw_parms(m0);
  if (eoflg != 1) {
    set_tm_parms(eoflg);
  }

#ifdef DFL_MODES_DBG
  if (my_rank == 0) {
    printf("status = %d\n", status[0]);
    printf("dfl_update: all done\n\n");
    fflush(stdout);
  }
#endif
}

void dfl_modes2(int *status)
{
  dfl_modes(status);

  if (status[0] == -3) {
#ifdef DFL_MODES_DBG
    if (my_rank == 0) {
      printf("Generation of deflation subspace failed\n");
      printf("Start second attempt\n");
      fflush(stdout);
    }
#endif

    dfl_modes(status + 1);
  } else {
    status[1] = 0;
  }
}

void dfl_update2(int nsm, int *status)
{
  dfl_update(nsm, status);

  if (status[0] == -3) {
#ifdef DFL_MODES_DBG
    if (my_rank == 0) {
      printf("Update of deflation subspace failed\n");
      printf("Attempt to regenerate subspace\n");
      fflush(stdout);
    }
#endif

    dfl_modes(status + 1);
  } else {
    status[1] = 0;
  }
}
