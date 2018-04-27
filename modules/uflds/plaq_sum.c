
/*******************************************************************************
 *
 * File plaq_sum.c
 *
 * Copyright (C) 2005, 2011-2013, 2016 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Calculation of plaquette sums.
 *
 * The externally accessible functions are
 *
 *   void plaq_sum_split_dble(int icom, double *result)
 *     Computes the sum of Re[tr{U(p)}] over all unoriented plaquettes p, where
 *     U(p) is the product of the double-precision link variables around p. If
 *     icom=1 the global sum of the local sums is returned and otherwise just
 *     the local sum. The calculation is stored in result in which the first
 *     element holds the sum of plaquettes in which temporal links enter and the
 *     second element has the purely spatial links.
 *
 *   double plaq_sum_dble(int icom)
 *     Returns the sum of the first and second element from calling
 *     plaq_sum_split_dble.
 *
 *   void plaq_wsum_split_dble(int icom, double *result)
 *     Same as plaq_sum_splut_dble(), but giving weight 1/2 to the contribution
 *     of the space-like plaquettes at the boundaries of the lattice if boundary
 *     conditions of type 0, 1 or 2 are chosen.
 *
 *   double plaq_wsum_dble(int icom)
 *     Returns the sum of the first and second element from calling
 *     plaq_wsum_sum_split_dble.
 *
 *   double plaq_action_slices(double *asl)
 *     Computes the time-slice sums asl[x0] of the tree-level O(a)-improved
 *     plaquette action density of the double-precision gauge field. The
 *     factor 1/g0^2 is omitted and the time x0 runs from 0 to NPROC0*L0-1.
 *     The program returns the total action.
 *
 *   double spatial_link_sum(int icom)
 *     Returns the sum of the real trace of all gauge links pointing in a
 *     spatial direction.
 *
 *   double temporal_link_sum(int icom)
 *     Returns the sum of the real trace of all gauge links pointing in a
 *     temporal direction.
 *
 * Notes:
 *
 * The Wilson plaquette action density is defined so that it converges to the
 * Yang-Mills action in the classical continuum limit with a rate proportional
 * to a^2. In particular, at the boundaries of the lattice (if there are any),
 * the space-like plaquettes are given the weight 1/2 and the contribution of
 * a plaquette p in the bulk is 2*Re[tr{1-U(p)}].
 *
 * The time-slice sum asl[x0] computed by plaq_action_slices() includes the
 * full contribution to the action of the space-like plaquettes at time x0 and
 * 1/2 of the contribution of the time-like plaquettes at time x0 and x0-1.
 *
 * The programs in this module perform global communications and must be
 * called simultaneously on all MPI processes.
 *
 *******************************************************************************/

#define PLAQ_SUM_C
#define OPENQCD_INTERNAL

#include "global.h"
#include "lattice.h"
#include "mpi.h"
#include "su3fcts.h"
#include "uflds.h"

#define N0 (NPROC0 * L0)

static su3_dble *udb;
static su3_dble wd1 ALIGNED16;
static su3_dble wd2 ALIGNED16;

#if defined(LIBRARY)

static int isA[2], *isE, *isB, init = 0;
static double *aslE, *aslB;

static void allocate_buffers(void)
{
  isE = malloc(L0 * sizeof(*isE));
  isB = malloc(L0 * sizeof(*isB));

  aslE = amalloc(N0 * sizeof(*aslE), ALIGN);
  aslB = amalloc(N0 * sizeof(*aslB), ALIGN);
}

#else

static int isA[2], isE[L0], isB[L0], init = 0;
static double aslE[N0], aslB[N0];

static void allocate_buffers(void) {}

#endif

static double real_trace(su3_dble const *u)
{
  return u->c11.re + u->c22.re + u->c33.re;
}

static double plaq_dble(int n, int ix)
{
  int ip[4];
  double sm;

  plaq_uidx(n, ix, ip);

  su3xsu3(udb + ip[0], udb + ip[1], &wd1);
  su3dagxsu3dag(udb + ip[3], udb + ip[2], &wd2);
  cm3x3_retr(&wd1, &wd2, &sm);

  return sm;
}

static void local_plaq_sum_dble(int iw)
{
  int bc, ix, t, n;
  double wp, pat, pas;

  if (init < 1) {
    allocate_buffers();

    isA[0] = init_hsum(1);
    isA[1] = init_hsum(1);
    init = 1;
  }

  bc = bc_type();

  if (iw == 0) {
    wp = 1.0;
  } else {
    wp = 0.5;
  }

  udb = udfld();
  reset_hsum(isA[0]);
  reset_hsum(isA[1]);

  for (ix = 0; ix < VOLUME; ix++) {
    t = global_time(ix);
    pat = 0.0;
    pas = 0.0;

    if ((t < (N0 - 1)) || (bc != 0)) {
      for (n = 0; n < 3; n++) {
        pat += plaq_dble(n, ix);
      }
    }

    if (((t > 0) || (bc == 3)) && ((t < (N0 - 1)) || (bc != 0))) {
      for (n = 3; n < 6; n++) {
        pas += plaq_dble(n, ix);
      }
    } else {
      for (n = 3; n < 6; n++) {
        pas += wp * plaq_dble(n, ix);
      }
    }

    if ((t == (N0 - 1)) && ((bc == 1) || (bc == 2))) {
      pat += 9.0 * wp;
    }

    if (not_equal_d(pat, 0.0)) {
      add_to_hsum(isA[0], &pat);
    }

    if (not_equal_d(pas, 0.0)) {
      add_to_hsum(isA[1], &pas);
    }
  }
}

void plaq_sum_split_dble(int icom, double *result)
{
  if (query_flags(UDBUF_UP2DATE) != 1) {
    copy_bnd_ud();
  }

  local_plaq_sum_dble(0);

  if ((icom == 1) && (NPROC > 1)) {
    global_hsum(isA[0], result);
    global_hsum(isA[1], result + 1);
  } else {
    local_hsum(isA[0], result);
    local_hsum(isA[1], result + 1);
  }
}

double plaq_sum_dble(int icom)
{
  double p[2];

  plaq_sum_split_dble(icom, p);

  return p[0] + p[1];
}

void plaq_wsum_split_dble(int icom, double *result)
{
  if (query_flags(UDBUF_UP2DATE) != 1) {
    copy_bnd_ud();
  }

  local_plaq_sum_dble(1);

  if ((icom == 1) && (NPROC > 1)) {
    global_hsum(isA[0], result);
    global_hsum(isA[1], result + 1);
  } else {
    local_hsum(isA[0], result);
    local_hsum(isA[1], result + 1);
  }
}

double plaq_wsum_dble(int icom)
{
  double p[2];

  plaq_wsum_split_dble(icom, p);

  return p[0] + p[1];
}

double plaq_action_slices(double *asl)
{
  int bc, ix, t, t0, n;
  double A, smE, smB;

  if (init < 2) {
    if (init < 1) {
      allocate_buffers();
      isA[0] = init_hsum(1);
    }

    for (t = 0; t < L0; t++) {
      isE[t] = init_hsum(1);
      isB[t] = init_hsum(1);
    }

    init = 2;
  }

  if (query_flags(UDBUF_UP2DATE) != 1) {
    copy_bnd_ud();
  }

  bc = bc_type();
  t0 = cpr[0] * L0;
  udb = udfld();

  for (t = 0; t < L0; t++) {
    reset_hsum(isE[t]);
    reset_hsum(isB[t]);
  }

  for (ix = 0; ix < VOLUME; ix++) {
    t = global_time(ix);
    smE = 0.0;
    smB = 0.0;

    if ((t < (N0 - 1)) || (bc != 0)) {
      for (n = 0; n < 3; n++) {
        smE += (3.0 - plaq_dble(n, ix));
      }
    }

    if ((t > 0) || (bc != 1)) {
      for (n = 3; n < 6; n++) {
        smB += (3.0 - plaq_dble(n, ix));
      }
    }

    t -= t0;

    if (not_equal_d(smE, 0.0)) {
      add_to_hsum(isE[t], &smE);
    }
    if (not_equal_d(smB, 0.0)) {
      add_to_hsum(isB[t], &smB);
    }
  }

  for (t = 0; t < N0; t++) {
    asl[t] = 0.0;
  }

  for (t = 0; t < L0; t++) {
    local_hsum(isE[t], &smE);
    asl[t + t0] = smE;
  }

  if (NPROC > 1) {
    MPI_Reduce(asl, aslE, N0, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(aslE, N0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    for (t = 0; t < N0; t++) {
      aslE[t] = asl[t];
    }
  }

  for (t = 0; t < N0; t++) {
    asl[t] = 0.0;
  }

  for (t = 0; t < L0; t++) {
    local_hsum(isB[t], &smB);
    asl[t + t0] = smB;
  }

  if (NPROC > 1) {
    MPI_Reduce(asl, aslB, N0, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(aslB, N0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    for (t = 0; t < N0; t++) {
      aslB[t] = asl[t];
    }
  }

  if (bc != 3) {
    asl[0] = aslE[0] + aslB[0];
  } else {
    asl[0] = aslE[0] + aslE[N0 - 1] + 2.0 * aslB[0];
  }

  if (bc == 0) {
    for (t = 1; t < (N0 - 1); t++) {
      asl[t] = aslE[t - 1] + aslE[t] + 2.0 * aslB[t];
    }

    asl[N0 - 1] = aslE[N0 - 2] + aslB[N0 - 1];
  } else {
    for (t = 1; t < N0; t++) {
      asl[t] = aslE[t - 1] + aslE[t] + 2.0 * aslB[t];
    }
  }

  reset_hsum(isA[0]);

  if ((bc == 1) || (bc == 2)) {
    add_to_hsum(isA[0], aslE + N0 - 1);
  }

  for (t = 0; t < N0; t++) {
    add_to_hsum(isA[0], asl + t);
  }

  local_hsum(isA[0], &A);

  return A;
}

double spatial_link_sum(int icom)
{
  int ix, mu;
  double local_link_sum, total_link_sum;
  su3_dble const *ufld;

  ufld = udfld();
  local_link_sum = 0.0;

  for (ix = VOLUME / 2; ix < VOLUME; ++ix) {
    for (mu = 1; mu < 4; ++mu) {
      local_link_sum += real_trace(ufld + 8 * (ix - VOLUME / 2) + 2 * mu);
      local_link_sum += real_trace(ufld + 8 * (ix - VOLUME / 2) + 2 * mu + 1);
    }
  }

  if ((NPROC > 1) && (icom == 1)) {
    MPI_Reduce(&local_link_sum, &total_link_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Bcast(&total_link_sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    local_link_sum = total_link_sum;
  }

  return local_link_sum;
}

double temporal_link_sum(int icom)
{
  int ix;
  double local_link_sum, total_link_sum;
  su3_dble const *ufld;

  ufld = udfld();
  local_link_sum = 0.0;

  for (ix = VOLUME / 2; ix < VOLUME; ++ix) {
    local_link_sum += real_trace(ufld + 8 * (ix - VOLUME / 2));
    local_link_sum += real_trace(ufld + 8 * (ix - VOLUME / 2) + 1);
  }

  if ((NPROC > 1) && (icom == 1)) {
    MPI_Reduce(&local_link_sum, &total_link_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Bcast(&total_link_sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    local_link_sum = total_link_sum;
  }

  return local_link_sum;
}
