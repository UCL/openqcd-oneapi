
/*******************************************************************************
 *
 * File bcnds.c
 *
 * Copyright (C) 2005, 2010-2014 Martin Luescher, John Bulava
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Programs related to the boundary conditions in the time direction.
 *
 *   int *bnd_lks(int *n)
 *     Returns the starting address of an array of length n whose elements
 *     are the integer offsets of the time-like link variables on the local
 *     lattice at global time NPROC0*L0-1.
 *
 *   int *bnd_pts(int *n)
 *     Returns the starting address of an array of length n whose elements
 *     are the indices of the points on the local lattice at global time 0
 *     (boundary conditions type 0,1 or 2) and time NPROC0*L0-1 (boundary
 *     conditions type 0). The ordering of the indices is such that the n/2
 *     even points come first.
 *
 *   void set_bc(void)
 *     Sets the double-precision link variables at time 0 and T to the
 *     values required by the chosen boundary conditions (see the notes).
 *
 *   int check_bc(double tol)
 *     Returns 1 if the double-precision gauge field has the proper boundary
 *     values and if no active link variables are equal to zero. Otherwise
 *     the program returns 0. The parameter tol>=0.0 sets an upper bound on
 *     the tolerated difference of the boundary values of the gauge field from
 *     the expected ones in the case of SF and open-SF boundary conditions.
 *
 *   void bnd_s2zero(ptset_t set, spinor *s)
 *     Sets the components of the single-precision spinor field s on the
 *     specified set of points at global time 0 (boundary conditions type
 *     0,1 or 2) and time NPROC0*L0-1 (boundary conditions type 0) to zero.
 *
 *   void bnd_sd2zero(ptset_t set, spinor_dble *sd)
 *     Sets the components of the double-precision spinor field sd on the
 *     specified set of points at global time 0 (boundary conditions type
 *     0,1 or 2) and time NPROC0*L0-1 (boundary conditions type 0) to zero.
 *
 * Notes:
 *
 * The time extent T of the lattice is
 *
 *  NPROC0*L0-1      for open boundary conditions,
 *
 *  NPROC0*L0        for SF, open-SF and periodic boundary conditions.
 *
 * Note that in the latter cases the points at time T are not in the local
 * lattice and are omitted in the programs bnd_pts(), bnd_s2zero() and
 * bnd_sd2zero().
 *
 * The action performed by set_bc() is the following:
 *
 *  Open bc:         Set all link variables U(x,0) at time T to zero.
 *
 *  SF bc:           Reads the boundary values of the gauge field from the
 *                   data base and assigns them to the link variables at
 *                   time 0 and T. At time T the link variables are stored
 *                   in the buffers appended to the local field on the MPI
 *                   processes where cpr[0]=NPROC0-1.
 *
 *  Open-SF bc:      Same as SF bc, but omitting the assignment of the link
 *                   variables at time 0.
 *
 *  Periodic bc:     No action is performed.
 *
 * Then the program checks whether any active link variables are equal to
 * zero and, if some are found, aborts the program with an error message.
 * An error occurs if set_bc() or check_bc() is called when the gauge field
 * is phase-set (see set_ud_phase() [uflds.c]).
 *
 * The programs in this module act globally and must be called simultaneously
 * on all MPI processes. After the first time, the programs bnd_s2zero() and
 * bnd_sd2zero() may be locally called.
 *
 *******************************************************************************/

#define BCNDS_C
#define OPENQCD_INTERNAL

#include "global.h"
#include "lattice.h"
#include "mpi.h"
#include "uflds.h"

#define N0 (NPROC0 * L0)

typedef union
{
  su3_dble u;
  double r[18];
} umat_t;

static int init0 = 0, nlks, *lks, nbclks, *bclks;
static int init1 = 0, npts, *pts;
static int init2 = 0;
static const su3_dble ud0 = {{0.0}};
static const spinor s0 = {{{0.0f}}};
static const spinor_dble sd0 = {{{0.0}}};
static su3_dble ubnd[2][3];

static void alloc_lks(void)
{
  int ix, iy, t, bc, *lk;
  int pidx[4];
  bc_parms_t bcp;

  error(iup[0][0] == 0, 1, "alloc_lks [bcnds.c]",
        "Geometry arrays are not set");

  alloc_uidx();
  bcp = bc_parms();
  bc = bcp.type;

  if ((cpr[0] == 0) || (cpr[0] == (NPROC0 - 1))) {
    nlks = (L1 * L2 * L3) / 2;
    nbclks = 0;

    if (cpr[0] == (NPROC0 - 1)) {
      nlks += (NPROC0 == 1) * (L1 * L2 * L3 / 2);
      nbclks += (FACE0 > 0) * (L1 * L2 * L3 / 2);
      nbclks += (FACE1 > 0) * L2 * L3;
      nbclks += (FACE2 > 0) * L1 * L3;
      nbclks += (FACE3 > 0) * L1 * L2;
    }

    lks = malloc(nlks * sizeof(*lks));

    if (lks != NULL) {
      lk = lks;

      for (ix = (VOLUME / 2); ix < VOLUME; ix++) {
        t = global_time(ix);

        if (t == 0) {
          (*lk) = 8 * (ix - (VOLUME / 2)) + 1;
          lk += 1;
        } else if (t == (N0 - 1)) {
          (*lk) = 8 * (ix - (VOLUME / 2));
          lk += 1;
        }
      }
    }

    if ((bc == 3) && (nbclks != 0)) {
      bclks = malloc(nbclks * sizeof(*bclks));
    } else {
      nbclks = 0;
    }

    /* Links on the boundary that crosses the temporal boundary */
    if ((bc == 3) && (nbclks != 0) && (bclks != NULL)) {

      lk = bclks;

      /* Type 1 links at FACE0 */
      for (ix = 0; ix < FACE0 / 2; ix++) {
        (*lk) = 4 * VOLUME + ix;
        lk += 1;
      }

      /* Type 2 links at the spatial faces pointing in t-dir at the time
       * boundary */

      /* Type 2 links at FACE1 */
      if (FACE1 != 0) {
        for (ix = 0; ix < L2; ++ix) {
          for (iy = 0; iy < L3; ++iy) {
            plaq_uidx(0, ipt[iy + L3 * (ix + L2 * (L1 - 1 + L1 * (L0 - 1)))],
                      pidx);
            (*lk) = pidx[3];
            lk += 1;
          }
        }
      }

      /* Type 2 links at FACE2 */
      if (FACE2 != 0) {
        for (ix = 0; ix < L1; ++ix) {
          for (iy = 0; iy < L3; ++iy) {
            plaq_uidx(1, ipt[iy + L3 * ((L2 - 1) + L2 * (ix + L1 * (L0 - 1)))],
                      pidx);
            (*lk) = pidx[3];
            lk += 1;
          }
        }
      }

      /* Type 2 links at FACE3 */
      if (FACE3 != 0) {
        for (ix = 0; ix < L1; ++ix) {
          for (iy = 0; iy < L2; ++iy) {
            plaq_uidx(2, ipt[(L3 - 1) + L3 * (iy + L2 * (ix + L1 * (L0 - 1)))],
                      pidx);
            (*lk) = pidx[3];
            lk += 1;
          }
        }
      }
    }
  } else {
    nlks = 0;
    nbclks = 0;
    lks = NULL;
    bclks = NULL;
  }

  error((nlks > 0) && (lks == NULL), 1, "alloc_lks [bcnds.c]",
        "Unable to allocate index array for bulk links");
  error((bc == 3) && (nbclks > 0) && (bclks == NULL), 1, "alloc_lks [bcnds.c]",
        "Unable to allocate index array for halo links");

  init0 = 1;
}

static void alloc_pts(void)
{
  int bc, ix, t, *pt;

  error(iup[0][0] == 0, 1, "alloc_pts [bcnds.c]",
        "Geometry arrays are not set");
  bc = bc_type();

  if (((cpr[0] == 0) && (bc != 3)) || ((cpr[0] == (NPROC0 - 1)) && (bc == 0))) {
    if ((NPROC0 == 1) && (bc == 0)) {
      npts = 2 * L1 * L2 * L3;
    } else {
      npts = L1 * L2 * L3;
    }

    pts = malloc(npts * sizeof(*pts));

    if (pts != NULL) {
      pt = pts;

      for (ix = 0; ix < VOLUME; ix++) {
        t = global_time(ix);

        if ((t == 0) || ((t == (N0 - 1)) && (bc == 0))) {
          (*pt) = ix;
          pt += 1;
        }
      }
    }
  } else {
    npts = 0;
    pts = NULL;
  }

  error((npts > 0) && (pts == NULL), 1, "alloc_pts [bcnds.c]",
        "Unable to allocate index array");
  init1 = 1;
}

int *bnd_lks(int *n)
{
  if (init0 == 0) {
    alloc_lks();
  }

  (*n) = nlks;

  return lks;
}

int *bnd_bnd_lks(int *n)
{
  if (init0 == 0) {
    alloc_lks();
  }

  (*n) = nbclks;

  return bclks;
}

int *bnd_pts(int *n)
{
  if (init1 == 0) {
    alloc_pts();
  }

  (*n) = npts;

  return pts;
}

static int is_zero(su3_dble const *u)
{
  int i, it;
  umat_t *um;

  um = (umat_t *)(u);
  it = 1;

  for (i = 0; i < 18; i++) {
    it &= is_equal_d((*um).r[i], 0.0);
  }

  return it;
}

static int is_equal(double tol, su3_dble const *u, su3_dble const *v)
{
  int i, it;
  umat_t const *um, *vm;

  um = (umat_t *)(u);
  vm = (umat_t *)(v);
  it = 1;

  for (i = 0; i < 18; i++) {
    it &= (fabs((*um).r[i] - (*vm).r[i]) <= tol);
  }

  return it;
}

static int check_zero(int bc)
{
  int it, ix, t, ifc;
  su3_dble *u;

  it = 1;
  u = udfld();

  for (ix = (VOLUME / 2); ix < VOLUME; ix++) {
    t = global_time(ix);

    if ((bc == 0) && (t == 0)) {
      it &= (0x1 ^ is_zero(u));
      u += 1;
      it &= is_zero(u);
      u += 1;
    } else if ((bc == 0) && (t == (N0 - 1))) {
      it &= is_zero(u);
      u += 1;
      it &= (0x1 ^ is_zero(u));
      u += 1;
    } else {
      it &= (0x1 ^ is_zero(u));
      u += 1;
      it &= (0x1 ^ is_zero(u));
      u += 1;
    }

    for (ifc = 2; ifc < 8; ifc++) {
      it &= (0x1 ^ is_zero(u));
      u += 1;
    }
  }

  return it;
}

static void set_ubnd(void)
{
  int i, k;
  double s[3];
  bc_parms_t bcp;

  bcp = bc_parms();
  s[0] = (double)(NPROC1 * L1);
  s[1] = (double)(NPROC2 * L2);
  s[2] = (double)(NPROC3 * L3);

  for (i = 0; i < 2; i++) {
    for (k = 0; k < 3; k++) {
      ubnd[i][k] = ud0;
      ubnd[i][k].c11.re = cos(bcp.phi[i][0] / s[k]);
      ubnd[i][k].c11.im = sin(bcp.phi[i][0] / s[k]);
      ubnd[i][k].c22.re = cos(bcp.phi[i][1] / s[k]);
      ubnd[i][k].c22.im = sin(bcp.phi[i][1] / s[k]);
      ubnd[i][k].c33.re = cos(bcp.phi[i][2] / s[k]);
      ubnd[i][k].c33.im = sin(bcp.phi[i][2] / s[k]);
    }
  }

  init2 = 1;
}

static void open_bc(void)
{
  int *lk, *lkm;
  su3_dble *ub;

  if (init0 == 0) {
    alloc_lks();
  }

  ub = udfld();
  lk = lks;
  lkm = lk + nlks;

  for (; lk < lkm; lk++) {
    ub[*lk] = ud0;
  }

  set_flags(UPDATED_UD);
}

static void SF_bc(void)
{
  int k, *pt, *ptm;
  su3_dble *ub, *u;

  if (init1 == 0) {
    alloc_pts();
  }
  if (init2 == 0) {
    set_ubnd();
  }

  ub = udfld();

  if (cpr[0] == 0) {
    pt = pts + (npts / 2);
    ptm = pts + npts;

    for (; pt < ptm; pt++) {
      u = ub + 8 * (pt[0] - (VOLUME / 2));

      for (k = 0; k < 3; k++) {
        u[2 + 2 * k] = ubnd[0][k];
        u[3 + 2 * k] = ubnd[0][k];
      }
    }
  }

  if (cpr[0] == (NPROC0 - 1)) {
    u = ub + 4 * VOLUME + 7 * (BNDRY / 4);

    for (k = 0; k < 3; k++) {
      u[k] = ubnd[1][k];
    }
  }

  set_flags(UPDATED_UD);
}

static void openSF_bc(void)
{
  int k;
  su3_dble *ub, *u;

  if (init2 == 0) {
    set_ubnd();
  }

  ub = udfld();

  if (cpr[0] == (NPROC0 - 1)) {
    u = ub + 4 * VOLUME + 7 * (BNDRY / 4);

    for (k = 0; k < 3; k++) {
      u[k] = ubnd[1][k];
    }
  }

  set_flags(UPDATED_UD);
}

void set_bc(void)
{
  int bc, it;

  error_root(query_flags(UD_IS_CLEAN) == 0, 1, "set_bc [bcnds.c]",
             "Gauge configuration must not be modified (phase|smearing)");
  bc = bc_type();

  if (bc == 0) {
    open_bc();
  } else if (bc == 1) {
    SF_bc();
  } else if (bc == 2) {
    openSF_bc();
  }

  it = check_zero(bc);
  error(it != 1, 1, "set_bc [bcnds.c]",
        "Link variables vanish on an incorrect set of links");
}

static int check_SF(double tol)
{
  int it, k, *pt, *ptm;
  su3_dble *ub, *u;

  if (init1 == 0) {
    alloc_pts();
  }
  if (init2 == 0) {
    set_ubnd();
  }

  it = 1;
  ub = udfld();

  if (cpr[0] == 0) {
    pt = pts + (npts / 2);
    ptm = pts + npts;

    for (; pt < ptm; pt++) {
      u = ub + 8 * (pt[0] - (VOLUME / 2));

      for (k = 0; k < 3; k++) {
        it &= is_equal(tol, u + 2 + 2 * k, ubnd[0] + k);
        it &= is_equal(tol, u + 3 + 2 * k, ubnd[0] + k);
      }
    }
  }

  if (cpr[0] == (NPROC0 - 1)) {
    u = ub + 4 * VOLUME + 7 * (BNDRY / 4);

    for (k = 0; k < 3; k++) {
      it &= is_equal(tol, u + k, ubnd[1] + k);
    }
  }

  return it;
}

static int check_openSF(double tol)
{
  int it, k;
  su3_dble *ub, *u;

  if (init2 == 0) {
    set_ubnd();
  }

  it = 1;
  ub = udfld();

  if (cpr[0] == (NPROC0 - 1)) {
    u = ub + 4 * VOLUME + 7 * (BNDRY / 4);

    for (k = 0; k < 3; k++) {
      it &= is_equal(tol, u + k, ubnd[1] + k);
    }
  }

  return it;
}

int check_bc(double tol)
{
  int bc, it, is;
  double dprms[1];

  error_root(query_flags(UD_IS_CLEAN) == 0, 1, "check_bc [bcnds.c]",
             "Gauge configuration must not be modified (phase|smearing)");

  if (NPROC > 1) {
    dprms[0] = tol;
    MPI_Bcast(dprms, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    error(dprms[0] != tol, 1, "check_bc [bcnds.c]", "Parameter is not global");
  }

  bc = bc_type();
  it = check_zero(bc);

  if (bc == 1) {
    it &= check_SF(tol);
  } else if (bc == 2) {
    it &= check_openSF(tol);
  }

  if (NPROC > 1) {
    is = it;
    MPI_Allreduce(&is, &it, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  }

  return it;
}

void bnd_s2zero(ptset_t set, spinor *s)
{
  int *pt, *pm;

  if (init1 == 0) {
    alloc_pts();
  }

  if (npts > 0) {
    if (set == ALL_PTS) {
      pt = pts;
      pm = pts + npts;
    } else if (set == EVEN_PTS) {
      pt = pts;
      pm = pts + npts / 2;
    } else if (set == ODD_PTS) {
      pt = pts + npts / 2;
      pm = pts + npts;
    } else {
      return;
    }

    for (; pt < pm; pt++) {
      s[*pt] = s0;
    }
  }
}

void bnd_sd2zero(ptset_t set, spinor_dble *sd)
{
  int *pt, *pm;

  if (init1 == 0) {
    alloc_pts();
  }

  if (npts > 0) {
    if (set == ALL_PTS) {
      pt = pts;
      pm = pts + npts;
    } else if (set == EVEN_PTS) {
      pt = pts;
      pm = pts + npts / 2;
    } else if (set == ODD_PTS) {
      pt = pts + npts / 2;
      pm = pts + npts;
    } else {
      return;
    }

    for (; pt < pm; pt++) {
      sd[*pt] = sd0;
    }
  }
}
