
/*******************************************************************************
 *
 * File genfrc.c
 *
 * Copyright (C) 2006, 2011, 2013 Martin Luescher, Stefan Schaefer
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Calculation of quark forces.
 *
 * The externally accessible functions are
 *
 *   void sw_frc(double c)
 *     Computes the SW part of the quark force, using the current value
 *     of the X tensor field (see the notes). The calculated force is then
 *     multiplied by c and added to the MD force field.
 *
 *   void hop_frc(double c)
 *     Computes the hopping part of the quark force, using the current
 *     value of the X vector field (see the notes). The calculated force
 *     is then multiplied by c and added to the MD force field.
 *
 * Notes:
 *
 * The computation of the quark forces is described in the notes
 *
 *  M. Luescher: "Molecular-dynamics quark forces" (January 2012)
 *
 * For explanations of the X tensor and vector fields, see xtensor.c and
 * frcfcts.c. The MD force field is the one returned by the program mdflds()
 * (see mdflds/mdflds.c).
 *
 * If the X tensor field is obtained from the SW term calculated by sw_term(),
 * and if the X vector field is obtained from quark fields vanishing at the
 * boundaries of the lattice, as required by the chosen boundary conditions,
 * the programs sw_frc() and hop_frc() leave the force field on the static
 * links unchanged.
 *
 * The coefficient csw of the SW term is retrieved from the parameter data
 * base (flags/lat_parms.c). The programs in this module perform global
 * operations and must be called simultaneously on all MPI processes.
 *
 *******************************************************************************/

#define GENFRC_C
#define OPENQCD_INTERNAL

#include "forces.h"
#include "global.h"
#include "lattice.h"
#include "linalg.h"
#include "mdflds.h"
#include "mpi.h"
#include "sflds.h"
#include "su3fcts.h"
#include "tcharge.h"
#include "uflds.h"

#define N0 (NPROC0 * L0)

static su3_dble w[6] ALIGNED16;
static su3_alg_dble Y[8] ALIGNED16;

void sw_frc(double c)
{
  int bc, n, ix, t;
  int ipu[4], ipx[4];
  u3_alg_dble **xt, *Xb;
  su3_alg_dble *fb, *fr;
  su3_dble *ub;
  mdflds_t *mdfs;
  sw_parms_t swp;

  ani_params_t ani;
  double ct, cs, c_ani;

  ani = ani_parms();

  ct = ani.cT /
       (ani.us_fermion * ani.us_fermion * ani.ut_fermion * ani.ut_fermion);
  cs = ani.cR / ani.xi * 1.0 /
       (ani.us_fermion * ani.us_fermion * ani.us_fermion * ani.ut_fermion);

  bc = bc_type();
  swp = sw_parms();
  c *= 0.0625 * swp.csw;

  mdfs = mdflds();
  fb = (*mdfs).frc;
  set_alg2zero(7 * (BNDRY / 4), fb + 4 * VOLUME);

  if (query_flags(UDBUF_UP2DATE) != 1) {
    copy_bnd_ud();
  }
  ub = udfld();
  xt = xtensor();

  for (n = 0; n < 6; n++) {
    Xb = xt[n];
    copy_bnd_ft(n, Xb);

    if (n < 3) {
      c_ani = c * ct;
    } else {
      c_ani = c * cs;
    }

    for (ix = 0; ix < VOLUME; ix++) {
      t = global_time(ix);

      plaq_uidx(n, ix, ipu);
      plaq_ftidx(n, ix, ipx);

      su3dagxsu3(ub + ipu[2], ub + ipu[0], w);
      su3xsu3dag(ub + ipu[1], ub + ipu[3], w + 1);
      u3algxsu3(Xb + ipx[0], ub + ipu[0], w + 2);
      su3dagxsu3(ub + ipu[2], w + 2, w + 2);
      u3algxsu3(Xb + ipx[2], w, w + 3);
      u3algxsu3dag(Xb + ipx[3], ub + ipu[3], w + 4);
      su3xsu3(ub + ipu[1], w + 4, w + 4);
      su3xu3alg(w, Xb + ipx[1], w + 5);

      prod2su3alg(w + 1, w + 2, Y);
      prod2su3alg(w + 2, w + 1, Y + 1);
      prod2su3alg(w + 1, w + 3, Y + 2);
      prod2su3alg(w + 3, w + 1, Y + 3);
      prod2su3alg(w + 4, w, Y + 4);
      prod2su3alg(w, w + 4, Y + 5);
      prod2su3alg(w + 1, w + 5, Y + 6);
      prod2su3alg(w + 5, w + 1, Y + 7);

      _su3_alg_add_assign(Y[0], Y[2]);
      _su3_alg_add_assign(Y[0], Y[4]);
      _su3_alg_add_assign(Y[6], Y[0]);

      _su3_alg_add_assign(Y[1], Y[5]);
      _su3_alg_add_assign(Y[1], Y[7]);
      _su3_alg_add_assign(Y[3], Y[1]);

      rotate_su3alg(ub + ipu[0], Y);
      rotate_su3alg(ub + ipu[0], Y + 2);
      rotate_su3alg(ub + ipu[2], Y + 1);
      rotate_su3alg(ub + ipu[2], Y + 7);

      _su3_alg_add_assign(Y[0], Y[7]);
      _su3_alg_add_assign(Y[1], Y[2]);

      fr = fb + ipu[0];
      _su3_alg_mul_add_assign(*fr, c_ani, Y[0]);

      if ((n >= 3) || (bc == 0) || (bc == 3)) {
        fr = fb + ipu[1];
        _su3_alg_mul_add_assign(*fr, c_ani, Y[6]);
        fr = fb + ipu[2];
        _su3_alg_mul_sub_assign(*fr, c_ani, Y[1]);
      } else {
        if (t < (N0 - 1)) {
          fr = fb + ipu[1];
          _su3_alg_mul_add_assign(*fr, c_ani, Y[6]);
        }

        if ((t > 0) || (bc == 2)) {
          fr = fb + ipu[2];
          _su3_alg_mul_sub_assign(*fr, c_ani, Y[1]);
        }
      }

      fr = fb + ipu[3];
      _su3_alg_mul_sub_assign(*fr, c_ani, Y[3]);
    }
  }

  add_bnd_frc();
}

void hop_frc(double c)
{
  su3_alg_dble *fr;
  su3_dble *xv, *u, *um;
  mdflds_t *mdfs;

  xv = xvector();
  mdfs = mdflds();
  fr = (*mdfs).frc;

  u = udfld();
  um = u + 4 * VOLUME;
  c *= -0.5;

  apply_ani_ud();

  for (; u < um; u++) {
    prod2su3alg(u, xv, Y);
    _su3_alg_mul_add_assign(*fr, c, *Y);

    xv += 1;
    fr += 1;
  }

  remove_ani_ud();
}
