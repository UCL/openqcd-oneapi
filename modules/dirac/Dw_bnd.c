/*******************************************************************************
 *
 * File Dw_bnd.c
 *
 * Copyright (C) 2005, 2011, 2013 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Block boundary part of the Wilson-Dirac operator.
 *
 * The externally accessible function is
 *
 *   void Dw_bnd(blk_grid_t grid,int n,int k,int l)
 *     Applies the boundary part of the Wilson-Dirac operator to the field
 *     b.s[k] on the n'th block b of the specified block grid and assigns
 *     the result to the field bb.w[l] on the boundary bb of the block.
 *
 * Notes:
 *
 * The boundary part of the Wilson-Dirac operator on a block is the sum of
 * the hopping terms that lead from the block to its exterior boundary. If
 * the faces of the block in the -0,+0,...,-3,+3 directions are labeled by
 * an integer ifc=0,..,7, the Dirac spinors psi computed by Dw_bnd() along
 * the boundary number ifc satisfy
 *
 *   theta[ifc]*psi=0
 *
 * (see sflds/Pbnd.c for the definition of the projectors theta[ifc]). The
 * program Dw_bnd() assigns the upper two components of psi to the Weyl
 * fields on the boundaries of the block.
 *
 * The input field is not changed except possibly at the points at global
 * time 0 and NPROC0*L0-1, where it is set to zero if so required by the
 * chosen boundary conditions. In the case of boundary conditions of type
 * 0,1 and 2, the output field is set to zero at the exterior boundaries
 * of the lattice at time -1 and NPROC0*L0.
 *
 * The program in this module does not perform any communications and can be
 * called locally.
 *
 *******************************************************************************/

#define DW_BND_C

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "su3.h"
#include "flags.h"
#include "utils.h"
#include "block.h"
#include "dirac.h"
#include "global.h"

static weyl chi;
static const weyl w0 = {{{0.0f}}};
static const spinor s0 = {{{0.0f}}};

static void mul_umat(weyl *s, su3 *u, weyl *r)
{
  _su3_multiply((*r).c1, *u, (*s).c1);
  _su3_multiply((*r).c2, *u, (*s).c2);
}

static void mul_uinv(weyl *s, su3 *u, weyl *r)
{
  _su3_inverse_multiply((*r).c1, *u, (*s).c1);
  _su3_inverse_multiply((*r).c2, *u, (*s).c2);
}

void Dw_bnd(blk_grid_t grid, int n, int k, int l)
{
  int bc, nb, isw, *ipp;
  float moh;
  float one_over_gammaf;
  su3 *u;
  weyl *w, *wm;
  spinor *s, *sn;
  block_t *b;
  bndry_t *bb;
  ani_params_t ani;
  float ut_tilde;

  ani = ani_parms();
  one_over_gammaf = (float)(ani.nu / ani.xi);
  ut_tilde = (float)(1.0 / ani.ut_tilde);

  b = blk_list(grid, &nb, &isw);

  if ((n < 0) || (n >= nb)) {
    error_loc(1, 1, "Dw_bnd [Dw_bnd.c]",
              "Block grid is not allocated or block number out of range");
    return;
  }

  b += n;
  bb = (*b).bb;

  if ((k < 0) || (k >= (*b).ns) || ((*b).u == NULL) || (bb == NULL) ||
      (l >= (*bb).nw)) {
    error_loc(1, 1, "Dw_bnd [Dw_bnd.c]",
              "Attempt to access unallocated memory space");
    return;
  }

  bc = bc_type();
  moh = -0.5f;
  moh *= ut_tilde;

  s = (*b).s[k];

  /********************************** face -0
   * ***********************************/

  ipp = (*bb).ipp;
  w = (*bb).w[l];
  wm = w + (*bb).vol;

  if ((cpr[0] == 0) && ((*b).bo[0] == 0) && (bc != 3)) {
    for (; w < wm; w++) {
      sn = s + (*ipp);
      ipp += 1;
      (*sn) = s0;
      (*w) = w0;
    }
  } else {
    u = (*bb).u;

    for (; w < wm; w++) {
      sn = s + (*ipp);
      ipp += 1;
      _vector_add(chi.c1, (*sn).c1, (*sn).c3);
      _vector_add(chi.c2, (*sn).c2, (*sn).c4);
      _vector_mul(chi.c1, moh, chi.c1);
      _vector_mul(chi.c2, moh, chi.c2);
      mul_umat(&chi, u, w);
      u += 1;
    }
  }

  /********************************** face +0
   * ***********************************/

  bb += 1;
  ipp = (*bb).ipp;
  w = (*bb).w[l];
  wm = w + (*bb).vol;

  if ((cpr[0] == (NPROC0 - 1)) && (((*b).bo[0] + (*b).bs[0]) == L0) &&
      (bc != 3)) {
    if (bc == 0) {
      for (; w < wm; w++) {
        sn = s + (*ipp);
        ipp += 1;
        (*sn) = s0;
        (*w) = w0;
      }
    } else {
      for (; w < wm; w++)
        (*w) = w0;
    }
  } else {
    u = (*bb).u;

    for (; w < wm; w++) {
      sn = s + (*ipp);
      ipp += 1;
      _vector_sub(chi.c1, (*sn).c1, (*sn).c3);
      _vector_sub(chi.c2, (*sn).c2, (*sn).c4);
      _vector_mul(chi.c1, moh, chi.c1);
      _vector_mul(chi.c2, moh, chi.c2);
      mul_uinv(&chi, u, w);
      u += 1;
    }
  }

  /********************************** face -1
   * ***********************************/

  bb += 1;
  ipp = (*bb).ipp;
  w = (*bb).w[l];
  wm = w + (*bb).vol;
  u = (*bb).u;

  for (; w < wm; w++) {

    sn = s + (*ipp);
    ipp += 1;
    _vector_i_add(chi.c1, (*sn).c1, (*sn).c4);
    _vector_i_add(chi.c2, (*sn).c2, (*sn).c3);
    _vector_mul(chi.c1, moh, chi.c1);
    _vector_mul(chi.c2, moh, chi.c2);
    _vector_mul(chi.c1, one_over_gammaf, chi.c1);
    _vector_mul(chi.c2, one_over_gammaf, chi.c2);
    mul_umat(&chi, u, w);

    u += 1;
  }

  /********************************** face +1
   * ***********************************/

  bb += 1;
  ipp = (*bb).ipp;
  w = (*bb).w[l];
  wm = w + (*bb).vol;
  u = (*bb).u;

  for (; w < wm; w++) {

    sn = s + (*ipp);
    ipp += 1;
    _vector_i_sub(chi.c1, (*sn).c1, (*sn).c4);
    _vector_i_sub(chi.c2, (*sn).c2, (*sn).c3);
    _vector_mul(chi.c1, moh, chi.c1);
    _vector_mul(chi.c2, moh, chi.c2);
    _vector_mul(chi.c1, one_over_gammaf, chi.c1);
    _vector_mul(chi.c2, one_over_gammaf, chi.c2);
    mul_uinv(&chi, u, w);

    u += 1;
  }

  /********************************** face -2
   * ***********************************/

  bb += 1;
  ipp = (*bb).ipp;
  w = (*bb).w[l];
  wm = w + (*bb).vol;
  u = (*bb).u;

  for (; w < wm; w++) {

    sn = s + (*ipp);
    ipp += 1;
    _vector_add(chi.c1, (*sn).c1, (*sn).c4);
    _vector_sub(chi.c2, (*sn).c2, (*sn).c3);
    _vector_mul(chi.c1, moh, chi.c1);
    _vector_mul(chi.c2, moh, chi.c2);
    _vector_mul(chi.c1, one_over_gammaf, chi.c1);
    _vector_mul(chi.c2, one_over_gammaf, chi.c2);
    mul_umat(&chi, u, w);

    u += 1;
  }

  /********************************** face +2
   * ***********************************/

  bb += 1;
  ipp = (*bb).ipp;
  w = (*bb).w[l];
  wm = w + (*bb).vol;
  u = (*bb).u;

  for (; w < wm; w++) {

    sn = s + (*ipp);
    ipp += 1;
    _vector_sub(chi.c1, (*sn).c1, (*sn).c4);
    _vector_add(chi.c2, (*sn).c2, (*sn).c3);
    _vector_mul(chi.c1, moh, chi.c1);
    _vector_mul(chi.c2, moh, chi.c2);
    _vector_mul(chi.c1, one_over_gammaf, chi.c1);
    _vector_mul(chi.c2, one_over_gammaf, chi.c2);
    mul_uinv(&chi, u, w);

    u += 1;
  }

  /********************************** face -3
   * ***********************************/

  bb += 1;
  ipp = (*bb).ipp;
  w = (*bb).w[l];
  wm = w + (*bb).vol;
  u = (*bb).u;

  for (; w < wm; w++) {

    sn = s + (*ipp);
    ipp += 1;
    _vector_i_add(chi.c1, (*sn).c1, (*sn).c3);
    _vector_i_sub(chi.c2, (*sn).c2, (*sn).c4);
    _vector_mul(chi.c1, moh, chi.c1);
    _vector_mul(chi.c2, moh, chi.c2);
    _vector_mul(chi.c1, one_over_gammaf, chi.c1);
    _vector_mul(chi.c2, one_over_gammaf, chi.c2);
    mul_umat(&chi, u, w);

    u += 1;
  }

  /********************************** face +3
   * ***********************************/

  bb += 1;
  ipp = (*bb).ipp;
  w = (*bb).w[l];
  wm = w + (*bb).vol;
  u = (*bb).u;

  for (; w < wm; w++) {

    sn = s + (*ipp);
    ipp += 1;
    _vector_i_sub(chi.c1, (*sn).c1, (*sn).c3);
    _vector_i_add(chi.c2, (*sn).c2, (*sn).c4);
    _vector_mul(chi.c1, moh, chi.c1);
    _vector_mul(chi.c2, moh, chi.c2);
    _vector_mul(chi.c1, one_over_gammaf, chi.c1);
    _vector_mul(chi.c2, one_over_gammaf, chi.c2);
    mul_uinv(&chi, u, w);

    u += 1;
  }
}
