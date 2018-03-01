
/*******************************************************************************
*
* File liealg.c
*
* Copyright (C) 2005, 2009-2011, 2016 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Basic functions for fields with values in the Lie algebra of SU(3)
*
* The externally accessible functions are
*
*   void random_alg(int vol,su3_alg_dble *X)
*     Initializes the Lie algebra elements X to random values
*     with distribution proportional to exp{tr[X^2]}.
*
*   double norm_square_alg(int vol,int icom,su3_alg_dble *X)
*     Computes the square of the norm of the norm squared of the field X.
*
*   double scalar_prod_alg(int vol,int icom,su3_alg_dble *X,su3_alg_dble *Y)
*     Computes the scalar product of the fields X and Y.
*
*   void set_alg2zero(int vol,su3_alg_dble *X)
*     Sets the array elements X to zero.
*
*   void set_ualg2zero(int vol,u3_alg_dble *X)
*     Sets the array elements X to zero.
*
*   void assign_alg2alg(int vol,su3_alg_dble *X,su3_alg_dble *Y)
*     Assigns the field X to the field Y.
*
*   void swap_alg(int vol,su3_alg_dble *X,su3_alg_dble *Y)
*     Swaps the fields X and Y.
*
*   void muladd_assign_alg(int vol,double r,su3_alg_dble *X,su3_alg_dble *Y)
*     Adds r*X to Y.
*
*   void project_to_su3alg(const su3_dble *u, su3_alg_dble *X)
*     Projects an arbitrary 3x3 complex matrix in u to the su3 algebra and
*     stores the result in X. The projection formula is
*
*     X = P{U} = 1/2 (U - U^dag) - 1/6 tr (U - U^dag)
*
*   void su3alg_to_cm3x3(su3_alg_dble const *X, su3_dble *u)
*     Computes the corresponding complex 3x3 matrix from the generator
*     representation of an su3 algebra object.
*
* Notes:
*
* Lie algebra elements X are traceless antihermitian 3x3 matrices that
* are represented by structures with real elements x1,...,x8 through
*
*  X_11=i*(x1+x2), X_22=i*(x2-2*x1), X_33=i*(x1-2*x2),
*
*  X_12=x3+i*x4, X_13=x5+i*x6, X_23=x7+i*x8
*
* The scalar product (X,Y) of any two elements of the Lie algebra is
*
*  (X,Y)=-2*tr{XY}
*
* and the norm of X is (X,X)^(1/2).
*
* All programs in this module operate on arrays of Lie algebra elements whose
* base address is passed through the arguments. The length of the array is
* specified by the parameter vol. Scalar products etc. are globally summed if
* the parameter icom is equal to 1. In this case the calculated values are
* guaranteed to be exactly the same on all processes.
*
*******************************************************************************/

#define LIEALG_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "utils.h"
#include "random.h"
#include "linalg.h"
#include "global.h"

static int ism, init = 0;
static double c1 = 0.0, c2, c3, rb[8];

void random_alg(int vol, su3_alg_dble *X)
{
  su3_alg_dble *Xm;
#ifdef SITERANDOM
  int ix = VOLUME / 2;
  int divide_by = vol / (VOLUME / 2);
#endif

  if (c1 == 0.0) {
    c1 = (sqrt(3.0) + 1.0) / 6.0;
    c2 = (sqrt(3.0) - 1.0) / 6.0;
    c3 = 1.0 / sqrt(2.0);
  }

  Xm = X + vol;

  for (; X < Xm; X++) {
#ifdef SITERANDOM
    gauss_dble(rb, 8, ix++ / divide_by);
#else
    gauss_dble(rb, 8);
#endif

    (*X).c1 = c1 * rb[0] + c2 * rb[1];
    (*X).c2 = c1 * rb[1] + c2 * rb[0];
    (*X).c3 = c3 * rb[2];
    (*X).c4 = c3 * rb[3];
    (*X).c5 = c3 * rb[4];
    (*X).c6 = c3 * rb[5];
    (*X).c7 = c3 * rb[6];
    (*X).c8 = c3 * rb[7];
  }
}

double norm_square_alg(int vol, int icom, su3_alg_dble *X)
{
  double sm;
  su3_alg_dble *Xm;

  if (init == 0) {
    ism = init_hsum(1);
    init = 1;
  }

  reset_hsum(ism);
  Xm = X + vol;

  for (; X < Xm; X++) {
    sm = 3.0 * ((*X).c1 * (*X).c1 + (*X).c2 * (*X).c2 - (*X).c1 * (*X).c2) +
         (*X).c3 * (*X).c3 + (*X).c4 * (*X).c4 + (*X).c5 * (*X).c5 +
         (*X).c6 * (*X).c6 + (*X).c7 * (*X).c7 + (*X).c8 * (*X).c8;

    add_to_hsum(ism, &sm);
  }

  if ((icom == 1) && (NPROC > 1))
    global_hsum(ism, &sm);
  else
    local_hsum(ism, &sm);

  return 4.0 * sm;
}

double scalar_prod_alg(int vol, int icom, su3_alg_dble *X, su3_alg_dble *Y)
{
  double sm;
  su3_alg_dble *Xm;

  if (init == 0) {
    ism = init_hsum(1);
    init = 1;
  }

  reset_hsum(ism);
  Xm = X + vol;

  for (; X < Xm; X++) {
    sm = 12.0 * ((*X).c1 * (*Y).c1 + (*X).c2 * (*Y).c2) -
         6.0 * ((*X).c1 * (*Y).c2 + (*X).c2 * (*Y).c1) +
         4.0 * ((*X).c3 * (*Y).c3 + (*X).c4 * (*Y).c4 + (*X).c5 * (*Y).c5 +
                (*X).c6 * (*Y).c6 + (*X).c7 * (*Y).c7 + (*X).c8 * (*Y).c8);

    Y += 1;
    add_to_hsum(ism, &sm);
  }

  if ((icom == 1) && (NPROC > 1))
    global_hsum(ism, &sm);
  else
    local_hsum(ism, &sm);

  return sm;
}

void set_alg2zero(int vol, su3_alg_dble *X)
{
  su3_alg_dble *Xm;

  Xm = X + vol;

  for (; X < Xm; X++) {
    (*X).c1 = 0.0;
    (*X).c2 = 0.0;
    (*X).c3 = 0.0;
    (*X).c4 = 0.0;
    (*X).c5 = 0.0;
    (*X).c6 = 0.0;
    (*X).c7 = 0.0;
    (*X).c8 = 0.0;
  }
}

void set_ualg2zero(int vol, u3_alg_dble *X)
{
  u3_alg_dble *Xm;

  Xm = X + vol;

  for (; X < Xm; X++) {
    (*X).c1 = 0.0;
    (*X).c2 = 0.0;
    (*X).c3 = 0.0;
    (*X).c4 = 0.0;
    (*X).c5 = 0.0;
    (*X).c6 = 0.0;
    (*X).c7 = 0.0;
    (*X).c8 = 0.0;
    (*X).c9 = 0.0;
  }
}

void assign_alg2alg(int vol, su3_alg_dble const *X, su3_alg_dble *Y)
{
  su3_alg_dble const *Xm;

  Xm = X + vol;

  for (; X < Xm; X++) {
    (*Y).c1 = (*X).c1;
    (*Y).c2 = (*X).c2;
    (*Y).c3 = (*X).c3;
    (*Y).c4 = (*X).c4;
    (*Y).c5 = (*X).c5;
    (*Y).c6 = (*X).c6;
    (*Y).c7 = (*X).c7;
    (*Y).c8 = (*X).c8;

    Y += 1;
  }
}

void swap_alg(int vol, su3_alg_dble *X, su3_alg_dble *Y)
{
  double r;
  su3_alg_dble *Xm;

  Xm = X + vol;

  for (; X < Xm; X++) {
    r = (*Y).c1;
    (*Y).c1 = (*X).c1;
    (*X).c1 = r;

    r = (*Y).c2;
    (*Y).c2 = (*X).c2;
    (*X).c2 = r;

    r = (*Y).c3;
    (*Y).c3 = (*X).c3;
    (*X).c3 = r;

    r = (*Y).c4;
    (*Y).c4 = (*X).c4;
    (*X).c4 = r;

    r = (*Y).c5;
    (*Y).c5 = (*X).c5;
    (*X).c5 = r;

    r = (*Y).c6;
    (*Y).c6 = (*X).c6;
    (*X).c6 = r;

    r = (*Y).c7;
    (*Y).c7 = (*X).c7;
    (*X).c7 = r;

    r = (*Y).c8;
    (*Y).c8 = (*X).c8;
    (*X).c8 = r;

    Y += 1;
  }
}

void add_alg(int vol, su3_alg_dble const *X, su3_alg_dble *Y)
{
  su3_alg_dble const *Xm;

  Xm = X + vol;

  for (; X < Xm; X++) {
    (*Y).c1 += (*X).c1;
    (*Y).c2 += (*X).c2;
    (*Y).c3 += (*X).c3;
    (*Y).c4 += (*X).c4;
    (*Y).c5 += (*X).c5;
    (*Y).c6 += (*X).c6;
    (*Y).c7 += (*X).c7;
    (*Y).c8 += (*X).c8;

    Y += 1;
  }
}

void muladd_assign_alg(int vol, double r, su3_alg_dble const *X,
                       su3_alg_dble *Y)
{
  su3_alg_dble const *Xm;

  Xm = X + vol;

  for (; X < Xm; X++) {
    (*Y).c1 += r * (*X).c1;
    (*Y).c2 += r * (*X).c2;
    (*Y).c3 += r * (*X).c3;
    (*Y).c4 += r * (*X).c4;
    (*Y).c5 += r * (*X).c5;
    (*Y).c6 += r * (*X).c6;
    (*Y).c7 += r * (*X).c7;
    (*Y).c8 += r * (*X).c8;

    Y += 1;
  }
}

void project_to_su3alg(su3_dble const *u, su3_alg_dble *X)
{
  (*X).c1 = ((*u).c11.im - (*u).c22.im) / 3.;
  (*X).c2 = ((*u).c11.im - (*u).c33.im) / 3.;
  (*X).c3 = ((*u).c12.re - (*u).c21.re) / 2.;
  (*X).c4 = ((*u).c21.im + (*u).c12.im) / 2.;
  (*X).c5 = ((*u).c13.re - (*u).c31.re) / 2.;
  (*X).c6 = ((*u).c31.im + (*u).c13.im) / 2.;
  (*X).c7 = ((*u).c23.re - (*u).c32.re) / 2.;
  (*X).c8 = ((*u).c32.im + (*u).c23.im) / 2.;
}

void su3alg_to_cm3x3(su3_alg_dble const *X, su3_dble *u)
{
  (*u).c11.re = 0;
  (*u).c11.im = (*X).c1 + (*X).c2;
  (*u).c12.re = (*X).c3;
  (*u).c12.im = (*X).c4;
  (*u).c13.re = (*X).c5;
  (*u).c13.im = (*X).c6;
  (*u).c21.re = -(*X).c3;
  (*u).c21.im = (*X).c4;
  (*u).c22.re = 0;
  (*u).c22.im = -2 * (*X).c1 + (*X).c2;
  (*u).c23.re = (*X).c7;
  (*u).c23.im = (*X).c8;
  (*u).c31.re = -(*X).c5;
  (*u).c31.im = (*X).c6;
  (*u).c32.re = -(*X).c7;
  (*u).c32.im = (*X).c8;
  (*u).c33.re = 0;
  (*u).c33.im = (*X).c1 - 2 * (*X).c2;
}
