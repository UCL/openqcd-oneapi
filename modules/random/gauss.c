
/*******************************************************************************
 *
 * File gauss.c
 *
 * Copyright (C) 2005 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Generation of Gaussian random numbers
 *
 * The externally accessible functions are
 *
 *   void gauss(float r[], int n)
 *     Generates n single-precision Gaussian random numbers x with distribution
 *     proportional to exp(-x^2) and assigns them to r[0],..,r[n-1]
 *
 *   void gauss_dble(double rd[], int n)
 *     Generates n double-precision Gaussian random numbers x with distribution
 *     proportional to exp(-x^2) and assigns them to rd[0],..,rd[n-1]
 *
 * Notes:
 *
 * If the SITERANDOM flag has been set both functions take one additional
 * argument, the site for which to generate the number.
 *
 *******************************************************************************/

#define GAUSS_C
#define OPENQCD_INTERNAL

#include "random.h"

static int init = 0;
static double twopi;

#ifdef SITERANDOM
void gauss(float r[], int n, int ix)
#else
void gauss(float r[], int n)
#endif
{
  int k;
  float u[2];
  double x1, x2, rho, y1, y2;

  if (init == 0) {
    twopi = 8.0 * atan(1.0);
    init = 1;
  }

  for (k = 0; k < n;) {
#ifdef SITERANDOM
    if (ix == -1) {
      ranlxs(u, 2);
    }
    ranlxs_site(u, 2, ix);
#else
    ranlxs(u, 2);
#endif
    x1 = (double)(u[0]);
    x2 = (double)(u[1]);

    rho = -log(1.0 - x1);
    rho = sqrt(rho);
    x2 *= twopi;
    y1 = rho * sin(x2);
    y2 = rho * cos(x2);

    r[k++] = (float)(y1);
    if (k < n) {
      r[k++] = (float)(y2);
    }
  }
}

#ifdef SITERANDOM
void gauss_dble(double rd[], int n, int ix)
#else
void gauss_dble(double rd[], int n)
#endif
{
  int k;
  double ud[2];
  double x1, x2, rho, y1, y2;

  if (init == 0) {
    twopi = 8.0 * atan(1.0);
    init = 1;
  }

  for (k = 0; k < n;) {
#ifdef SITERANDOM
    if (ix == -1) {
      ranlxd(ud, 2);
    }
    ranlxd_site(ud, 2, ix);
#else
    ranlxd(ud, 2);
#endif
    x1 = ud[0];
    x2 = ud[1];

    rho = -log(1.0 - x1);
    rho = sqrt(rho);
    x2 *= twopi;
    y1 = rho * sin(x2);
    y2 = rho * cos(x2);

    rd[k++] = y1;
    if (k < n) {
      rd[k++] = y2;
    }
  }
}
