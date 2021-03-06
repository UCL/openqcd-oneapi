
/*******************************************************************************
 *
 * File check1.c
 *
 * Copyright (C) 2010 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Computation of the modified Bessel function I0(x) [program i0m()]
 *
 *******************************************************************************/

#define OPENQCD_INTERNAL

#if !defined (STATIC_SIZES)
#error : This test cannot be compiled with dynamic lattice sizes
#endif

#include "extras.h"
#include "utils.h"

int main(void)
{
  double x, y;

  printf("\n");
  printf("Modified Bessel function I0(x) [program i0m()]\n");
  printf("----------------------------------------------\n\n");

  printf("Print selected values:\n\n");

  for (;;) {
    printf("Specify x: ");

    if (scanf("%lf", &x) == 1) {
      y = i0m(x);
      printf("x = %.4e, exp(-x)*I0(x) = %.15e\n\n", x, y);
    } else {
      printf("No value specified, program stopped\n\n");
      break;
    }
  }

  exit(0);
}
