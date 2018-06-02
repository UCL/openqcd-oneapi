
/*******************************************************************************
 *
 * File check2.c
 *
 * Copyright (C) 2013 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Test of the program fdigits()
 *
 *******************************************************************************/

#define OPENQCD_INTERNAL

#if !defined (STATIC_SIZES)
#error : This test cannot be compiled with dynamic lattice sizes
#endif

#include "utils.h"

int main(void)
{
  int n, ret;
  double x;

  printf("\n");
  printf("Test of the program fdigits()\n");
  printf("-----------------------------\n\n");

  while (1) {
    printf("x = ");
    ret = scanf("%lf", &x);

    if (ret != 1) {
      printf("Scanf failed\n");
      exit(1);
    }

    n = fdigits(x);
    printf("    %.*f\n\n", n, x);
  }

  exit(0);
}
