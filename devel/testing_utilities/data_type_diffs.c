#define DATA_TYPE_DIFFS_C

#include "data_type_diffs.h"
#include <math.h>
#include <utils.h>

double abs_diff_double(double l, double r) { return fabs(l - r); }

double abs_diff_complex(complex_dble l, complex_dble r)
{
  return abs_diff_double(l.re, r.re) + abs_diff_double(l.im, r.im);
}

double norm_complex(complex_dble l)
{
  return sqrt(l.re * l.re + l.im * l.im);
}

double norm_diff_complex(complex_dble l, complex_dble r)
{
  return sqrt(square_dble(l.re - r.re) + square_dble(l.im - r.im));
}

double abs_diff_array_double(double l[], double r[], size_t N)
{
  size_t n;
  double result = 0.;

  for (n = 0; n < N; ++n)
    result += abs_diff_double(l[n], r[n]);

  return result;
}

double abs_diff_array_complex(complex_dble l[], complex_dble r[], size_t N)
{
  size_t n;
  double result = 0.;

  for (n = 0; n < N; ++n)
    result += abs_diff_complex(l[n], r[n]);

  return result;
}

double norm_diff_array_complex(complex_dble l[], complex_dble r[], size_t N)
{
  size_t n;
  double result = 0.;

  for (n = 0; n < N; ++n)
    result += norm_diff_complex(l[n], r[n]);

  return result;
}

double abs_diff_su3(su3_dble const *l, su3_dble const *r)
{
  double result = 0.;

  result += abs_diff_complex((*l).c11, (*r).c11);
  result += abs_diff_complex((*l).c12, (*r).c12);
  result += abs_diff_complex((*l).c13, (*r).c13);

  result += abs_diff_complex((*l).c21, (*r).c21);
  result += abs_diff_complex((*l).c22, (*r).c22);
  result += abs_diff_complex((*l).c23, (*r).c23);

  result += abs_diff_complex((*l).c31, (*r).c31);
  result += abs_diff_complex((*l).c32, (*r).c32);
  result += abs_diff_complex((*l).c33, (*r).c33);

  return result;
}

double norm_su3(su3_dble const* l)
{
  double result = 0.;

  result += norm_complex((*l).c11);
  result += norm_complex((*l).c12);
  result += norm_complex((*l).c13);

  result += norm_complex((*l).c21);
  result += norm_complex((*l).c22);
  result += norm_complex((*l).c23);

  result += norm_complex((*l).c31);
  result += norm_complex((*l).c32);
  result += norm_complex((*l).c33);

  return result;

}

double norm_diff_su3(su3_dble const *l, su3_dble const *r)
{
  double result = 0.;

  result += norm_diff_complex((*l).c11, (*r).c11);
  result += norm_diff_complex((*l).c12, (*r).c12);
  result += norm_diff_complex((*l).c13, (*r).c13);

  result += norm_diff_complex((*l).c21, (*r).c21);
  result += norm_diff_complex((*l).c22, (*r).c22);
  result += norm_diff_complex((*l).c23, (*r).c23);

  result += norm_diff_complex((*l).c31, (*r).c31);
  result += norm_diff_complex((*l).c32, (*r).c32);
  result += norm_diff_complex((*l).c33, (*r).c33);

  return result;
}

double abs_diff_su3_alg(su3_alg_dble const *l, su3_alg_dble const *r)
{
  double result = 0.;

  result += abs_diff_double((*l).c1, (*r).c1);
  result += abs_diff_double((*l).c2, (*r).c2);
  result += abs_diff_double((*l).c3, (*r).c3);
  result += abs_diff_double((*l).c4, (*r).c4);
  result += abs_diff_double((*l).c5, (*r).c5);
  result += abs_diff_double((*l).c6, (*r).c6);
  result += abs_diff_double((*l).c7, (*r).c7);
  result += abs_diff_double((*l).c8, (*r).c8);

  return result;
}

size_t index_diff_array_int(int l[], int r[], size_t N)
{
  size_t n;

  for (n = 0; n < N; ++n)
    if (l[n] != r[n])
      return n;

  return N;
}

size_t index_diff_array_double(double l[], double r[], size_t N, double eps)
{
  size_t n;

  for (n = 0; n < N; ++n)
    if (abs_diff_double(l[n], r[n]) > eps)
      return n;

  return N;
}
