#define DIFF_PRINTING_C

#ifndef OPENQCD_INTERNAL
#define OPENQCD_INTERNAL
#endif

#include "diff_printing.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static const char padder[] = "......................";

void print_complex_diff(complex_dble l, complex_dble r)
{
  printf("%.1e ({%.1e, %.1e} - {%.1e, %.1e})", norm_diff_complex(l, r), l.re,
         l.im, r.re, r.im);
}

void report_complex_diff(complex_dble l, complex_dble r, char const *name)
{
  printf("diff %s: ", name);
  print_complex_diff(l, r);
  printf("\n");
}

void report_complex_diff_indent(complex_dble l, complex_dble r,
                                char const *name, int indent_lvl)
{
  int i = 0;
  for (; i < indent_lvl; ++i)
    printf("  ");

  report_complex_diff(l, r, name);
}

void report_complex_array_diff(complex_dble l[], complex_dble r[], size_t N,
                               char const *name)
{
  size_t n = 0;
  for (; n < N; ++n) {
    printf("  diff %s[%lu]: ", name, n);
    print_complex_diff(l[n], r[n]);
    printf("\n");
  }
}

void report_su3_diff(su3_dble const *l, su3_dble const *r, char const *name)
{
  printf("   diff %s:\n", name);
  printf("    |({%+.2e, %+.2e} {%+.2e, %+.2e} {%+.2e, %+.2e})\n", (*l).c11.re,
         (*l).c11.im, (*l).c12.re, (*l).c12.im, (*l).c13.re, (*l).c13.im);
  printf("    |({%+.2e, %+.2e} {%+.2e, %+.2e} {%+.2e, %+.2e})\n", (*l).c21.re,
         (*l).c21.im, (*l).c22.re, (*l).c22.im, (*l).c23.re, (*l).c23.im);
  printf("    |({%+.2e, %+.2e} {%+.2e, %+.2e} {%+.2e, %+.2e})\n", (*l).c31.re,
         (*l).c31.im, (*l).c32.re, (*l).c32.im, (*l).c33.re, (*l).c33.im);
  printf("   -\n");
  printf("     ({%+.2e, %+.2e} {%+.2e, %+.2e} {%+.2e, %+.2e})|\n", (*r).c11.re,
         (*r).c11.im, (*r).c12.re, (*r).c12.im, (*r).c13.re, (*r).c13.im);
  printf("     ({%+.2e, %+.2e} {%+.2e, %+.2e} {%+.2e, %+.2e})|\n", (*r).c21.re,
         (*r).c21.im, (*r).c22.re, (*r).c22.im, (*r).c23.re, (*r).c23.im);
  printf("     ({%+.2e, %+.2e} {%+.2e, %+.2e} {%+.2e, %+.2e})|\n", (*r).c31.re,
         (*r).c31.im, (*r).c32.re, (*r).c32.im, (*r).c33.re, (*r).c33.im);
  printf("   =\n");
  printf("    |(%+.2e %+.2e %+.2e)|\n", norm_diff_complex((*l).c11, (*r).c11),
         norm_diff_complex((*l).c12, (*r).c12),
         norm_diff_complex((*l).c13, (*r).c13));
  printf("    |(%+.2e %+.2e %+.2e)|\n", norm_diff_complex((*l).c21, (*r).c21),
         norm_diff_complex((*l).c22, (*r).c22),
         norm_diff_complex((*l).c23, (*r).c23));
  printf("    |(%+.2e %+.2e %+.2e)|\n", norm_diff_complex((*l).c31, (*r).c31),
         norm_diff_complex((*l).c32, (*r).c32),
         norm_diff_complex((*l).c33, (*r).c33));
}

void print_int_array_comparison_tail(int l[], int r[], size_t Nl, size_t Nr,
                                     size_t num)
{
  size_t begin, end, N, pos;
  int num_digits;

  if (Nl > Nr) {
    N = Nl;
    pos = Nr - 1;
  } else {
    N = Nr;
    pos = Nl - 1;
  }

  /* end index of printing */
  if (pos + num >= N)
    end = N;
  else
    end = pos + num + 1;

  num_digits = (int)log10((double)(end - 1)) + 1;
  printf("%*c%s%*c%s\n", num_digits + 7, ' ', "test", 1, ' ', "expec");

  /* pre block */
  if (pos < num)
    begin = 0;
  else
    begin = pos - num;

  for (; begin < pos; ++begin)
    printf("     [%lu]: %3d  %3d\n", begin, l[begin], r[begin]);

  /* Main diff */
  printf(" end [%lu]: %3d  %3d\n", pos, l[pos], r[pos]);

  /* end_block */
  for (begin = pos + 1; begin < end; ++begin) {
    if (begin >= Nl) {
      printf("     [%lu]:      %3d\n", begin, r[begin]);
    } else if (begin >= Nr) {
      printf("     [%lu]: %3d\n", begin, l[begin]);
    }
  }

  if (end != N) {
    printf("     [%s]\n", padder + strlen(padder) - num_digits);

    if (N == Nl) {
      printf("     [%lu]: %3d\n", N - 1, l[N - 1]);
    } else {
      printf("     [%lu]:      %3d\n", N - 1, r[N - 1]);
    }
  }
}

void print_int_array_comparison_mid(int l[], int r[], size_t N, size_t pos,
                                    size_t num)
{
  size_t begin, end;
  int num_digits;

  /* end index of printing */
  if (pos + num >= N)
    end = N;
  else
    end = pos + num + 1;

  num_digits = (int)log10((double)(end - 1)) + 1;
  printf("%*c%s%*c%s\n", num_digits + 7, ' ', "test", 1, ' ', "expec");

  /* pre block */
  if (pos < num)
    begin = 0;
  else
    begin = pos - num;

  for (; begin < pos; ++begin)
    printf("    [%lu]: %3d  %3d\n", begin, l[begin], r[begin]);

  /* Main diff */
  printf(" -> [%lu]: %3d  %3d\n", pos, l[pos], r[pos]);

  /* end_block */
  for (begin = pos + 1; begin < end; ++begin)
    printf("    [%lu]: %3d  %3d\n", begin, l[begin], r[begin]);
}

void print_double_array_comparison_tail(double l[], double r[], size_t Nl,
                                        size_t Nr, size_t num)
{
  size_t begin, end, N, pos;

  if (Nl > Nr) {
    N = Nl;
    pos = Nr - 1;
  } else {
    N = Nr;
    pos = Nl - 1;
  }

  printf("%*c%s%*c%s\n", 16, ' ', "test", 5, ' ', "expec");

  /* pre block */
  if (pos < num)
    begin = 0;
  else
    begin = pos - num;

  for (; begin < pos; ++begin)
    printf("     [%3lu]: %.2e  %.2e\n", begin, l[begin], r[begin]);

  /* Main diff */
  printf(" end [%3lu]: %.2e  %.2e\n", pos, l[pos], r[pos]);

  /* end index of printing */
  if (pos + num >= N)
    end = N;
  else
    end = pos + num + 1;

  for (begin = pos + 1; begin < end; ++begin) {
    if (begin >= Nl) {
      printf("     [%3lu]:           %.2e\n", begin, r[begin]);
    } else if (begin >= Nr) {
      printf("     [%3lu]: %.2e\n", begin, l[begin]);
    }
  }

  if (end != N) {
    printf("     [...]\n");

    if (N == Nl) {
      printf("     [%3lu]: %.2e\n", N - 1, l[N - 1]);
    } else {
      printf("     [%3lu]:           %.2e\n", N - 1, r[N - 1]);
    }
  }
}

void print_double_array_comparison_mid(double l[], double r[], size_t N,
                                       size_t pos, size_t num)
{
  size_t begin, end;

  printf("%*c%s%*c%s\n", 15, ' ', "test", 5, ' ', "expec");

  /* pre block */
  if (pos < num)
    begin = 0;
  else
    begin = pos - num;

  for (; begin < pos; ++begin)
    printf("    [%3lu]: %.2e  %.2e\n", begin, l[begin], r[begin]);

  /* Main diff */
  printf(" -> [%3lu]: %.2e  %.2e\n", pos, l[pos], r[pos]);

  /* end index of printing */
  if (pos + num >= N)
    end = N;
  else
    end = pos + num + 1;

  for (begin = pos + 1; begin < end; ++begin)
    printf("    [%3lu]: %.2e  %.2e\n", begin, l[begin], r[begin]);
}
