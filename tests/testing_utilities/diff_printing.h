#ifndef DIFF_PRINTING_H
#define DIFF_PRINTING_H

#ifndef OPENQCD_INTERNAL
#define OPENQCD_INTERNAL
#endif

#include "data_type_diffs.h"

extern void print_complex_diff(complex_dble l, complex_dble r);
extern void report_complex_diff(complex_dble l, complex_dble r, char const *n);
extern void report_complex_diff_indent(complex_dble l, complex_dble r,
                                       char const *n, int lvl);
extern void report_complex_array_diff(complex_dble l[], complex_dble r[],
                                      size_t N, char const *n);

extern void print_int_array_comparison_tail(int l[], int r[], size_t Nl,
                                            size_t Nr, size_t num);
extern void print_int_array_comparison_mid(int l[], int r[], size_t N,
                                           size_t pos, size_t num);

extern void print_double_array_comparison_tail(double l[], double r[],
                                               size_t Nl, size_t Nr,
                                               size_t num);
extern void print_double_array_comparison_mid(double l[], double r[], size_t N,
                                              size_t pos, size_t num);

#endif /* DIFF_PRINTING_H */
