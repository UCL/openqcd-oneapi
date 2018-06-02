#ifndef DATA_TYPE_DIFFS_H
#define DATA_TYPE_DIFFS_H

#ifndef OPENQCD_INTERNAL
#define OPENQCD_INTERNAL
#endif

#include "su3.h"
#include <stdlib.h>

extern double abs_diff_double(double l, double r);
extern double abs_diff_complex(complex_dble l, complex_dble r);
extern double norm_diff_complex(complex_dble l, complex_dble r);
extern double abs_diff_array_double(double l[], double r[], size_t N);
extern double abs_diff_array_complex(complex_dble l[], complex_dble r[],
                                     size_t N);
extern double norm_diff_array_complex(complex_dble l[], complex_dble r[],
                                      size_t N);
extern double abs_diff_su3(su3_dble const *l, su3_dble const *r);
extern double norm_diff_su3(su3_dble const *l, su3_dble const *r);
extern double abs_diff_su3_alg(su3_alg_dble const *l, su3_alg_dble const *r);

extern size_t index_diff_array_int(int l[], int r[], size_t N);
extern size_t index_diff_array_double(double l[], double r[], size_t N,
                                      double eps);

#endif /* DATA_TYPE_DIFFS_H */
