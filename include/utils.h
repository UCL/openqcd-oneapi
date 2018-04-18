
/*******************************************************************************
 *
 * File utils.h
 *
 * Copyright (C) 2011, 2016 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef UTILS_H
#define UTILS_H

#include "su3.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>

#if ((DBL_MANT_DIG != 53) || (DBL_MIN_EXP != -1021) || (DBL_MAX_EXP != 1024))
#error : Machine is not compliant with the IEEE-754 standard
#endif

#if (SHRT_MAX == 0x7fffffff)
typedef short int stdint_t;
typedef unsigned short int stduint_t;
#elif (INT_MAX == 0x7fffffff)
typedef int stdint_t;
typedef unsigned int stduint_t;
#elif (LONG_MAX == 0x7fffffff)
typedef long int stdint_t;
typedef unsigned long int stduint_t;
#else
#error : There is no four-byte integer type on this machine
#endif

#undef UNKNOWN_ENDIAN
#undef LITTLE_ENDIAN
#undef BIG_ENDIAN

#define UNKNOWN_ENDIAN 0
#define LITTLE_ENDIAN 1
#define BIG_ENDIAN 2

#undef IMAX
#define IMAX(n, m) ((n) + ((m) - (n)) * ((m) > (n)))

typedef enum
{
  ALL_PTS,
  EVEN_PTS,
  ODD_PTS,
  NO_PTS,
  PT_SETS
} ptset_t;

/* ENDIAN_C */
extern int endianness(void);
extern void bswap_int(int n, void *a);
extern void bswap_double(int n, void *a);

/* ERROR_C */
extern void set_error_file(char const *path, int loc_flag);
extern void error(int test, int no, char const *name, char const *format, ...);
extern void error_root(int test, int no, char const *name, char const *format,
                       ...);
extern void error_loc(int test, int no, char const *name, char const *format,
                      ...);

/* HSUM_C */
extern int init_hsum(int n);
extern void reset_hsum(int id);
extern void add_to_hsum(int id, double const *x);
extern void local_hsum(int id, double *sx);
extern void global_hsum(int id, double *sx);

/* MUTILS_C */
extern int find_opt(int argc, char *argv[], char const *opt);
extern int fdigits(double x);
extern void check_dir(char const *dir);
extern void check_dir_root(char const *dir);
extern int name_size(char const *format, ...);
extern long find_section(char const *title);
extern long find_optional_section(char const *title);
extern long read_line(char const *tag, char const *format, ...);
extern long read_optional_line(char const *tag, char const *format, ...);
extern int count_tokens(char const *tag);
extern long read_iprms(char const *tag, int n, int *iprms);
extern long read_optional_iprms(char const *tag, int n, int *iprms);
extern long read_dprms(char const *tag, int n, double *dprms);
extern long read_optional_dprms(char const *tag, int n, double *dprms);
extern void copy_file(char const *in, char const *out);

extern long const No_Section_Found;

/* UTILS_C */
extern int safe_mod(int x, int y);
extern void *amalloc(size_t size, int p);
extern void afree(void *addr);
extern double amem_use_mb(void);
extern double amem_max_mb(void);
extern int mpi_permanent_tag(void);
extern int mpi_tag(void);
extern void message(char const *format, ...);
extern void mpc_bcast_c(char *buf, int num);
extern void mpc_bcast_d(double *buf, int num);
extern void mpc_bcast_i(int *buf, int num);
extern void mpc_gsum_d(double const *src, double *dst, int num);
extern void mpc_print_info(void);
extern double square_dble(double);
extern double sinc_dble(double);
extern double smear_xi0_dble(double);
extern double smear_xi1_dble(double);
extern void mul_icomplex(complex_dble *);
extern void mul_assign_scalar_complex(double, complex_dble *);
extern int is_equal_f(float, float);
extern int not_equal_f(float, float);
extern int is_equal_d(double, double);
extern int not_equal_d(double, double);

/* WSPACE_C */
extern void alloc_wud(int n);
extern su3_dble **reserve_wud(int n);
extern int release_wud(void);
extern int wud_size(void);
extern void alloc_wfd(int n);
extern su3_alg_dble **reserve_wfd(int n);
extern int release_wfd(void);
extern int wfd_size(void);
extern void alloc_ws(int n);
extern spinor **reserve_ws(int n);
extern int release_ws(void);
extern int ws_size(void);
extern void alloc_wsd(int n);
extern spinor_dble **reserve_wsd(int n);
extern int release_wsd(void);
extern int wsd_size(void);
extern void alloc_wv(int n);
extern complex **reserve_wv(int n);
extern int release_wv(void);
extern int wv_size(void);
extern void alloc_wvd(int n);
extern complex_dble **reserve_wvd(int n);
extern int release_wvd(void);
extern int wvd_size(void);

#endif
