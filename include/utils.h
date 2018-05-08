
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
typedef short int openqcd_utils__stdint_t;
typedef unsigned short int openqcd_utils__stduint_t;
#elif (INT_MAX == 0x7fffffff)
typedef int openqcd_utils__stdint_t;
typedef unsigned int openqcd_utils__stduint_t;
#elif (LONG_MAX == 0x7fffffff)
typedef long int openqcd_utils__stdint_t;
typedef unsigned long int openqcd_utils__stduint_t;
#else
#error : There is no four-byte integer type on this machine
#endif

#undef openqcd_utils__UNKNOWN_ENDIAN
#undef openqcd_utils__LITTLE_ENDIAN
#undef openqcd_utils__BIG_ENDIAN

#define openqcd_utils__UNKNOWN_ENDIAN 0
#define openqcd_utils__LITTLE_ENDIAN 1
#define openqcd_utils__BIG_ENDIAN 2

#undef openqcd_utils__IMAX
#define openqcd_utils__IMAX(n, m) ((n) + ((m) - (n)) * ((m) > (n)))

typedef enum
{
  ALL_PTS,
  EVEN_PTS,
  ODD_PTS,
  NO_PTS,
  PT_SETS
} openqcd_utils__ptset_t;

/* ENDIAN_C */
extern int openqcd_utils__endianness(void);
extern void openqcd_utils__bswap_int(int n, void *a);
extern void openqcd_utils__bswap_double(int n, void *a);

/* ERROR_C */
extern void openqcd_utils__set_error_file(char const *path, int loc_flag);
extern void openqcd_utils__error(int test, int no, char const *name,
                                 char const *format, ...);
extern void openqcd_utils__error_root(int test, int no, char const *name,
                                      char const *format, ...);
extern void openqcd_utils__error_loc(int test, int no, char const *name,
                                     char const *format, ...);

/* HSUM_C */
extern int openqcd_utils__init_hsum(int n);
extern void openqcd_utils__reset_hsum(int id);
extern void openqcd_utils__add_to_hsum(int id, double const *x);
extern void openqcd_utils__local_hsum(int id, double *sx);
extern void openqcd_utils__global_hsum(int id, double *sx);

/* MUTILS_C */
extern int openqcd_utils__find_opt(int argc, char *argv[], char const *opt);
extern int openqcd_utils__fdigits(double x);
extern void openqcd_utils__check_dir(char const *dir);
extern void openqcd_utils__check_dir_root(char const *dir);
extern int openqcd_utils__name_size(char const *format, ...);
extern long openqcd_utils__find_section(char const *title);
extern long openqcd_utils__find_optional_section(char const *title);
extern long openqcd_utils__read_line(char const *tag, char const *format, ...);
extern long openqcd_utils__read_optional_line(char const *tag,
                                              char const *format, ...);
extern int openqcd_utils__count_tokens(char const *tag);
extern long openqcd_utils__read_iprms(char const *tag, int n, int *iprms);
extern long openqcd_utils__read_optional_iprms(char const *tag, int n,
                                               int *iprms);
extern long openqcd_utils__read_dprms(char const *tag, int n, double *dprms);
extern long openqcd_utils__read_optional_dprms(char const *tag, int n,
                                               double *dprms);
extern void openqcd_utils__copy_file(char const *in, char const *out);

extern long const openqcd_utils__No_Section_Found;

/* UTILS_C */
extern int openqcd_utils__safe_mod(int x, int y);
extern void *openqcd_utils__amalloc(size_t size, int p);
extern void openqcd_utils__afree(void *addr);
extern double openqcd_utils__amem_use_mb(void);
extern double openqcd_utils__amem_max_mb(void);
extern int openqcd_utils__mpi_permanent_tag(void);
extern int openqcd_utils__mpi_tag(void);
extern void openqcd_utils__message(char const *format, ...);
extern void openqcd_utils__mpc_bcast_c(char *buf, int num);
extern void openqcd_utils__mpc_bcast_d(double *buf, int num);
extern void openqcd_utils__mpc_bcast_i(int *buf, int num);
extern void openqcd_utils__mpc_gsum_d(double const *src, double *dst, int num);
extern void openqcd_utils__mpc_print_info(void);
extern double openqcd_utils__square_dble(double);
extern double openqcd_utils__sinc_dble(double);
extern double openqcd_utils__smear_xi0_dble(double);
extern double openqcd_utils__smear_xi1_dble(double);
extern void openqcd_utils__mul_icomplex(openqcd__complex_dble *);
extern void openqcd_utils__mul_assign_scalar_complex(double,
                                                     openqcd__complex_dble *);
extern int openqcd_utils__is_equal_f(float, float);
extern int openqcd_utils__not_equal_f(float, float);
extern int openqcd_utils__is_equal_d(double, double);
extern int openqcd_utils__not_equal_d(double, double);

/* WSPACE_C */
extern void openqcd_utils__alloc_wud(int n);
extern openqcd__su3_dble **openqcd_utils__reserve_wud(int n);
extern int openqcd_utils__release_wud(void);
extern int openqcd_utils__wud_size(void);
extern void openqcd_utils__alloc_wfd(int n);
extern openqcd__su3_alg_dble **openqcd_utils__reserve_wfd(int n);
extern int openqcd_utils__release_wfd(void);
extern int openqcd_utils__wfd_size(void);
extern void openqcd_utils__alloc_ws(int n);
extern openqcd__spinor **openqcd_utils__reserve_ws(int n);
extern int openqcd_utils__release_ws(void);
extern int openqcd_utils__ws_size(void);
extern void openqcd_utils__alloc_wsd(int n);
extern openqcd__spinor_dble **openqcd_utils__reserve_wsd(int n);
extern int openqcd_utils__release_wsd(void);
extern int openqcd_utils__wsd_size(void);
extern void openqcd_utils__alloc_wv(int n);
extern openqcd__complex **openqcd_utils__reserve_wv(int n);
extern int openqcd_utils__release_wv(void);
extern int openqcd_utils__wv_size(void);
extern void openqcd_utils__alloc_wvd(int n);
extern openqcd__complex_dble **openqcd_utils__reserve_wvd(int n);
extern int openqcd_utils__release_wvd(void);
extern int openqcd_utils__wvd_size(void);

#if defined(OPENQCD_INTERNAL)

#define stdint_t openqcd_utils__stdint_t
#define stduint_t openqcd_utils__stduint_t

#define UNKNOWN_ENDIAN openqcd_utils__UNKNOWN_ENDIAN
#define LITTLE_ENDIAN openqcd_utils__LITTLE_ENDIAN
#define BIG_ENDIAN openqcd_utils__BIG_ENDIAN
#define IMAX openqcd_utils__IMAX
#define ptset_t openqcd_utils__ptset_t
#define No_Section_Found openqcd_utils__No_Section_Found

/* ENDIAN_C */
#define endianness(...) openqcd_utils__endianness(__VA_ARGS__)
#define bswap_int(...) openqcd_utils__bswap_int(__VA_ARGS__)
#define bswap_double(...) openqcd_utils__bswap_double(__VA_ARGS__)

/* ERROR_C */
#define set_error_file(...) openqcd_utils__set_error_file(__VA_ARGS__)
#define error(...) openqcd_utils__error(__VA_ARGS__)
#define error_root(...) openqcd_utils__error_root(__VA_ARGS__)
#define error_loc(...) openqcd_utils__error_loc(__VA_ARGS__)

/* HSUM_C */
#define init_hsum(...) openqcd_utils__init_hsum(__VA_ARGS__)
#define reset_hsum(...) openqcd_utils__reset_hsum(__VA_ARGS__)
#define add_to_hsum(...) openqcd_utils__add_to_hsum(__VA_ARGS__)
#define local_hsum(...) openqcd_utils__local_hsum(__VA_ARGS__)
#define global_hsum(...) openqcd_utils__global_hsum(__VA_ARGS__)

/* MUTILS_C */
#define find_opt(...) openqcd_utils__find_opt(__VA_ARGS__)
#define fdigits(...) openqcd_utils__fdigits(__VA_ARGS__)
#define check_dir(...) openqcd_utils__check_dir(__VA_ARGS__)
#define check_dir_root(...) openqcd_utils__check_dir_root(__VA_ARGS__)
#define name_size(...) openqcd_utils__name_size(__VA_ARGS__)
#define find_section(...) openqcd_utils__find_section(__VA_ARGS__)
#define find_optional_section(...)                                             \
  openqcd_utils__find_optional_section(__VA_ARGS__)
#define read_line(...) openqcd_utils__read_line(__VA_ARGS__)
#define read_optional_line(...) openqcd_utils__read_optional_line(__VA_ARGS__)
#define count_tokens(...) openqcd_utils__count_tokens(__VA_ARGS__)
#define read_iprms(...) openqcd_utils__read_iprms(__VA_ARGS__)
#define read_optional_iprms(...) openqcd_utils__read_optional_iprms(__VA_ARGS__)
#define read_dprms(...) openqcd_utils__read_dprms(__VA_ARGS__)
#define read_optional_dprms(...) openqcd_utils__read_optional_dprms(__VA_ARGS__)
#define copy_file(...) openqcd_utils__copy_file(__VA_ARGS__)

/* UTILS_C */
#define safe_mod(...) openqcd_utils__safe_mod(__VA_ARGS__)
#define amalloc(...) openqcd_utils__amalloc(__VA_ARGS__)
#define afree(...) openqcd_utils__afree(__VA_ARGS__)
#define amem_use_mb(...) openqcd_utils__amem_use_mb(__VA_ARGS__)
#define amem_max_mb(...) openqcd_utils__amem_max_mb(__VA_ARGS__)
#define mpi_permanent_tag(...) openqcd_utils__mpi_permanent_tag(__VA_ARGS__)
#define mpi_tag(...) openqcd_utils__mpi_tag(__VA_ARGS__)
#define message(...) openqcd_utils__message(__VA_ARGS__)
#define mpc_bcast_c(...) openqcd_utils__mpc_bcast_c(__VA_ARGS__)
#define mpc_bcast_d(...) openqcd_utils__mpc_bcast_d(__VA_ARGS__)
#define mpc_bcast_i(...) openqcd_utils__mpc_bcast_i(__VA_ARGS__)
#define mpc_gsum_d(...) openqcd_utils__mpc_gsum_d(__VA_ARGS__)
#define mpc_print_info(...) openqcd_utils__mpc_print_info(__VA_ARGS__)
#define square_dble(...) openqcd_utils__square_dble(__VA_ARGS__)
#define sinc_dble(...) openqcd_utils__sinc_dble(__VA_ARGS__)
#define smear_xi0_dble(...) openqcd_utils__smear_xi0_dble(__VA_ARGS__)
#define smear_xi1_dble(...) openqcd_utils__smear_xi1_dble(__VA_ARGS__)
#define mul_icomplex(...) openqcd_utils__mul_icomplex(__VA_ARGS__)
#define mul_assign_scalar_complex(...)                                         \
  openqcd_utils__mul_assign_scalar_complex(__VA_ARGS__)
#define is_equal_f(...) openqcd_utils__is_equal_f(__VA_ARGS__)
#define not_equal_f(...) openqcd_utils__not_equal_f(__VA_ARGS__)
#define is_equal_d(...) openqcd_utils__is_equal_d(__VA_ARGS__)
#define not_equal_d(...) openqcd_utils__not_equal_d(__VA_ARGS__)

/* WSPACE_C */
#define alloc_wud(...) openqcd_utils__alloc_wud(__VA_ARGS__)
#define reserve_wud(...) openqcd_utils__reserve_wud(__VA_ARGS__)
#define release_wud(...) openqcd_utils__release_wud(__VA_ARGS__)
#define wud_size(...) openqcd_utils__wud_size(__VA_ARGS__)
#define alloc_wfd(...) openqcd_utils__alloc_wfd(__VA_ARGS__)
#define reserve_wfd(...) openqcd_utils__reserve_wfd(__VA_ARGS__)
#define release_wfd(...) openqcd_utils__release_wfd(__VA_ARGS__)
#define wfd_size(...) openqcd_utils__wfd_size(__VA_ARGS__)
#define alloc_ws(...) openqcd_utils__alloc_ws(__VA_ARGS__)
#define reserve_ws(...) openqcd_utils__reserve_ws(__VA_ARGS__)
#define release_ws(...) openqcd_utils__release_ws(__VA_ARGS__)
#define ws_size(...) openqcd_utils__ws_size(__VA_ARGS__)
#define alloc_wsd(...) openqcd_utils__alloc_wsd(__VA_ARGS__)
#define reserve_wsd(...) openqcd_utils__reserve_wsd(__VA_ARGS__)
#define release_wsd(...) openqcd_utils__release_wsd(__VA_ARGS__)
#define wsd_size(...) openqcd_utils__wsd_size(__VA_ARGS__)
#define alloc_wv(...) openqcd_utils__alloc_wv(__VA_ARGS__)
#define reserve_wv(...) openqcd_utils__reserve_wv(__VA_ARGS__)
#define release_wv(...) openqcd_utils__release_wv(__VA_ARGS__)
#define wv_size(...) openqcd_utils__wv_size(__VA_ARGS__)
#define alloc_wvd(...) openqcd_utils__alloc_wvd(__VA_ARGS__)
#define reserve_wvd(...) openqcd_utils__reserve_wvd(__VA_ARGS__)
#define release_wvd(...) openqcd_utils__release_wvd(__VA_ARGS__)
#define wvd_size(...) openqcd_utils__wvd_size(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif
