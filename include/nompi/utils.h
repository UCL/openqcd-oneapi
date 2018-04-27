
/*******************************************************************************
 *
 * File nompi/utils.h
 *
 * Copyright (C) 2009, 2010, 2011 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef UTILS_H
#define UTILS_H

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define openqcd_utils__NAME_SIZE 128

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
} ptset_t;

/* ENDIAN_C */
extern int openqcd_utils__endianness(void);
extern void openqcd_utils__bswap_int(int n, void *a);
extern void openqcd_utils__bswap_double(int n, void *a);

/* MUTILS_C */
extern int openqcd_utils__find_opt(int argc, char *argv[], char *opt);
extern int openqcd_utils__digits(double x, double dx, char *fmt);
extern int openqcd_utils__fdigits(double x);
extern int openqcd_utils__name_size(char *format, ...);
extern long openqcd_utils__find_section(FILE *stream, char *title);
extern long openqcd_utils__read_line(FILE *stream, char *tag, char *format,
                                     ...);
extern int openqcd_utils__count_tokens(FILE *stream, char *tag);
extern void openqcd_utils__read_iprms(FILE *stream, char *tag, int n,
                                      int *iprms);
extern void openqcd_utils__read_dprms(FILE *stream, char *tag, int n,
                                      double *dprms);

/* UTILS_C */
extern int openqcd_utils__safe_mod(int x, int y);
extern void *openqcd_utils__amalloc(size_t size, int p);
extern void openqcd_utils__afree(void *addr);
extern void openqcd_utils__error(int test, int no, char *name, char *format,
                                 ...);
extern void openqcd_utils__error_root(int test, int no, char *name,
                                      char *format, ...);
extern void openqcd_utils__error_loc(int test, int no, char *name, char *format,
                                     ...);
extern void openqcd_utils__message(char *format, ...);
extern int openqcd_utils__is_equal_f(float, float);
extern int openqcd_utils__not_equal_f(float, float);
extern int openqcd_utils__is_equal_d(double, double);
extern int openqcd_utils__not_equal_d(double, double);

#if defined(OPENQCD_INTERNAL)
#define NAME_SIZE openqcd_utils__NAME_SIZE
#define UNKNOWN_ENDIAN openqcd_utils__UNKNOWN_ENDIAN
#define LITTLE_ENDIAN openqcd_utils__LITTLE_ENDIAN
#define BIG_ENDIAN openqcd_utils__BIG_ENDIAN
#define IMAX openqcd_utils__IMAX
#define ptset_t openqcd_utils__ptset_t

/* ENDIAN_C */
#define endianness(...) openqcd_utils__endianness(__VA_ARGS__)
#define bswap_int(...) openqcd_utils__bswap_int(__VA_ARGS__)
#define bswap_double(...) openqcd_utils__bswap_double(__VA_ARGS__)

/* MUTILS_C */
#define find_opt(...) openqcd_utils__find_opt(__VA_ARGS__)
#define digits(...) openqcd_utils__digits(__VA_ARGS__)
#define fdigits(...) openqcd_utils__fdigits(__VA_ARGS__)
#define name_size(...) openqcd_utils__name_size(__VA_ARGS__)
#define find_section(...) openqcd_utils__find_section(__VA_ARGS__)
#define read_line(...) openqcd_utils__read_line(__VA_ARGS__)
#define count_tokens(...) openqcd_utils__count_tokens(__VA_ARGS__)
#define read_iprms(...) openqcd_utils__read_iprms(__VA_ARGS__)
#define read_dprms(...) openqcd_utils__read_dprms(__VA_ARGS__)

/* UTILS_C */
#define safe_mod(...) openqcd_utils__safe_mod(__VA_ARGS__)
#define amalloc(...) openqcd_utils__amalloc(__VA_ARGS__)
#define afree(...) openqcd_utils__afree(__VA_ARGS__)
#define error(...) openqcd_utils__error(__VA_ARGS__)
#define error_root(...) openqcd_utils__error_root(__VA_ARGS__)
#define error_loc(...) openqcd_utils__error_loc(__VA_ARGS__)
#define message(...) openqcd_utils__message(__VA_ARGS__)
#define is_equal_f(...) openqcd_utils__is_equal_f(__VA_ARGS__)
#define not_equal_f(...) openqcd_utils__not_equal_f(__VA_ARGS__)
#define is_equal_d(...) openqcd_utils__is_equal_d(__VA_ARGS__)
#define not_equal_d(...) openqcd_utils__not_equal_d(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif
