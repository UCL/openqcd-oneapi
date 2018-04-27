
/*******************************************************************************
*
* File utils.c
*
* Copyright (C) 2005, 2008, 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)

* Collection of basic utility programs
*
* The externally accessible functions are
*
*   int safe_mod(int x,int y)
*     Returns x mod y, where y is assumed positive and x can have any
*     sign. The return value is in the interval [0,y)
*
*   void *amalloc(size_t size,int p)
*     Allocates an aligned memory area of "size" bytes, with starting
*     address (the return value) that is an integer multiple of 2^p
*
*   void afree(void *addr)
*     Frees the aligned memory area at address "addr" that was
*     previously allocated using amalloc
*
*   void error(int test,int no,char *name,char *format,...)
*     Checks whether "test"=0 and if not aborts the program gracefully
*     with error number "no" after printing the "name" of the calling
*     program and an error message to stdout. The message is formed using
*     the "format" string and any additional arguments, exactly as in a
*     printf statement
*
*   void error_root(int test,int no,char *name,char *format,...)
*     Same as error(), provided for compatibility
*
*   void error_loc(int test,int no,char *name,char *format,...)
*     Same as error(), provided for compatibility
*
*   void message(char *format,...)
*     Same as printf(), provided for compatibility
*
*******************************************************************************/

#define UTILS_C
#define OPENQCD_INTERNAL

#include "utils.h"
#include <stdarg.h>
#include <stdint.h>

struct addr_t
{
  char *addr;
  char *true_addr;
  struct addr_t *last, *next;
};

static struct addr_t *rpos = NULL;

static const float fixed_epsilon_float = 1e-8;
static const double fixed_epsilon_double = 1e-16;

static const int32_t ulp_epsilon_float = 8;
static const int64_t ulp_epsilon_double = 16;

int safe_mod(int x, int y)
{
  if (x >= 0) {
    return (x % y);
  } else {
    return ((y - (abs(x) % y)) % y);
  }
}

void *amalloc(size_t size, int p)
{
  int shift;
  char *true_addr, *addr;
  unsigned long mask;
  struct addr_t *new, *rnxt;

  if ((size <= 0) || (p < 0)) {
    return (NULL);
  }

  shift = 1 << p;
  mask = (unsigned long)(shift - 1);

  true_addr = malloc(size + shift);
  new = malloc(sizeof(*new));

  if ((true_addr == NULL) || (new == NULL)) {
    free(true_addr);
    free(new);
    return NULL;
  }

  addr = (char *)(((unsigned long)(true_addr + shift)) & (~mask));
  (*new).addr = addr;
  (*new).true_addr = true_addr;

  if (rpos != NULL) {
    rnxt = (*rpos).next;

    (*new).next = rnxt;
    (*rpos).next = new;
    (*rnxt).last = new;
    (*new).last = rpos;
  } else {
    (*new).next = new;
    (*new).last = new;
  }

  rpos = new;

  return (void *)(addr);
}

void afree(void *addr)
{
  struct addr_t *p, *pn, *pl;

  if (rpos != NULL) {
    p = rpos;

    for (;;) {
      if ((*p).addr == addr) {
        pn = (*p).next;
        pl = (*p).last;

        if (pn != p) {
          (*pl).next = pn;
          (*pn).last = pl;
          rpos = pl;
        } else {
          rpos = NULL;
        }

        free((*p).true_addr);
        free(p);
        return;
      }

      p = (*p).next;
      if (p == rpos) {
        return;
      }
    }
  }
}

void error(int test, int no, char *name, char *format, ...)
{
  va_list args;

  if (test != 0) {
    printf("\nError in %s:\n", name);
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\nProgram aborted\n\n");
    exit(no);
  }
}

void error_root(int test, int no, char *name, char *format, ...)
{
  va_list args;

  if (test != 0) {
    printf("\nError in %s:\n", name);
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\nProgram aborted\n\n");
    exit(no);
  }
}

void error_loc(int test, int no, char *name, char *format, ...)
{
  va_list args;

  if (test != 0) {
    printf("\nError in %s:\n", name);
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\nProgram aborted\n\n");
    exit(no);
  }
}

void message(char *format, ...)
{
  va_list args;

  va_start(args, format);
  vprintf(format, args);
  va_end(args);
}

/* Floating point comparison functions */

typedef union
{
  float f;
  int32_t i;
} float_as_int;

typedef union
{
  double d;
  int64_t i;
} double_as_int;

static float fabs_float(float f)
{
  return f * (1 - 2 * (f < 0.0f));
}

static int isnan_float(float f)
{
  return (f != f);
}

static int isnan_double(double d)
{
  return (d != d);
}

static int isinf_float(float f)
{
  return (f > FLT_MAX || f < -FLT_MAX);
}

static int isinf_double(double d)
{
  return (d > DBL_MAX || d < -DBL_MAX);
}

static int32_t max32(void)
{
  int32_t max = 1;
  max <<= 31;
  return max - 1;
}

static int64_t max64(void)
{
  int64_t max = 1;
  max <<= 63;
  return max - 1;
}

/* Return the ULP distance between two floating points */
/* If they differ in sign the value will be max32() */
static int32_t ulp_distance_f(float f1, float f2)
{
  int32_t max, distance;
  float_as_int if1, if2;

  if (f1 == f2) {
    return 0;
  }

  max = max32();

  if (isnan_float(f1) || isnan_double(f2)) {
    return max;
  }

  if (isinf_float(f1) || isinf_float(f2)) {
    return max;
  }

  if1.f = f1;
  if2.f = f2;

  /* Do not compare floats of different signs */
  if ((if1.i < 0) != (if2.i < 0)) {
    return max;
  }

  distance = if1.i - if2.i;

  /* Absolute value of distance */
  if (distance < 0) {
    distance = -distance;
  }

  return distance;
}

/* Return the ULP distance between two double precision floating points */
/* If they differ in sign the value will be max64() */
static int64_t ulp_distance_d(double d1, double d2)
{
  int64_t max, distance;
  double_as_int id1, id2;

  if (d1 == d2) {
    return 0;
  }

  max = max64();

  if (isnan_double(d1) || isnan_double(d2)) {
    return max;
  }

  if (isinf_double(d1) || isinf_double(d2)) {
    return max;
  }

  id1.d = d1;
  id2.d = d2;

  /* Do not compare doubles of different signs */
  if ((id1.i < 0) != (id2.i < 0)) {
    return max;
  }

  distance = id1.i - id2.i;

  /* Absolute value of distance */
  if (distance < 0) {
    distance = -distance;
  }

  return distance;
}

int is_equal_f(float f1, float f2)
{
  if (fabs_float(f1 - f2) < fixed_epsilon_float) {
    return 1;
  } else {
    return ulp_distance_f(f1, f2) <= ulp_epsilon_float;
  }
}

int not_equal_f(float f1, float f2)
{
  return !is_equal_f(f1, f2);
}

int is_equal_d(double d1, double d2)
{
  if (fabs(d1 - d2) < fixed_epsilon_double) {
    return 1;
  } else {
    return ulp_distance_d(d1, d2) <= ulp_epsilon_double;
  }
}

int not_equal_d(double d1, double d2)
{
  return !is_equal_d(d1, d2);
}
