
/*******************************************************************************
*
* File utils.c
*
* Copyright (C) 2005, 2008, 2011, 2016 Martin Luescher, 2013 Hubert Simma
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Collection of basic utility programs
*
* The externally accessible functions are
*
*   int safe_mod(int x,int y)
*     Returns x mod y, where y is assumed positive and x can have any
*     sign. The return value is in the interval [0,y)
*
*   void *amalloc(size_t size,int p)
*     Allocates an aligned memory area of "size" bytes, with a starting
*     address (the return value) that is an integer multiple of 2^p. A
*     NULL pointer is returned if the allocation was not successful
*
*   void afree(void *addr)
*     Frees the aligned memory area at address "addr" that was previously
*     allocated using amalloc. If the memory space at this address was
*     already freed using afree, or if the address does not match an
*     address previously returned by amalloc, the program does not do
*     anything
*
*   double amem_use_mb()
*     Returns the current size in MByte of the total memory area which
*     has been allocated through amalloc (but not yet freed by afree)
*
*   double amem_max_mb()
*     Returns the maximum size in MByte of the total memory area which
*     has been allocated through amalloc at any moment since the start
*     of the program execution and until the current call to amem_max_mb()
*
*   int mpi_permanent_tag(void)
*     Returns a new send tag that is guaranteed to be unique and which
*     is therefore suitable for use in permanent communication requests.
*     The available number of tags of this kind is 16384
*
*   int mpi_tag(void)
*     Returns a new send tag for use in non-permanent communications.
*     Note that the counter for these tags wraps around after 16384
*     tags have been delivered
*
*   void message(char *format,...)
*     Prints a message from process 0 to stdout. The usage and argument
*     list is the same as in the case of the printf function
*
*   void mpc_gsum_d(double *src, double *dst, int num)
*     Compute global sum of num double precision values from src and
*     store results in dst
*
*   void mpc_bcast_c(char *buf, int num)
*     Broadcast num char values from buf of rank 0
*
*   void mpc_bcast_d(double *buf, int num)
*     Broadcast num double values from buf of rank 0
*
*   void mpc_bcast_i(int *buf, int num)
*     Broadcast num int values from buf of rank 0
*
*   void mpc_print_info()
*     Print info how mpc functions are implemented
*
* Notes:
*
* If an error is detected in a locally operating program, it is not possible
* to stop the global program immediately in a decent manner. Such errors
* should first be recorded by the error_loc() function, and the main program
* may later be aborted by calling error_chk() outside the program where the
* error was detected.
*
* An important point to note is that the programs error() and error_chk()
* require a global communication. They must therefore be called on all
* processes simultaneously.
*
*******************************************************************************/

#define UTILS_C

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include "utils.h"
#include "global.h"

#define MAX_TAG 32767
#define MAX_PERMANENT_TAG MAX_TAG / 2

#define MPC_BUF_LEN 2048

static int mpcBuf[MPC_BUF_LEN];
static int mpcRank = -1;

static int pcmn_cnt = -1, cmn_cnt = MAX_TAG;

static long long int amem_use = 0;
static long long int amem_max = 0;

struct addr_t
{
  char *addr;
  char *true_addr;
  size_t true_size;
  struct addr_t *next;
};

static struct addr_t *first = NULL;

int safe_mod(int x, int y)
{
  if (x >= 0)
    return x % y;
  else
    return (y - (abs(x) % y)) % y;
}

void *amalloc(size_t size, int p)
{
  int shift;
  char *true_addr, *addr;
  unsigned long mask;
  struct addr_t *new;

  if ((size <= 0) || (p < 0))
    return (NULL);

  shift = 1 << p;
  mask = (unsigned long)(shift - 1);

  true_addr = malloc(size + shift);
  new = malloc(sizeof(*first));

  if ((true_addr == NULL) || (new == NULL)) {
    free(true_addr);
    free(new);
    return (NULL);
  }

  addr = (char *)(((unsigned long)(true_addr + shift)) & (~mask));
  (*new).addr = addr;
  (*new).true_addr = true_addr;
  (*new).true_size = size + shift;
  (*new).next = first;
  first = new;

  amem_use += size + shift;
  if (amem_max < amem_use)
    amem_max = amem_use;

  return (void *)(addr);
}

void afree(void *addr)
{
  struct addr_t *p, *q;

  q = NULL;

  for (p = first; p != NULL; p = (*p).next) {
    if ((*p).addr == addr) {
      if (q != NULL)
        (*q).next = (*p).next;
      else
        first = (*p).next;

      free((*p).true_addr);
      amem_use -= (*p).true_size;
      free(p);
      return;
    }

    q = p;
  }
}

double amem_use_mb() { return ((double)amem_use) / (1024 * 1024); }

double amem_max_mb() { return ((double)amem_max) / (1024 * 1024); }

int mpi_permanent_tag(void)
{
  if (pcmn_cnt < MAX_PERMANENT_TAG)
    pcmn_cnt += 1;
  else
    error_loc(1, 1, "mpi_permanent_tag [utils.c]",
              "Requested more than 16384 tags");

  return pcmn_cnt;
}

int mpi_tag(void)
{
  if (cmn_cnt == MAX_TAG)
    cmn_cnt = MAX_PERMANENT_TAG;

  cmn_cnt += 1;

  return cmn_cnt;
}

void message(char *format, ...)
{
  int my_rank;
  va_list args;

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
  }
}

#undef USE_MPI_BCAST
#define USE_MPI_ALLREDUCE

void mpc_print_info()
{
#ifdef USE_MPI_BCAST
  message("mpc_bcast implemented as MPI_Bcast\n");
#else
  message("mpc_bcast implemented as MPI_Allreduce\n");
#endif

#ifdef USE_MPI_ALLREDUCE
  message("mpc_gsum_d implemented as MPI_Allreduce\n");
#else
  message("mpc_gsum_d implemented as MPI_Reduce + mpc_bcast\n");
#endif
}

void mpc_bcast_c(char *buf, int num)
{
#ifdef USE_MPI_BCAST
  MPI_Bcast(buf, num, MPI_CHAR, 0, MPI_COMM_WORLD);
#else
  int i, nint;
  char *pc;
  int *pi;
  nint = (sizeof(char) * num) / sizeof(int);
  while (nint * sizeof(int) < num * sizeof(char))
    nint++;
  pc = (char *)mpcBuf;
  pi = (int *)mpcBuf;
  if (mpcRank < 0)
    MPI_Comm_rank(MPI_COMM_WORLD, &mpcRank);
  error_root(nint > MPC_BUF_LEN, 0, "mpc_bcast_c [utils.c]",
             "Too many elements: %d", num);
  if (mpcRank == 0) {
    for (i = 0; i < num; i++)
      pc[i] = buf[i];
  } else {
    for (i = 0; i < nint; i++)
      pi[i] = 0;
  }
  MPI_Allreduce((int *)mpcBuf, (int *)buf, nint, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
#endif
}

void mpc_bcast_d(double *buf, int num)
{
#ifdef USE_MPI_BCAST
  MPI_Bcast(buf, num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
  int i;
  double *p = (double *)mpcBuf;
  if (mpcRank < 0)
    MPI_Comm_rank(MPI_COMM_WORLD, &mpcRank);
  error_root(num * sizeof(double) > MPC_BUF_LEN * sizeof(int), 0,
             "mpc_bcast_d [utils.c]", "Too many elements: %d", num);
  if (mpcRank == 0) {
    for (i = 0; i < num; i++)
      p[i] = buf[i];
  } else {
    for (i = 0; i < num; i++)
      p[i] = 0;
  }
  MPI_Allreduce((double *)mpcBuf, buf, num, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
#endif
}

void mpc_bcast_i(int *buf, int num)
{
#ifdef USE_MPI_BCAST
  MPI_Bcast(buf, num, MPI_INT, 0, MPI_COMM_WORLD);
#else
  int i;
  if (mpcRank < 0)
    MPI_Comm_rank(MPI_COMM_WORLD, &mpcRank);
  error_root(num > MPC_BUF_LEN, 0, "mpc_bcast_i [utils.c]",
             "Too many elements: %d", num);
  if (mpcRank == 0) {
    for (i = 0; i < num; i++)
      mpcBuf[i] = buf[i];
  } else {
    for (i = 0; i < num; i++)
      mpcBuf[i] = 0;
  }
  MPI_Allreduce(mpcBuf, buf, num, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
}

void mpc_gsum_d(double *src, double *dst, int num)
{
#ifdef USE_MPI_ALLREDUCE
  MPI_Allreduce(src, dst, num, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
  MPI_Reduce(src, dst, num, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  mpc_bcast_d(dst, num);
#endif
}

/* Generic mathematical functions */

double square_dble(double d)
{
  return d*d;
}

double sinc_dble(double d)
{
  double dd;

  if (fabs(d) < 0.05) {
    dd = square_dble(d);
    return 1. - dd/6*(1. - dd/20*(1.-dd/42));
  } else {
    return sin(d)/d;
  }
}

double smear_xi0_dble(double d)
{
  return sinc_dble(d);
}

double smear_xi1_dble(double d)
{
  double dd;
  dd = square_dble(d);

  if (fabs(d) < 0.05) {
    return -(1. - dd/10 *(1. - dd/28 * (1. - dd/54)))/3;
  } else {
    return (cos(d) - sinc_dble(d))/dd;
  }
}

void mul_icomplex(complex_dble *c)
{
  double tmp;

  tmp = (*c).re;

  (*c).re = -(*c).im;
  (*c).im = tmp;
}

void mul_assign_scalar_complex(double d, complex_dble *c)
{
  (*c).re = d * (*c).re;
  (*c).im = d * (*c).im;
}
