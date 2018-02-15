
/*******************************************************************************
*
* File flags.c
*
* Copyright (C) 2009, 2011, 2012, 2016 Martin Luescher, Isabel Campos
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Flags data base input and query programs
*
* The externally accessible functions are
*
*   void set_flags(event_t event)
*     Reports an event to the data base, where some of the global field
*     arrays are changed.
*
*   void set_grid_flags(blk_grid_t grid,event_t event)
*     Reports an event to the data base, where some of the field arrays
*     on the specified block grid are changed.
*
*   int query_flags(query_t query)
*     Queries the data base on the status of the global field arrays.
*     The program returns 1 or 0 depending on whether the answer to the
*     specified query is "yes" or "no". If the query is unknown to the
*     the data base, the program returns -1.
*
*   int query_grid_flags(blk_grid_t grid,query_t query)
*     Queries the data base on the status of the field arrays on the
*     specified block grid. The program returns 1 or 0 depending on
*     whether the answer to the specified query is "yes" or "no". If
*     the query is unknown to the data base, the program returns -1.
*
*   void print_flags(void)
*     Prints the current values of all flags describing the state of
*     the global field arrays to stdout on process 0.
*
*   void print_grid_flags(blk_grid_t grid)
*     Prints the current values of all flags describing the state of
*     the field arrays on the specified block grid to stdout on
*     process 0.
*
* Notes:
*
* The programs set_flags() and set_grid_flags() perform global operations
* and must be called on all processes simultaneously. As a consequence,
* the contents of the data base is the same everywhere. All other programs
* in this module can be called locally.
*
* The possible events and queries are defined in the header file flags.h.
* The associated actions are defined in the *.h files in the include/flags
* directory (application programs do not need to include these).
*
* For further explanations, see the file README.flags in this directory.
*
*******************************************************************************/

#define FLAGS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include "mpi.h"
#include "utils.h"
#include "flags.h"
#include "global.h"

#define NFLGS (10 + 4 * (int)(BLK_GRIDS))

const int SMEARED_STATE_BIT = 1;
const int PHASE_STATE_BIT = (1 << 1);

typedef struct
{
  int tag;
  int state;
} cfg_state_t;

typedef struct
{
  cfg_state_t state;
  int even_flag, odd_flag;
} sw_state_t;

static struct
{
  cfg_state_t u, ud, udbuf;
  cfg_state_t bstap, fts;
  sw_state_t sw, swd;
  cfg_state_t aw, awh;
  int smeared_tag;
} lat = {{0, 0},         {0, 0},         {0, 0}, {0, 0}, {0, 0},
         {{0, 0}, 0, 0}, {{0, 0}, 0, 0}, {0, 0}, {0, 0}, 0};

typedef struct
{
  int shf;
  cfg_state_t u, ud;
  sw_state_t sw, swd;
} grid_flags_t;

static int init = 0, tag = 0;
static int flgs[NFLGS];
static grid_flags_t gfs[(int)(BLK_GRIDS) + 1] = {
    {0x0, {0, 0}, {0, 0}, {{0, 0}, 0, 0}, {{0, 0}, 0, 0}}};
static grid_flags_t *gf;

static void set_flgs(void)
{
  int n, igr;

  flgs[0] = lat.u.tag;
  flgs[1] = lat.ud.tag;
  flgs[2] = lat.udbuf.tag;
  flgs[3] = lat.bstap.tag;
  flgs[4] = lat.fts.tag;
  flgs[5] = lat.sw.state.tag;
  flgs[6] = lat.swd.state.tag;
  flgs[7] = lat.aw.tag;
  flgs[8] = lat.awh.tag;
  flgs[9] = lat.smeared_tag;

  n = 10;

  for (igr = 0; igr < (int)(BLK_GRIDS); igr++) {
    flgs[n++] = gfs[igr].u.tag;
    flgs[n++] = gfs[igr].ud.tag;
    flgs[n++] = gfs[igr].sw.state.tag;
    flgs[n++] = gfs[igr].swd.state.tag;
  }
}

static void find_gap(int *a, int *d)
{
  int k, l;
  int fk, h, hmax;

  (*a) = 0;
  (*d) = INT_MAX;

  for (k = 0; k < NFLGS; k++) {
    h = flgs[k];

    if ((h > 0) && (h < (*d)))
      (*d) = h;
  }

  for (k = 0; k < NFLGS; k++) {
    fk = flgs[k];
    hmax = INT_MAX - fk;

    for (l = 0; l < NFLGS; l++) {
      h = flgs[l] - fk;

      if ((h > 0) && (h < hmax)) {
        hmax = h;

        if (h <= (*d))
          break;
      }
    }

    if (hmax > (*d)) {
      (*a) = fk;
      (*d) = hmax;
    }
  }
}

static void compress_flags(void)
{
  int k, a, d;
  int n, igr;

  set_flgs();
  find_gap(&a, &d);
  d -= 1;

  for (k = 0; k < NFLGS; k++) {
    if (flgs[k] > a)
      flgs[k] -= d;
  }

  lat.u.tag = flgs[0];
  lat.ud.tag = flgs[1];
  lat.udbuf.tag = flgs[2];
  lat.bstap.tag = flgs[3];
  lat.fts.tag = flgs[4];
  lat.sw.state.tag = flgs[5];
  lat.swd.state.tag = flgs[6];
  lat.aw.tag = flgs[7];
  lat.awh.tag = flgs[8];
  lat.smeared_tag = flgs[9];

  n = 10;

  for (igr = 0; igr < (int)(BLK_GRIDS); igr++) {
    gfs[igr].u.tag = flgs[n++];
    gfs[igr].ud.tag = flgs[n++];
    gfs[igr].sw.state.tag = flgs[n++];
    gfs[igr].swd.state.tag = flgs[n++];
  }

  tag -= d;
}

static int next_tag(void)
{
  if (tag == INT_MAX)
    compress_flags();
  tag += 1;

  return tag;
}

static void enable_bit_state(int *bitmap, int state_bit)
{
  *bitmap |= state_bit;
}

static void disable_bit_state(int *bitmap, int state_bit)
{
  *bitmap &= (~state_bit);
}

static int check_bit_state(int bitmap, int state_bit)
{
  return (bitmap & state_bit) != 0;
}

#include "flags/events.h"
#include "flags/grid_events.h"
#include "flags/queries.h"
#include "flags/grid_queries.h"

static void set_arrays(void)
{
  int igr;

  for (igr = 1; igr <= (int)(BLK_GRIDS); igr++)
    gfs[igr] = gfs[0];

  gfs[(int)(SAP_BLOCKS)].shf = 0x0;
  gfs[(int)(DFL_BLOCKS)].shf = 0x2;

  set_events();
  set_grid_events();
  set_queries();
  set_grid_queries();

  init = 1;
}

void set_flags(event_t event)
{
  int iprms[1], iev;

  if (init == 0)
    set_arrays();

  iev = (int)(event);

  if (NPROC > 1) {
    iprms[0] = iev;

    MPI_Bcast(iprms, 1, MPI_INT, 0, MPI_COMM_WORLD);

    error(iprms[0] != iev, 1, "set_flags [flags.c]", "Parameter is not global");
  }

  if (event_fcts[iev] == NULL)
    error_root(1, 1, "set_flags [flags.c]", "No action associated to event");
  else
    event_fcts[iev]();
}

void set_grid_flags(blk_grid_t grid, event_t event)
{
  int iprms[2], igr, iev;

  if (init == 0)
    set_arrays();

  igr = (int)(grid);
  iev = (int)(event);

  if (NPROC > 1) {
    iprms[0] = igr;
    iprms[1] = iev;

    MPI_Bcast(iprms, 2, MPI_INT, 0, MPI_COMM_WORLD);

    error((iprms[0] != igr) || (iprms[1] != iev), 1, "set_grid_flags [flags.c]",
          "Parameters are not global");
  }

  if (grid == BLK_GRIDS)
    error_root(1, 1, "set_grid_flags [flags.c]",
               "BLK_GRIDS is a dummy block grid");

  if (grid_event_fcts[iev] == NULL)
    error_root(1, 1, "set_grid_flags [flags.c]",
               "No action associated to event");
  else {
    gf = gfs + igr;
    grid_event_fcts[iev]();
  }
}

int query_flags(query_t query)
{
  int iqr;

  if (init == 0)
    set_arrays();

  iqr = (int)(query);

  if (query_fcts[iqr] == NULL) {
    error_loc(1, 1, "query_flags [flags.c]", "No response to query");
    return -1;
  } else
    return query_fcts[iqr]();
}

int query_grid_flags(blk_grid_t grid, query_t query)
{
  int iqr;

  if (init == 0)
    set_arrays();

  iqr = (int)(query);

  if (grid_query_fcts[iqr] == NULL) {
    error_loc(1, 1, "query_grid_flags [flags.c]", "No response to query");
    return -1;
  } else {
    gf = gfs + (int)(grid);
    return grid_query_fcts[iqr]();
  }
}

void print_flags(void)
{
  int my_rank;

  if (init == 0)
    set_arrays();

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    printf("Full lattice flags:\n");
    printf("u           = %d(%d)\n", lat.u.tag, lat.u.state);
    printf("ud,udbuf    = %d(%d),%d(%d)\n", lat.ud.tag, lat.ud.state,
           lat.udbuf.tag, lat.udbuf.state);
    printf("bstap,fts   = %d(%d),%d(%d)\n", lat.bstap.tag, lat.bstap.state,
           lat.fts.tag, lat.fts.state);
    printf("sw          = %d(%d),%d,%d\n", lat.sw.state.tag, lat.sw.state.state,
           lat.sw.even_flag, lat.sw.odd_flag);
    printf("swd         = %d(%d),%d,%d\n", lat.swd.state.tag,
           lat.swd.state.state, lat.swd.even_flag, lat.swd.odd_flag);
    printf("aw,awh      = %d(%d),%d(%d)\n", lat.aw.tag, lat.aw.state,
           lat.awh.tag, lat.awh.state);
    printf("smeared_tag = %d\n", lat.smeared_tag);
    printf("\n");
  }
}

void print_grid_flags(blk_grid_t grid)
{
  int my_rank;

  if (init == 0)
    set_arrays();

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    gf = gfs + (int)(grid);

    if (grid == SAP_BLOCKS)
      printf("Flags on the SAP block grid:\n");
    else if (grid == DFL_BLOCKS)
      printf("Flags on the DFL block grid:\n");
    else
      error_root(1, 1, "print_grid_flags [flags.c]", "Unknown block grid");

    printf("shf        = %#x\n", (*gf).shf);
    printf("u          = %d(%d)\n", (*gf).u.tag, (*gf).u.state);
    printf("ud         = %d(%d)\n", (*gf).ud.tag, (*gf).ud.state);
    printf("sw         = %d(%d),%d,%d\n", (*gf).sw.state.tag,
           (*gf).sw.state.state, (*gf).sw.even_flag, (*gf).sw.odd_flag);
    printf("swd        = %d(%d),%d,%d\n", (*gf).swd.state.tag,
           (*gf).swd.state.state, (*gf).swd.even_flag, (*gf).swd.odd_flag);
    printf("\n");
  }
}
