
/*******************************************************************************
 *
 * File grid_queries.h
 *
 * Copyright (C) 2011 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Block grid queries
 *
 *******************************************************************************/

#define GRID_QUERIES_H

#if (defined FLAGS_C)

static int check_grid_state(cfg_state_t const *state1,
                            cfg_state_t const *state2)
{
  return ((*state1).tag == (*state2).tag) &&
         ((*state1).state == (*state2).state);
}

static int (*grid_query_fcts[(int)(QUERIES) + 1])(void) = {NULL};

static int GridQueryUbgrMatchUd(void)
{
  if ((*gf).shf & 0x1) {
    error_loc(1, 1, "GridQueryUbgrMatchUd [grid_queries.h]",
              "Query involving shared fields");
    return -1;
  } else
    return check_grid_state(&(*gf).u, &lat.ud);
}

static int GridQueryUdbgrMatchUd(void)
{
  if ((*gf).shf & 0x2) {
    error_loc(1, 1, "GridQueryUdbgrMatchUd [grid_queries.h]",
              "Query involving shared fields");
    return -1;
  } else
    return check_grid_state(&(*gf).ud, &lat.ud);
}

static int GridQuerySwUp2date(void)
{
  if ((*gf).shf & 0x1) {
    error_loc(1, 1, "GridQuerySwUp2date [grid_queries.h]",
              "Query involving shared fields");
    return -1;
  } else
    return check_grid_state(&(*gf).sw.state, &(*gf).u);
}

static int GridQuerySwEInverted(void)
{
  if ((*gf).shf & 0x1) {
    error_loc(1, 1, "GridQuerySwEInverted [grid_queries.h]",
              "Query involving shared fields");
    return -1;
  } else
    return ((*gf).sw.even_flag == 1);
}

static int GridQuerySwOInverted(void)
{
  if ((*gf).shf & 0x1) {
    error_loc(1, 1, "GridQuerySwOInverted [grid_queries.h]",
              "Query involving shared fields");
    return -1;
  } else
    return ((*gf).sw.odd_flag == 1);
}

static int GridQuerySwdUp2date(void)
{
  if ((*gf).shf & 0x2) {
    error_loc(1, 1, "GridQuerySwdUp2date [grid_queries.h]",
              "Query involving shared fields");
    return -1;
  } else
    return check_grid_state(&(*gf).swd.state, &(*gf).ud);
}

static int GridQuerySwdEInverted(void)
{
  if ((*gf).shf & 0x2) {
    error_loc(1, 1, "GridQuerySwdEInverted [grid_queries.h]",
              "Query involving shared fields");
    return -1;
  } else
    return ((*gf).swd.even_flag == 1);
}

static int GridQuerySwdOInverted(void)
{
  if ((*gf).shf & 0x2) {
    error_loc(1, 1, "GridQuerySwdOInverted [grid_queries.h]",
              "Query involving shared fields");
    return -1;
  } else
    return ((*gf).swd.odd_flag == 1);
}

static void set_grid_queries(void)
{
  grid_query_fcts[(int)(UBGR_MATCH_UD)] = GridQueryUbgrMatchUd;
  grid_query_fcts[(int)(UDBGR_MATCH_UD)] = GridQueryUdbgrMatchUd;
  grid_query_fcts[(int)(SW_UP2DATE)] = GridQuerySwUp2date;
  grid_query_fcts[(int)(SW_E_INVERTED)] = GridQuerySwEInverted;
  grid_query_fcts[(int)(SW_O_INVERTED)] = GridQuerySwOInverted;
  grid_query_fcts[(int)(SWD_UP2DATE)] = GridQuerySwdUp2date;
  grid_query_fcts[(int)(SWD_E_INVERTED)] = GridQuerySwdEInverted;
  grid_query_fcts[(int)(SWD_O_INVERTED)] = GridQuerySwdOInverted;
}

#endif
