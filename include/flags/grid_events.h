
/*******************************************************************************
 *
 * File grid_events.h
 *
 * Copyright (C) 2011 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Block grid events
 *
 *******************************************************************************/

#define GRID_EVENTS_H

#if (defined FLAGS_C)

static void (*grid_event_fcts[(int)(EVENTS) + 1])(void) = {NULL};

static void GridAssignedU2ubgr(void)
{
  if ((*gf).shf & 0x1)
    error_root(1, 1, "GridAssignedU2ubgr [grid_events.h]",
               "Event involving shared fields");
  else
    (*gf).u = lat.u;
}

static void GridAssignedUd2ubgr(void)
{
  if ((*gf).shf & 0x1)
    error_root(1, 1, "GridAssignedUd2ubgr [grid_events.h]",
               "Event involving shared fields");
  else
    (*gf).u = lat.ud;
}

static void GridAssignedUd2udbgr(void)
{
  if ((*gf).shf & 0x2)
    error_root(1, 1, "GridAssignedUd2udbgr [grid_events.h]",
               "Event involving shared fields");
  else
    (*gf).ud = lat.ud;
}

static void GridAssignedSwd2swbgr(void)
{
  if ((*gf).shf & 0x1)
    error_root(1, 1, "GridAssignedSwd2swbgr [grid_events.h]",
               "Event involving shared fields");
  else {
    (*gf).sw.state = lat.swd.state;
    (*gf).sw.even_flag = lat.swd.even_flag;
    (*gf).sw.odd_flag = lat.swd.odd_flag;
  }
}

static void GridAssignedSwd2swdbgr(void)
{
  if ((*gf).shf & 0x2)
    error_root(1, 1, "GridAssignedSwd2swdbgr [grid_events.h]",
               "Event involving shared fields");
  else {
    (*gf).swd.state = lat.swd.state;
    (*gf).swd.even_flag = lat.swd.even_flag;
    (*gf).swd.odd_flag = lat.swd.odd_flag;
  }
}

static void GridInvertedSwdE(void)
{
  if ((*gf).shf & 0x2)
    error_root(1, 1, "GridInvertedSwdE [grid_events.h]",
               "Event involving shared fields");
  else
    (*gf).swd.even_flag ^= 0x1;
}

static void GridInvertedSwdO(void)
{
  if ((*gf).shf & 0x2)
    error_root(1, 1, "GridInvertedSwdO [grid_events.h]",
               "Event involving shared fields");
  else
    (*gf).swd.odd_flag ^= 0x1;
}

static void GridInvertedSwE(void)
{
  if ((*gf).shf & 0x1)
    error_root(1, 1, "GridInvertedSwE [grid_events.h]",
               "Event involving shared fields");
  else
    (*gf).sw.even_flag ^= 0x1;
}

static void GridInvertedSwO(void)
{
  if ((*gf).shf & 0x1)
    error_root(1, 1, "GridInvertedSwO [grid_events.h]",
               "Event involving shared fields");
  else
    (*gf).sw.odd_flag ^= 0x1;
}

static void GridErasedSw(void)
{
  if ((*gf).shf & 0x1)
    error_root(1, 1, "GridErasedSw [grid_events.h]",
               "Event involving shared fields");
  else {
    (*gf).sw.state.tag = 0;
    (*gf).sw.state.state = 0;
    (*gf).sw.even_flag = 0;
    (*gf).sw.odd_flag = 0;
  }
}

static void GridErasedSwd(void)
{
  if ((*gf).shf & 0x2)
    error_root(1, 1, "GridErasedSwd [grid_events.h]",
               "Event involving shared fields");
  else {
    (*gf).swd.state.tag = 0;
    (*gf).swd.state.state = 0;
    (*gf).swd.even_flag = 0;
    (*gf).swd.odd_flag = 0;
  }
}

static void set_grid_events(void)
{
  grid_event_fcts[(int)(ASSIGNED_U2UBGR)] = GridAssignedU2ubgr;
  grid_event_fcts[(int)(ASSIGNED_UD2UBGR)] = GridAssignedUd2ubgr;
  grid_event_fcts[(int)(ASSIGNED_UD2UDBGR)] = GridAssignedUd2udbgr;
  grid_event_fcts[(int)(ASSIGNED_SWD2SWBGR)] = GridAssignedSwd2swbgr;
  grid_event_fcts[(int)(ASSIGNED_SWD2SWDBGR)] = GridAssignedSwd2swdbgr;
  grid_event_fcts[(int)(INVERTED_SWD_E)] = GridInvertedSwdE;
  grid_event_fcts[(int)(INVERTED_SWD_O)] = GridInvertedSwdO;
  grid_event_fcts[(int)(INVERTED_SW_E)] = GridInvertedSwE;
  grid_event_fcts[(int)(INVERTED_SW_O)] = GridInvertedSwO;
  grid_event_fcts[(int)(ERASED_SW)] = GridErasedSw;
  grid_event_fcts[(int)(ERASED_SWD)] = GridErasedSwd;
}

#endif
