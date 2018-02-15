
/*******************************************************************************
*
* File flags/queries.h
*
* Copyright (C) 2009-2012, 2016 Martin Luescher, Isabel Campos
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Query descriptions
*
*******************************************************************************/

#define QUERIES_H

#if (defined FLAGS_C)

static int check_state(cfg_state_t const *state1, cfg_state_t const *state2)
{
  return ((*state1).tag == (*state2).tag) &&
         ((*state1).state == (*state2).state);
}

static int (*query_fcts[(int)(QUERIES) + 1])(void) = {NULL};

static int QueryUMatchUd(void) { return check_state(&lat.u, &lat.ud); }
static int QueryUdbufUp2date(void)
{
  return ((lat.ud.tag > 0) && check_state(&lat.udbuf, &lat.ud));
}

static int QueryBstapUp2date(void)
{
  return ((lat.ud.tag > 0) && check_state(&lat.bstap, &lat.ud));
}

static int QueryFtsUp2date(void)
{
  return ((lat.ud.tag > 0) && check_state(&lat.fts, &lat.ud));
}

static int QuerySwUp2date(void)
{
  return ((lat.u.tag > 0) && check_state(&lat.sw.state, &lat.u));
}

static int QuerySwEInverted(void) { return (lat.sw.even_flag == 1); }

static int QuerySwOInverted(void) { return (lat.sw.odd_flag == 1); }

static int QuerySwdUp2date(void)
{
  return ((lat.ud.tag > 0) && check_state(&lat.swd.state, &lat.ud));
}

static int QuerySwdEInverted(void) { return (lat.swd.even_flag == 1); }

static int QuerySwdOInverted(void) { return (lat.swd.odd_flag == 1); }

static int QueryAwUp2date(void)
{
  return ((lat.ud.tag > 0) && check_state(&lat.aw, &lat.ud));
}

static int QueryAwhatUp2date(void)
{
  return ((lat.ud.tag > 0) && check_state(&lat.awh, &lat.ud));
}

static int QueryUdIsClean(void)
{
  return lat.ud.state == 0;
}

static int QueryUdIsSmeared(void)
{
  return check_bit_state(lat.ud.state, SMEARED_STATE_BIT) != 0;
}

static int QueryUdSmearingUp2date(void)
{
  return (lat.smeared_tag == lat.ud.tag);
}

static int QueryUdPhaseSet(void)
{
  return check_bit_state(lat.ud.state, PHASE_STATE_BIT) != 0;
}

static void set_queries(void)
{
  query_fcts[(int)(U_MATCH_UD)] = QueryUMatchUd;
  query_fcts[(int)(UDBUF_UP2DATE)] = QueryUdbufUp2date;
  query_fcts[(int)(BSTAP_UP2DATE)] = QueryBstapUp2date;
  query_fcts[(int)(FTS_UP2DATE)] = QueryFtsUp2date;
  query_fcts[(int)(SW_UP2DATE)] = QuerySwUp2date;
  query_fcts[(int)(SW_E_INVERTED)] = QuerySwEInverted;
  query_fcts[(int)(SW_O_INVERTED)] = QuerySwOInverted;
  query_fcts[(int)(SWD_UP2DATE)] = QuerySwdUp2date;
  query_fcts[(int)(SWD_E_INVERTED)] = QuerySwdEInverted;
  query_fcts[(int)(SWD_O_INVERTED)] = QuerySwdOInverted;
  query_fcts[(int)(AW_UP2DATE)] = QueryAwUp2date;
  query_fcts[(int)(AWHAT_UP2DATE)] = QueryAwhatUp2date;
  query_fcts[(int)(UD_IS_CLEAN)] = QueryUdIsClean;
  query_fcts[(int)(UD_IS_SMEARED)] = QueryUdIsSmeared;
  query_fcts[(int)(SMEARED_UD_UP2DATE)] = QueryUdSmearingUp2date;
  query_fcts[(int)(UD_PHASE_SET)] = QueryUdPhaseSet;
}

#endif
