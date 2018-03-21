
/*******************************************************************************
 *
 * File flags/events.h
 *
 * Copyright (C) 2009, 2010, 2012, 2016 Martin Luescher, Isabel Campos
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Full-lattice events
 *
 *******************************************************************************/

#define EVENTS_H

#if (defined FLAGS_C)

static void (*event_fcts[(int)(EVENTS) + 1])(void) = {NULL};

static void LatUpdatedU(void)
{
  error(check_bit_state(lat.u.state, SMEARED_STATE_BIT) != 0, 1,
        "LatUpdatedU [events.h]",
        "Trying to update a single precision gauge "
        "field that is not in its smeared state");

  lat.u.tag = next_tag();
}

static void LatUpdatedUd(void)
{
  error(check_bit_state(lat.ud.state, SMEARED_STATE_BIT) != 0, 1,
        "LatUpdatedUd [events.h]",
        "Trying to update a double precision gauge "
        "field that is in its smeared state");

  lat.ud.tag = next_tag();
}

static void LatAssignedUd2u(void) { lat.u = lat.ud; }

static void LatCopiedBndUd(void) { lat.udbuf = lat.ud; }

static void LatSetBstap(void) { lat.bstap = lat.ud; }

static void LatShiftedUd(void)
{
  lat.ud.tag = next_tag();
  lat.udbuf.tag = 0;
  lat.udbuf.state = 0;
}

static void LatComputedFts(void) { lat.fts = lat.ud; }

static void LatErasedSw(void)
{
  lat.sw.state.tag = 0;
  lat.sw.state.state = 0;
  lat.sw.even_flag = 0;
  lat.sw.odd_flag = 0;
}

static void LatErasedSwd(void)
{
  lat.swd.state.tag = 0;
  lat.swd.state.state = 0;
  lat.swd.even_flag = 0;
  lat.swd.odd_flag = 0;
}

static void LatComputedSwd(void)
{
  lat.swd.state = lat.ud;
  lat.swd.even_flag = 0;
  lat.swd.odd_flag = 0;
}

static void LatAssignedSwd2sw(void) { lat.sw = lat.swd; }

static void LatInvertedSwdE(void) { lat.swd.even_flag ^= 0x1; }

static void LatInvertedSwdO(void) { lat.swd.odd_flag ^= 0x1; }

static void LatErasedAw(void)
{
  lat.aw.tag = 0;
  lat.aw.state = 0;
}

static void LatErasedAwhat(void)
{
  lat.awh.tag = 0;
  lat.awh.state = 0;
}

static void LatComputedAw(void) { lat.aw = lat.ud; }

static void LatComputedAwhat(void) { lat.awh = lat.ud; }

static void LatUdFieldSmeared(void)
{
  lat.smeared_tag = lat.ud.tag;
  enable_bit_state(&lat.ud.state, SMEARED_STATE_BIT);
  enable_bit_state(&lat.udbuf.state, SMEARED_STATE_BIT);
}

static void LatUdFieldUnsmeared(void)
{
  disable_bit_state(&lat.ud.state, SMEARED_STATE_BIT);
  disable_bit_state(&lat.udbuf.state, SMEARED_STATE_BIT);
}

static void LatSetUdPhase(void)
{
  enable_bit_state(&lat.ud.state, PHASE_STATE_BIT);
  enable_bit_state(&lat.udbuf.state, PHASE_STATE_BIT);
}

static void LatUnsetUdPhase(void)
{
  disable_bit_state(&lat.ud.state, PHASE_STATE_BIT);
  disable_bit_state(&lat.udbuf.state, PHASE_STATE_BIT);
}

static void set_events(void)
{
  event_fcts[(int)(UPDATED_U)] = LatUpdatedU;
  event_fcts[(int)(UPDATED_UD)] = LatUpdatedUd;
  event_fcts[(int)(ASSIGNED_UD2U)] = LatAssignedUd2u;
  event_fcts[(int)(COPIED_BND_UD)] = LatCopiedBndUd;
  event_fcts[(int)(SET_BSTAP)] = LatSetBstap;
  event_fcts[(int)(SHIFTED_UD)] = LatShiftedUd;
  event_fcts[(int)(COMPUTED_FTS)] = LatComputedFts;
  event_fcts[(int)(ERASED_SW)] = LatErasedSw;
  event_fcts[(int)(ERASED_SWD)] = LatErasedSwd;
  event_fcts[(int)(COMPUTED_SWD)] = LatComputedSwd;
  event_fcts[(int)(ASSIGNED_SWD2SW)] = LatAssignedSwd2sw;
  event_fcts[(int)(INVERTED_SWD_E)] = LatInvertedSwdE;
  event_fcts[(int)(INVERTED_SWD_O)] = LatInvertedSwdO;
  event_fcts[(int)(ERASED_AW)] = LatErasedAw;
  event_fcts[(int)(ERASED_AWHAT)] = LatErasedAwhat;
  event_fcts[(int)(COMPUTED_AW)] = LatComputedAw;
  event_fcts[(int)(COMPUTED_AWHAT)] = LatComputedAwhat;
  event_fcts[(int)(SMEARED_UD)] = LatUdFieldSmeared;
  event_fcts[(int)(UNSMEARED_UD)] = LatUdFieldUnsmeared;
  event_fcts[(int)(SET_UD_PHASE)] = LatSetUdPhase;
  event_fcts[(int)(UNSET_UD_PHASE)] = LatUnsetUdPhase;
}

#endif
