
/*******************************************************************************
 *
 * File uflds.c
 *
 * Copyright (C) 2006, 2010-2013, 2016 Martin Luescher, Isabel Campos
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Allocation and initialization of the global gauge fields.
 *
 * The externally accessible functions are
 *
 *   su3 *ufld(void)
 *     Returns the base address of the single-precision gauge field. If it
 *     is not already allocated, the field is allocated and initialized to
 *     unity.
 *
 *   su3_dble *udfld(void)
 *     Returns the base address of the double-precision gauge field. If it
 *     is not already allocated, the field is allocated and initialized to
 *     unity. Then the boundary conditions are set according to the data
 *     base by calling set_bc() [bcnds.c].
 *
 *   void apply_ani_ud(void)
 *     Multiply the links by their respective anisotropy factors with respect
 *     to the hopping term. This means that the temporal links are multiplied by
 *     (1.0 / (ani.ut_fermion)) while the spatial links are multiplied by (1.0 /
 *     (gamma_f * ani.ut_fermion)).
 *
 *   void remove_ani_ud(void)
 *     Remove the hopping anisotropy factors from the gauge links by multiplying
 *     by the inverse anisotropy factors.
 *
 *   void random_ud(void)
 *     Initializes the active double-precision link variables to uniformly
 *     distributed random SU(3) matrices. Then the boundary conditions are
 *     set according to the data base by calling set_bc() [bcnds.c].
 *
 *   void set_ud_phase(void)
 *     Multiplies the double-precision link variables U(x,k) by the phase
 *     factor exp{i*theta[k-1]/N[k]}, for all k=1,2,3, where N[mu] is the
 *     size of the (global) lattice in direction mu. The angles theta[0],
 *     theta[1],theta[2] are set by set_bc_parms() [flags/lat_parms.c]. If
 *     periodic boundary conditions are chosen in time, the variables U(x,0)
 *     at global time N[0]-1 are multiplied by -1. The program does nothing
 *     if the phase is already set according to the flags data base.
 *
 *   void unset_ud_phase(void)
 *     Removes the phase of the double-precision link variables previously
 *     set by set_ud_phase(). No action is performed if the phase is not
 *     set according to the flags data base.
 *
 *   void renormalize_ud(void)
 *     Projects the active double-precision link variables back to SU(3).
 *     The static link variables are left untouched. An error occurs if
 *     the phase of the field is set according to the flags data base [see
 *     set_ud_phase() and unset_ud_phase()].
 *
 *   void assign_ud2u(void)
 *     Assigns the double-precision gauge field to the single-precision
 *     gauge field. All link variables in the local field, including the
 *     static ones, are copied.
 *
 *   extern void swap_udfld(su3_dble **new_field)
 *     Swaps the addresses of the global udfld with that of new_field. Used when
 *     cycling between smeared and unsmeared fields.
 *
 *   void copy_bnd_ud(void)
 *     Copies the double-precision link variables from the neighbouring MPI
 *     processes to the exterior boundaries of the local lattice. The field
 *     variables on the spatial links at time NPROC0*L0 are fetched only in
 *     the case of periodic boundary conditions.
 *
 * Notes:
 *
 * The double-precision field can only be allocated after the geometry arrays
 * are set up. All programs in this module act globally and must be called on
 * all MPI processes simultaneously.
 *
 *******************************************************************************/

#define UFLDS_C
#define OPENQCD_INTERNAL

#include "uflds.h"
#include "field_com.h"
#include "global.h"
#include "lattice.h"
#include "mpi.h"
#include "random.h"
#include "su3fcts.h"

#define N0 (NPROC0 * L0)
#define N1 (NPROC1 * L1)
#define N2 (NPROC2 * L2)
#define N3 (NPROC3 * L3)

static const su3 u0 = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f},
                       {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f},
                       {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
static const su3_dble ud0 = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                             {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                             {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
static complex_dble phase[3] ALIGNED16;
static su3 *ub = NULL;
static su3_dble *udb = NULL;

static void alloc_u(void)
{
  size_t n;
  su3 unity, *u, *um;

  error_root(sizeof(su3) != (18 * sizeof(float)), 1, "alloc_u [uflds.c]",
             "The su3 structures are not properly packed");

  n = 4 * VOLUME;
  ub = amalloc(n * sizeof(*ub), ALIGN);
  error(ub == NULL, 1, "alloc_u [uflds.c]",
        "Unable to allocate memory space for the gauge field");

  unity = u0;
  unity.c11.re = 1.0f;
  unity.c22.re = 1.0f;
  unity.c33.re = 1.0f;
  u = ub;
  um = ub + n;

  for (; u < um; u++) {
    (*u) = unity;
  }

  set_flags(UPDATED_U);
}

su3 *ufld(void)
{
  if (ub == NULL) {
    alloc_u();
  }

  return ub;
}

static void alloc_ud(void)
{
  int bc;
  size_t n;
  su3_dble unity, *ud, *um;

  error_root(sizeof(su3_dble) != (18 * sizeof(double)), 1, "alloc_ud [uflds.c]",
             "The su3_dble structures are not properly packed");

  error(iup[0][0] == 0, 1, "alloc_ud [uflds.c]", "Geometry arrays are not set");

  bc = bc_type();
  n = 4 * VOLUME + 7 * (BNDRY / 4);

  if ((cpr[0] == (NPROC0 - 1)) && ((bc == 1) || (bc == 2))) {
    n += 3;
  }

  udb = amalloc(n * sizeof(*udb), ALIGN);
  error(udb == NULL, 1, "alloc_ud [uflds.c]",
        "Unable to allocate memory space for the gauge field");

  unity = ud0;
  unity.c11.re = 1.0;
  unity.c22.re = 1.0;
  unity.c33.re = 1.0;
  ud = udb;
  um = udb + n;

  for (; ud < um; ud++) {
    (*ud) = unity;
  }

  set_flags(UPDATED_UD);
  set_flags(UNSET_UD_PHASE);
  set_bc();
}

su3_dble *udfld(void)
{
  if (udb == NULL) {
    alloc_ud();
  }

  return udb;
}

void apply_ani_ud(void)
{
  su3_dble *u, *um;
  ani_params_t ani;
  double ani_t, ani_s;

  ani = ani_parms();

  if (!ani.has_ani) {
    return;
  }

  ani_t = 1.0 / (ani.ut_fermion);
  ani_s = (ani.nu) / (ani.xi * ani.ut_fermion);

  u = udfld();
  um = u + 4 * VOLUME;
  for (; u < um; u += 8) {
    cm3x3_mulr(&ani_t, (u + 0), (u + 0));
    cm3x3_mulr(&ani_t, (u + 1), (u + 1));
    cm3x3_mulr(&ani_s, (u + 2), (u + 2));
    cm3x3_mulr(&ani_s, (u + 3), (u + 3));
    cm3x3_mulr(&ani_s, (u + 4), (u + 4));
    cm3x3_mulr(&ani_s, (u + 5), (u + 5));
    cm3x3_mulr(&ani_s, (u + 6), (u + 6));
    cm3x3_mulr(&ani_s, (u + 7), (u + 7));
  }
}

void remove_ani_ud(void)
{
  su3_dble *u, *um;
  ani_params_t ani;
  double ani_inv_t, ani_inv_s;

  ani = ani_parms();

  if (!ani.has_ani) {
    return;
  }

  ani_inv_t = ani.ut_fermion;
  ani_inv_s = (ani.xi * ani.ut_fermion) / ani.nu;

  u = udfld();
  um = u + 4 * VOLUME;
  for (; u < um; u += 8) {
    cm3x3_mulr(&ani_inv_t, (u + 0), (u + 0));
    cm3x3_mulr(&ani_inv_t, (u + 1), (u + 1));
    cm3x3_mulr(&ani_inv_s, (u + 2), (u + 2));
    cm3x3_mulr(&ani_inv_s, (u + 3), (u + 3));
    cm3x3_mulr(&ani_inv_s, (u + 4), (u + 4));
    cm3x3_mulr(&ani_inv_s, (u + 5), (u + 5));
    cm3x3_mulr(&ani_inv_s, (u + 6), (u + 6));
    cm3x3_mulr(&ani_inv_s, (u + 7), (u + 7));
  }
}

void random_ud(void)
{
  int bc, ix, t, ifc;
  su3_dble *ud;

  bc = bc_type();
  ud = udfld();

  for (ix = (VOLUME / 2); ix < VOLUME; ix++) {
    t = global_time(ix);

    if (t == 0) {
#ifdef SITERANDOM
      random_su3_dble(ud, ix);
#else
      random_su3_dble(ud);
#endif
      ud += 1;

      if (bc != 0) {
#ifdef SITERANDOM
        random_su3_dble(ud, ix);
#else
        random_su3_dble(ud);
#endif
      }
      ud += 1;

      for (ifc = 2; ifc < 8; ifc++) {
        if (bc != 1) {
#ifdef SITERANDOM
          random_su3_dble(ud, ix);
#else
          random_su3_dble(ud);
#endif
        }
        ud += 1;
      }
    } else if (t == (N0 - 1)) {
      if (bc != 0) {
#ifdef SITERANDOM
        random_su3_dble(ud, ix);
#else
        random_su3_dble(ud);
#endif
      }
      ud += 1;

      for (ifc = 1; ifc < 8; ifc++) {
#ifdef SITERANDOM
        random_su3_dble(ud, ix);
#else
        random_su3_dble(ud);
#endif
        ud += 1;
      }
    } else {
      for (ifc = 0; ifc < 8; ifc++) {
#ifdef SITERANDOM
        random_su3_dble(ud, ix);
#else
        random_su3_dble(ud);
#endif
        ud += 1;
      }
    }
  }

  set_flags(UPDATED_UD);
  set_flags(UNSET_UD_PHASE);
  set_bc();
}

static int set_phase(int pm, double *theta)
{
  int is;
  double p;

  p = theta[0] / (double)(N1);
  phase[0].re = cos(p);
  phase[0].im = sin(p);
  is = (not_equal_d(p, 0.0));

  p = theta[1] / (double)(N2);
  phase[1].re = cos(p);
  phase[1].im = sin(p);
  is |= (not_equal_d(p, 0.0));

  p = theta[2] / (double)(N3);
  phase[2].re = cos(p);
  phase[2].im = sin(p);
  is |= (not_equal_d(p, 0.0));

  if (pm == -1) {
    phase[0].im = -phase[0].im;
    phase[1].im = -phase[1].im;
    phase[2].im = -phase[2].im;
  }

  return is;
}

static void mult_ud_phase(int bc)
{
  int k;
  su3_dble *ud, *um;

  ud = udfld();
  um = ud + 4 * VOLUME;

  /* Multiply ud phase on the links in the bulk */
  for (; ud < um;) {
    ud += 2;

    for (k = 0; k < 3; k++) {
      cm3x3_mulc(phase + k, ud, ud);
      ud += 1;
      cm3x3_mulc(phase + k, ud, ud);
      ud += 1;
    }
  }

  /* Only update the links in the halo for periodic bc's */
  if (bc != 3) {
    return;
  }

  /* Multiply the type 1 boundary links with a phase */
  ud = udfld() + 4 * VOLUME + FACE0 / 2;

  um = ud + FACE1 / 2;
  for (; ud < um; ud++) {
    cm3x3_mulc(phase + 0, ud, ud);
  }

  um = ud + FACE2 / 2;
  for (; ud < um; ud++) {
    cm3x3_mulc(phase + 1, ud, ud);
  }

  um = ud + FACE3 / 2;
  for (; ud < um; ud++) {
    cm3x3_mulc(phase + 2, ud, ud);
  }

  /* Then multiply type 2 boundary links */
  um = ud + 3 * FACE0;
  for (; ud < um;) {
    for (k = 0; k < 3; ++k) {
      cm3x3_mulc(phase + k, ud, ud);
      ud += 1;
    }
  }

  um = ud + 3 * FACE1;
  for (; ud < um;) {
    ud += 1;
    cm3x3_mulc(phase + 1, ud, ud);
    ud += 1;
    cm3x3_mulc(phase + 2, ud, ud);
    ud += 1;
  }

  um = ud + 3 * FACE2;
  for (; ud < um;) {
    ud += 1;
    cm3x3_mulc(phase + 0, ud, ud);
    ud += 1;
    cm3x3_mulc(phase + 2, ud, ud);
    ud += 1;
  }

  um = ud + 3 * FACE3;
  for (; ud < um;) {
    ud += 1;
    cm3x3_mulc(phase + 0, ud, ud);
    ud += 1;
    cm3x3_mulc(phase + 1, ud, ud);
    ud += 1;
  }
}

static void change_sign_links(su3_dble *ud, int *links, int num_links)
{
  int *links_max;
  su3_dble *vd;

  links_max = links + num_links;

  for (; links < links_max; links++) {
    vd = ud + (*links);

    (*vd).c11.re = -(*vd).c11.re;
    (*vd).c11.im = -(*vd).c11.im;
    (*vd).c12.re = -(*vd).c12.re;
    (*vd).c12.im = -(*vd).c12.im;
    (*vd).c13.re = -(*vd).c13.re;
    (*vd).c13.im = -(*vd).c13.im;

    (*vd).c21.re = -(*vd).c21.re;
    (*vd).c21.im = -(*vd).c21.im;
    (*vd).c22.re = -(*vd).c22.re;
    (*vd).c22.im = -(*vd).c22.im;
    (*vd).c23.re = -(*vd).c23.re;
    (*vd).c23.im = -(*vd).c23.im;

    (*vd).c31.re = -(*vd).c31.re;
    (*vd).c31.im = -(*vd).c31.im;
    (*vd).c32.re = -(*vd).c32.re;
    (*vd).c32.im = -(*vd).c32.im;
    (*vd).c33.re = -(*vd).c33.re;
    (*vd).c33.im = -(*vd).c33.im;
  }
}

static void chs_ud0(void)
{
  int num_links, *links;
  su3_dble *ud;

  ud = udfld();

  /* Change links in bulk */
  links = bnd_lks(&num_links);
  change_sign_links(ud, links, num_links);

  /* Change links in the boundary */
  links = bnd_bnd_lks(&num_links);
  change_sign_links(ud, links, num_links);
}

void set_ud_phase(void)
{
  int bc, is;
  su3_dble *ud;
  bc_parms_t bcp;

  if (query_flags(UD_PHASE_SET) == 0) {

    bcp = bc_parms();
    bc = bcp.type;

    if ((bc == 3) && (query_flags(UDBUF_UP2DATE) == 0)) {
      copy_bnd_ud();
    }

    is = set_phase(1, bcp.theta);

    if (is) {
      mult_ud_phase(bc);

      if ((cpr[0] == (NPROC0 - 1)) && ((bc == 1) || (bc == 2))) {
        ud = udfld() + 4 * VOLUME + 7 * (BNDRY / 4);
        cm3x3_mulc(phase, ud, ud);
        ud += 1;
        cm3x3_mulc(phase + 1, ud, ud);
        ud += 1;
        cm3x3_mulc(phase + 2, ud, ud);
      }
    }

    if (bc == 3) {
      chs_ud0();
    }

    /* Halo only updated for periodic boundary conditions */
    if (bc != 3) {
      set_flags(UPDATED_UD);
    }

    set_flags(SET_UD_PHASE);
  }
}

void unset_ud_phase(void)
{
  int bc, is;
  bc_parms_t bcp;

  if (query_flags(UD_PHASE_SET) == 1) {

    set_flags(UNSET_UD_PHASE);

    bcp = bc_parms();
    bc = bcp.type;

    error(
        (bc == 3) && (query_flags(UDBUF_UP2DATE) == 0), 1,
        "unset_ud_phase [uflds.c]",
        "Trying to unset ud phase for a config where the boundary is not up to "
        "date. This means an update has happened on a dirty configuration.");

    is = set_phase(-1, bcp.theta);

    if (is) {
      mult_ud_phase(bc);

      if ((bc == 1) || (bc == 2)) {
        set_bc();
      }
    }

    if (bc == 3) {
      chs_ud0();
    }

    /* Halo only updated for periodic boundary conditions */
    if (bc != 3) {
      set_flags(UPDATED_UD);
    }
  }
}

void renormalize_ud(void)
{
  int bc, ix, t, ifc;
  su3_dble *ud;

  if (query_flags(UD_IS_CLEAN) == 1) {

    bc = bc_type();
    ud = udfld();

    for (ix = (VOLUME / 2); ix < VOLUME; ix++) {
      t = global_time(ix);

      if (t == 0) {
        project_to_su3_dble(ud);
        ud += 1;

        if (bc != 0) {
          project_to_su3_dble(ud);
        }
        ud += 1;

        for (ifc = 2; ifc < 8; ifc++) {
          if (bc != 1) {
            project_to_su3_dble(ud);
          }
          ud += 1;
        }
      } else if (t == (N0 - 1)) {
        if (bc != 0) {
          project_to_su3_dble(ud);
        }
        ud += 1;

        for (ifc = 1; ifc < 8; ifc++) {
          project_to_su3_dble(ud);
          ud += 1;
        }
      } else {
        for (ifc = 0; ifc < 8; ifc++) {
          project_to_su3_dble(ud);
          ud += 1;
        }
      }
    }

    set_flags(UPDATED_UD);
  } else {
    error(1, 1, "renormalize_ud [udflds.c]",
          "Attempt to renormalize dirty "
          "(smeared or phase transformed) "
          "link variables");
  }
}

void assign_ud2u(void)
{
  su3 *u, *um;
  su3_dble *ud;

  u = ufld();
  um = u + 4 * VOLUME;
  ud = udfld();

  for (; u < um; u++) {
    (*u).c11.re = (float)((*ud).c11.re);
    (*u).c11.im = (float)((*ud).c11.im);
    (*u).c12.re = (float)((*ud).c12.re);
    (*u).c12.im = (float)((*ud).c12.im);
    (*u).c13.re = (float)((*ud).c13.re);
    (*u).c13.im = (float)((*ud).c13.im);

    (*u).c21.re = (float)((*ud).c21.re);
    (*u).c21.im = (float)((*ud).c21.im);
    (*u).c22.re = (float)((*ud).c22.re);
    (*u).c22.im = (float)((*ud).c22.im);
    (*u).c23.re = (float)((*ud).c23.re);
    (*u).c23.im = (float)((*ud).c23.im);

    (*u).c31.re = (float)((*ud).c31.re);
    (*u).c31.im = (float)((*ud).c31.im);
    (*u).c32.re = (float)((*ud).c32.re);
    (*u).c32.im = (float)((*ud).c32.im);
    (*u).c33.re = (float)((*ud).c33.re);
    (*u).c33.im = (float)((*ud).c33.im);

    ud += 1;
  }

  set_flags(ASSIGNED_UD2U);
}

void swap_udfld(su3_dble **new_field)
{
  su3_dble *tmp = udb;
  udb = (*new_field);
  (*new_field) = tmp;
}

void copy_bnd_ud(void)
{
  copy_boundary_su3_field(udfld());
  set_flags(COPIED_BND_UD);
}
