
/*******************************************************************************
 *
 * File mdflds.h
 *
 * Copyright (C) 2011, 2013 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef MDFLDS_H
#define MDFLDS_H

#include "su3.h"

typedef struct
{
  int npf;
  su3_alg_dble *mom, *frc;
  spinor_dble **pf;
} openqcd_mdflds__mdflds_t;

/* MDFLDS_C */
extern openqcd_mdflds__mdflds_t *openqcd_mdflds__mdflds(void);
extern void openqcd_mdflds__set_frc2zero(void);
extern void openqcd_mdflds__bnd_mom2zero(void);
extern void openqcd_mdflds__random_mom(void);
extern double openqcd_mdflds__momentum_action(int icom);
extern void openqcd_mdflds__copy_bnd_frc(void);
extern void openqcd_mdflds__add_bnd_frc(void);

#if defined(OPENQCD_INTERNAL)
#define mdflds_t openqcd_mdflds__mdflds_t 

/* MDFLDS_C */
#define mdflds(...) openqcd_mdflds__mdflds(__VA_ARGS__)
#define set_frc2zero(...) openqcd_mdflds__set_frc2zero(__VA_ARGS__)
#define bnd_mom2zero(...) openqcd_mdflds__bnd_mom2zero(__VA_ARGS__)
#define random_mom(...) openqcd_mdflds__random_mom(__VA_ARGS__)
#define momentum_action(...) openqcd_mdflds__momentum_action(__VA_ARGS__)
#define copy_bnd_frc(...) openqcd_mdflds__copy_bnd_frc(__VA_ARGS__)
#define add_bnd_frc(...) openqcd_mdflds__add_bnd_frc(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif
