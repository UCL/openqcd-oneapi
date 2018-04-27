
/*******************************************************************************
 *
 * File global.h
 *
 * Copyright (C) 2009, 2011, 2013 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Global parameters and arrays
 *
 *******************************************************************************/

#ifndef GLOBAL_H
#define GLOBAL_H

#define openqcd__NAME_SIZE 128

#if (openqcd__NAME_SIZE < 128)
#error : NAME_SIZE must be greater or equal to 128
#endif

#define openqcd__ALIGN 6

#if defined (LIBRARY)

extern int openqcd__NPROC0;
extern int openqcd__NPROC1;
extern int openqcd__NPROC2;
extern int openqcd__NPROC3;

extern int openqcd__L0;
extern int openqcd__L1;
extern int openqcd__L2;
extern int openqcd__L3;

extern int openqcd__NPROC0_BLK;
extern int openqcd__NPROC1_BLK;
extern int openqcd__NPROC2_BLK;
extern int openqcd__NPROC3_BLK;

extern int openqcd__NPROC;
extern int openqcd__VOLUME;
extern int openqcd__FACE0;
extern int openqcd__FACE1;
extern int openqcd__FACE2;
extern int openqcd__FACE3;
extern int openqcd__BNDRY;
extern int openqcd__NSPIN;

extern int openqcd__cpr[4];
extern int openqcd__npr[8];

extern int *openqcd__ipt;
extern int **openqcd__iup;
extern int **openqcd__idn;
extern int *openqcd__map;

extern void openqcd__set_lattice_sizes(int nproc[4], int lat_sizes[4], int block_sizes[4]);

#if defined(OPENQCD_INTERNAL)

#define NPROC0 openqcd__NPROC0
#define NPROC1 openqcd__NPROC1
#define NPROC2 openqcd__NPROC2
#define NPROC3 openqcd__NPROC3

#define L0 openqcd__L0
#define L1 openqcd__L1
#define L2 openqcd__L2
#define L3 openqcd__L3

#define NPROC0_BLK openqcd__NPROC0_BLK
#define NPROC1_BLK openqcd__NPROC1_BLK
#define NPROC2_BLK openqcd__NPROC2_BLK
#define NPROC3_BLK openqcd__NPROC3_BLK

#define NPROC openqcd__NPROC
#define VOLUME openqcd__VOLUME
#define FACE0 openqcd__FACE0
#define FACE1 openqcd__FACE1
#define FACE2 openqcd__FACE2
#define FACE3 openqcd__FACE3
#define BNDRY openqcd__BNDRY
#define NSPIN openqcd__NSPIN

#define cpr openqcd__cpr
#define npr openqcd__npr

#define ipt openqcd__ipt
#define iup openqcd__iup
#define idn openqcd__idn
#define map openqcd__map

#define set_lattice_sizes(...) openqcd__set_lattice_sizes(__VA_ARGS__)

#endif /* defined OPENQCD_INTERNAL */

#else /* not defined LIBRARY */

/*
THESE QUANTITIES ARE NOW DEFINED IN A TEXT FILE READ BY THE MAKEFILE

#define NPROC0 4
#define NPROC1 8
#define NPROC2 8
#define NPROC3 8

#define L0 8
#define L1 4
#define L2 4
#define L3 4

#define NPROC0_BLK 2
#define NPROC1_BLK 2
#define NPROC2_BLK 2
#define NPROC3_BLK 2
*/

/* Global geometry checks */

#if ((NPROC0 < 1) || (NPROC1 < 1) || (NPROC2 < 1) || (NPROC3 < 1) ||           \
     ((NPROC0 > 1) && ((NPROC0 % 2) != 0)) ||                                  \
     ((NPROC1 > 1) && ((NPROC1 % 2) != 0)) ||                                  \
     ((NPROC2 > 1) && ((NPROC2 % 2) != 0)) ||                                  \
     ((NPROC3 > 1) && ((NPROC3 % 2) != 0)))
#error : The number of processes in each direction must be 1 or a multiple of 2
#endif

#if ((L0 < 4) || (L1 < 4) || (L2 < 4) || (L3 < 4) || ((L0 % 2) != 0) ||        \
     ((L1 % 2) != 0) || ((L2 % 2) != 0) || ((L3 % 2) != 0))
#error : The local lattice sizes must be even and not smaller than 4
#endif

#if ((NPROC0_BLK < 1) || (NBROC0_BLK > NPROC0) ||                              \
     ((NPROC0 % NPROC0_BLK) != 0) || (NPROC1_BLK < 1) ||                       \
     (NBROC1_BLK > NPROC1) || ((NPROC1 % NPROC1_BLK) != 0) ||                  \
     (NPROC2_BLK < 1) || (NBROC2_BLK > NPROC2) ||                              \
     ((NPROC2 % NPROC2_BLK) != 0) || (NPROC3_BLK < 1) ||                       \
     (NBROC3_BLK > NPROC3) || ((NPROC3 % NPROC3_BLK) != 0))
#error : Improper processor block sizes NPROC0_BLK,..,NPROC3_BLK
#endif

#define NPROC (NPROC0 * NPROC1 * NPROC2 * NPROC3)
#define VOLUME (L0 * L1 * L2 * L3)
#define FACE0 ((1 - (NPROC0 % 2)) * L1 * L2 * L3)
#define FACE1 ((1 - (NPROC1 % 2)) * L2 * L3 * L0)
#define FACE2 ((1 - (NPROC2 % 2)) * L3 * L0 * L1)
#define FACE3 ((1 - (NPROC3 % 2)) * L0 * L1 * L2)
#define BNDRY (2 * (FACE0 + FACE1 + FACE2 + FACE3))
#define NSPIN (VOLUME + (BNDRY / 2))

extern int cpr[4];
extern int npr[8];

extern int ipt[VOLUME];
extern int iup[VOLUME][4];
extern int idn[VOLUME][4];
extern int map[BNDRY + NPROC % 2];

#endif /* LIBRARY */


#if defined(OPENQCD_INTERNAL)
#define NAME_SIZE openqcd__NAME_SIZE
#define ALIGN openqcd__ALIGN
#endif /* defined OPENQCD_INTERNAL */

#ifdef dirac_counters

/* Counters for call to the Dirac operator  */
extern int openqcd__Dw_dble_counter;
extern int openqcd__Dwee_dble_counter;
extern int openqcd__Dwoo_dble_counter;
extern int openqcd__Dwoe_dble_counter;
extern int openqcd__Dweo_dble_counter;
extern int openqcd__Dwhat_dble_counter;
extern int openqcd__Dw_blk_dble_counter;
extern int openqcd__Dwee_blk_dble_counter;
extern int openqcd__Dwoo_blk_dble_counter;
extern int openqcd__Dwoe_blk_dble_counter;
extern int openqcd__Dweo_blk_dble_counter;
extern int openqcd__Dwhat_blk_dble_counter;

extern int openqcd__Dw_counter;
extern int openqcd__Dwee_counter;
extern int openqcd__Dwoo_counter;
extern int openqcd__Dwoe_counter;
extern int openqcd__Dweo_counter;
extern int openqcd__Dwhat_counter;
extern int openqcd__Dw_blk_counter;
extern int openqcd__Dwee_blk_counter;
extern int openqcd__Dwoo_blk_counter;
extern int openqcd__Dwoe_blk_counter;
extern int openqcd__Dweo_blk_counter;
extern int openqcd__Dwhat_blk_counter;


#if defined(OPENQCD_INTERNAL)
#define Dw_dble_counter openqcd__Dw_dble_counter
#define Dwee_dble_counter openqcd__Dwee_dble_counter
#define Dwoo_dble_counter openqcd__Dwoo_dble_counter
#define Dwoe_dble_counter openqcd__Dwoe_dble_counter
#define Dweo_dble_counter openqcd__Dweo_dble_counter
#define Dwhat_dble_counter openqcd__Dwhat_dble_counter
#define Dw_blk_dble_counter openqcd__Dw_blk_dble_counter
#define Dwee_blk_dble_counter openqcd__Dwee_blk_dble_counter
#define Dwoo_blk_dble_counter openqcd__Dwoo_blk_dble_counter
#define Dwoe_blk_dble_counter openqcd__Dwoe_blk_dble_counter
#define Dweo_blk_dble_counter openqcd__Dweo_blk_dble_counter
#define Dwhat_blk_dble_counter openqcd__Dwhat_blk_dble_counter

#define Dw_counter openqcd__Dw_counter
#define Dwee_counter openqcd__Dwee_counter
#define Dwoo_counter openqcd__Dwoo_counter
#define Dwoe_counter openqcd__Dwoe_counter
#define Dweo_counter openqcd__Dweo_counter
#define Dwhat_counter openqcd__Dwhat_counter
#define Dw_blk_counter openqcd__Dw_blk_counter
#define Dwee_blk_counter openqcd__Dwee_blk_counter
#define Dwoo_blk_counter openqcd__Dwoo_blk_counter
#define Dwoe_blk_counter openqcd__Dwoe_blk_counter
#define Dweo_blk_counter openqcd__Dweo_blk_counter
#define Dwhat_blk_counter openqcd__Dwhat_blk_counter
#endif /* defined OPENQCD_INTERNAL */

#endif /* dirac_counters */

#ifndef SU3_H
#include "su3.h"
#endif

#endif /* GLOBAL_H */
