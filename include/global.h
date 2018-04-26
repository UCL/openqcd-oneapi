
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

#define NAME_SIZE 128

#if (NAME_SIZE < 128)
#error : NAME_SIZE must be greater or equal to 128
#endif

#define ALIGN 6

#if defined (LIBRARY)

extern int NPROC0;
extern int NPROC1;
extern int NPROC2;
extern int NPROC3;

extern int L0;
extern int L1;
extern int L2;
extern int L3;

extern int NPROC0_BLK;
extern int NPROC1_BLK;
extern int NPROC2_BLK;
extern int NPROC3_BLK;

extern int NPROC;
extern int VOLUME;
extern int FACE0;
extern int FACE1;
extern int FACE2;
extern int FACE3;
extern int BNDRY;
extern int NSPIN;

extern int cpr[4];
extern int npr[8];

extern int *ipt;
extern int **iup;
extern int **idn;
extern int *map;

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

#ifdef dirac_counters

/* Counters for call to the Dirac operator  */
extern int Dw_dble_counter;
extern int Dwee_dble_counter;
extern int Dwoo_dble_counter;
extern int Dwoe_dble_counter;
extern int Dweo_dble_counter;
extern int Dwhat_dble_counter;
extern int Dw_blk_dble_counter;
extern int Dwee_blk_dble_counter;
extern int Dwoo_blk_dble_counter;
extern int Dwoe_blk_dble_counter;
extern int Dweo_blk_dble_counter;
extern int Dwhat_blk_dble_counter;

extern int Dw_counter;
extern int Dwee_counter;
extern int Dwoo_counter;
extern int Dwoe_counter;
extern int Dweo_counter;
extern int Dwhat_counter;
extern int Dw_blk_counter;
extern int Dwee_blk_counter;
extern int Dwoo_blk_counter;
extern int Dwoe_blk_counter;
extern int Dweo_blk_counter;
extern int Dwhat_blk_counter;

#endif /* dirac_counters */

#ifndef SU3_H
#include "su3.h"
#endif

#endif /* GLOBAL_H */
