
/*******************************************************************************
 *
 * File global.c
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#define GLOBAL_C

#include "global.h"

#if defined (LIBRARY)

int NPROC0;
int NPROC1;
int NPROC2;
int NPROC3;

int L0;
int L1;
int L2;
int L3;

int NPROC0_BLK;
int NPROC1_BLK;
int NPROC2_BLK;
int NPROC3_BLK;

int NPROC;
int VOLUME;
int FACE0;
int FACE1;
int FACE2;
int FACE3;
int BNDRY;
int NSPIN;

int cpr[4];
int npr[8];

int *ipt;
int **iup;
int **idn;
int *map;

#else

int cpr[4];
int npr[8];

int ipt[VOLUME];
int iup[VOLUME][4];
int idn[VOLUME][4];
int map[BNDRY + NPROC % 2];

#endif /* defined LIBRARY */

#ifdef dirac_counters

/* Counters for call to the Dirac operator  */
int Dw_dble_counter;
int Dwee_dble_counter;
int Dwoo_dble_counter;
int Dwoe_dble_counter;
int Dweo_dble_counter;
int Dwhat_dble_counter;
int Dw_blk_dble_counter;
int Dwee_blk_dble_counter;
int Dwoo_blk_dble_counter;
int Dwoe_blk_dble_counter;
int Dweo_blk_dble_counter;
int Dwhat_blk_dble_counter;

int Dw_counter;
int Dwee_counter;
int Dwoo_counter;
int Dwoe_counter;
int Dweo_counter;
int Dwhat_counter;
int Dw_blk_counter;
int Dwee_blk_counter;
int Dwoo_blk_counter;
int Dwoe_blk_counter;
int Dweo_blk_counter;
int Dwhat_blk_counter;

#endif /* dirac_counters */
