
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
#define OPENQCD_INTERNAL

#include "global.h"
#include "utils.h"

#if defined(LIBRARY)

int NPROC0 = 0;
int NPROC1 = 0;
int NPROC2 = 0;
int NPROC3 = 0;

int L0 = 0;
int L1 = 0;
int L2 = 0;
int L3 = 0;

int NPROC0_BLK = 0;
int NPROC1_BLK = 0;
int NPROC2_BLK = 0;
int NPROC3_BLK = 0;

int NPROC = 0;
int VOLUME = 0;
int FACE0 = 0;
int FACE1 = 0;
int FACE2 = 0;
int FACE3 = 0;
int BNDRY = 0;
int NSPIN = 0;

int cpr[4];
int npr[8];

int *ipt = NULL;
int **iup = NULL;
int **idn = NULL;
int *map = NULL;

static int already_init = 0;

static void check_lattice_sizes(void)
{
  error(
      ((NPROC0 < 1) || (NPROC1 < 1) || (NPROC2 < 1) || (NPROC3 < 1) ||
       ((NPROC0 > 1) && ((NPROC0 % 2) != 0)) ||
       ((NPROC1 > 1) && ((NPROC1 % 2) != 0)) ||
       ((NPROC2 > 1) && ((NPROC2 % 2) != 0)) ||
       ((NPROC3 > 1) && ((NPROC3 % 2) != 0))),
      1, "check_lattice_sizes [global.c]",
      "The number of processes in each direction must be 1 or a multiple of 2");

  error(((L0 < 4) || (L1 < 4) || (L2 < 4) || (L3 < 4) || ((L0 % 2) != 0) ||
         ((L1 % 2) != 0) || ((L2 % 2) != 0) || ((L3 % 2) != 0)),
        1, "check_lattice_sizes [global.c]",
        "The local lattice sizes must be even and not smaller than 4");

  error(((NPROC0_BLK < 1) || (NPROC0_BLK > NPROC0) ||
         ((NPROC0 % NPROC0_BLK) != 0) || (NPROC1_BLK < 1) ||
         (NPROC1_BLK > NPROC1) || ((NPROC1 % NPROC1_BLK) != 0) ||
         (NPROC2_BLK < 1) || (NPROC2_BLK > NPROC2) ||
         ((NPROC2 % NPROC2_BLK) != 0) || (NPROC3_BLK < 1) ||
         (NPROC3_BLK > NPROC3) || ((NPROC3 % NPROC3_BLK) != 0)),
        1, "check_lattice_sizes [global.c]",
        "Improper processor block sizes NPROC0_BLK,..,NPROC3_BLK");
}

void set_lattice_sizes(int nproc[4], int lat_sizes[4], int block_sizes[4])
{
  int i;

  error(already_init != 0, 1, "set_lattice_sizes [global.c]",
        "The lattice size has already been initialised");

  NPROC0 = nproc[0];
  NPROC1 = nproc[1];
  NPROC2 = nproc[2];
  NPROC3 = nproc[3];

  L0 = lat_sizes[0];
  L1 = lat_sizes[1];
  L2 = lat_sizes[2];
  L3 = lat_sizes[3];

  NPROC0_BLK = block_sizes[0];
  NPROC1_BLK = block_sizes[1];
  NPROC2_BLK = block_sizes[2];
  NPROC3_BLK = block_sizes[3];

  check_lattice_sizes();

  NPROC = (NPROC0 * NPROC1 * NPROC2 * NPROC3);
  VOLUME = (L0 * L1 * L2 * L3);
  FACE0 = ((1 - (NPROC0 % 2)) * L1 * L2 * L3);
  FACE1 = ((1 - (NPROC1 % 2)) * L2 * L3 * L0);
  FACE2 = ((1 - (NPROC2 % 2)) * L3 * L0 * L1);
  FACE3 = ((1 - (NPROC3 % 2)) * L0 * L1 * L2);
  BNDRY = (2 * (FACE0 + FACE1 + FACE2 + FACE3));
  NSPIN = (VOLUME + (BNDRY / 2));

  ipt = malloc(VOLUME * sizeof(*ipt));
  iup = malloc(VOLUME * sizeof(*iup));
  idn = malloc(VOLUME * sizeof(*idn));
  map = malloc((BNDRY + NPROC % 2) * sizeof(*map));

  iup[0] = malloc(4 * VOLUME * sizeof(**iup));
  idn[0] = malloc(4 * VOLUME * sizeof(**idn));

  for (i = 1; i < VOLUME; ++i) {
    iup[i] = iup[i - 1] + 4;
    idn[i] = idn[i - 1] + 4;
  }

  already_init = 1;
}

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
