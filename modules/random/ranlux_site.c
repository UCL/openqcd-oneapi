
/*******************************************************************************
*
* File ranlux_site.c
*
* Based on ranlux.c
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*   void start_ranlux(int level,int seed)
*     Initializes the random number generators ranlxs and ranlxd on all
*     sites in different ways. The luxury level should be 0 (recommended)
*     or 1 (exceptional) and the seed can be any positive integer less than
*     or equal to INT_MAX - n_global_sites. An error occurs if the seed is
*     not in this range.
*
*   void ranlxs_site(float r[], int n, int x)
*     Computes the next n single-precision random numbers and for site x
*     assigns them to the elements r[0],...,r[n-1] of the array r[]
*     NOTE: This funcion overwrites the state of the standard generator.
*           You should not use both ranlxs() and ranlxs_site() in the same run.
*
*   void ranlxd_site(float r[], int n, int x)
*     Computes the next n double-precision random numbers and for site x
*     assigns them to the elements r[0],...,r[n-1] of the array r[]
*     NOTE: This funcion overwrites the state of the standard generator.
*           You should not use both ranlxd() and ranlxd_site() in the same run.
*
*
*******************************************************************************/

#define RANLUX_C

#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include "mpi.h"
#include "utils.h"
#include "lattice.h"
#include "random.h"
#include "global.h"

static int **rlxs_state = NULL, **rlxd_state;
static stdint_t **state;

static int alloc_state(int x)
{
  int nlxs, nlxd, n;

  nlxs = rlxs_size();
  nlxd = rlxd_size();
  n = nlxs + nlxd;

  if (rlxs_state == NULL) {
    rlxs_state = malloc(VOLUME * sizeof(int *));
    rlxd_state = malloc(VOLUME * sizeof(int *));
    state = malloc(VOLUME * sizeof(stdint_t *));
    int x;
    for (x = 0; x < VOLUME; x++)
      rlxs_state[x] = NULL;
    error((rlxs_state == NULL) || (rlxd_state == NULL) || (state == NULL), 1,
          "alloc_state [ranlux.c]", "Unable to allocate arrays");
  }
  if (rlxs_state[x] == NULL) {
    rlxs_state[x] = malloc(n * sizeof(int));
    rlxd_state[x] = rlxs_state[x] + nlxs;
    state[x] = malloc(n * sizeof(stdint_t));
    error((rlxs_state[x] == NULL) || (state[x] == NULL), 1,
          "alloc_state [ranlux.c]", "Unable to allocate state arrays");
  }

  return n;
}

/* This writes the loca state into the ranlux variables on each call
 * and saves the new state after the call
 */
void ranlxs_site(float r[], int n, int x)
{
  int *global_state = malloc(rlxs_size() * sizeof(int));
  rlxs_get(global_state);
  rlxs_reset(rlxs_state[x]);
  ranlxs(r, n);
  rlxs_get(rlxs_state[x]);
  rlxs_reset(global_state);
  free(global_state);
}
void ranlxd_site(double r[], int n, int x)
{
  int *global_state = malloc(rlxd_size() * sizeof(int));
  rlxd_get(global_state);
  rlxd_reset(rlxd_state[x]);
  ranlxd(r, n);
  rlxd_get(rlxd_state[x]);
  rlxd_reset(global_state);
  free(global_state);
}

void start_ranlux_site(int level, int seed)
{
  int my_rank, max_seed, loc_seed;
  int iprms[2];

  if (NPROC > 1) {
    iprms[0] = level;
    iprms[1] = seed;

    MPI_Bcast(iprms, 2, MPI_INT, 0, MPI_COMM_WORLD);

    error((iprms[0] != level) || (iprms[1] != seed), 1,
          "start_ranlux [ranlux.c]", "Input parameters are not global");
  }

  max_seed = INT_MAX - NPROC * VOLUME;

  error_root((level < 0) || (level > 1) || (seed < 1) || (seed > max_seed), 1,
             "start_ranlux [ranlux.c]", "Parameters are out of range");

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  /* Loop over lattice sites and seed random number generators at local sites
   */
  int x[4];
  for (x[0] = 0; x[0] < NPROC0 * L0; x[0]++)
    for (x[1] = 0; x[1] < NPROC1 * L1; x[1]++)
      for (x[2] = 0; x[2] < NPROC2 * L2; x[2]++)
        for (x[3] = 0; x[3] < NPROC3 * L3; x[3]++) {
          int site_rank, site_index, global_index;
          ipt_global(x, &site_rank, &site_index);
          if (site_rank == my_rank) {
            alloc_state(site_index);

            global_index = ((x[0] * NPROC0 * L0 + x[1]) * NPROC1 * L1 + x[2]) *
                               NPROC2 * L2 +
                           x[3];
            loc_seed = seed + global_index;

            rlxs_init(level, loc_seed);
            rlxs_get(rlxs_state[site_index]);
            rlxd_init(level + 1, loc_seed);
            rlxd_get(rlxd_state[site_index]);
          }
        }
}
