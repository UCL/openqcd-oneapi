
/*******************************************************************************
 *
 * File polyakov_loop.c
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * The externally accessible functions are
 *
 *
 * Notes:
 *
 *
 *******************************************************************************/

#define POLYAKOV_LOOP_C
#define OPENQCD_INTERNAL

#include "global.h"
#include "lattice.h"
#include "mpi.h"
#include "su3fcts.h"
#include "uflds.h"

#define LSPATIAL (L1 * L2 * L3)

double polyakov_loop(void)
{
  su3_dble *pol_loops, *pol_buffers = NULL, *u;
  int it, full_idx, spat_idx, tag, phase_set_q;
  double inv_lspat, local_trace, global_trace;
  MPI_Status stat;

  error(iup[0][0] == 0, 1, "polyakov_loop [polyakov_loop.c]",
        "Geometry arrays are not set");

  pol_loops = amalloc(LSPATIAL * sizeof(*pol_loops), ALIGN);

  error(pol_loops == NULL, 1, "polyakov_loop [polyakov_loop.c]",
        "Unable to allocate memory space for the SU(3) matrices");

  if ((NPROC0 > 0) && (cpr[0] != (NPROC0 - 1))) {
    pol_buffers = amalloc(LSPATIAL * sizeof(*pol_buffers), ALIGN);

    error_root(
        pol_buffers == NULL, 1, "polyakov_loop [polyakov_loop.c]",
        "Unable to allocate memory space for the SU(3) communication buffers");
  }

  u = udfld();

  phase_set_q = query_flags(UD_PHASE_SET);
  if (phase_set_q == 1) {
    unset_ud_phase();
  }

  /* Initialise the Polyakov loops */
  for (spat_idx = 0; spat_idx < LSPATIAL; ++spat_idx) {
    full_idx = ipt[spat_idx];
    if (full_idx < VOLUME / 2) {
      pol_loops[spat_idx] = u[8 * (iup[full_idx][0] - VOLUME / 2) + 1];
      su3xsu3(u + 8 * (iup[full_idx][0] - VOLUME / 2), pol_loops + spat_idx,
              pol_loops + spat_idx);
    } else {
      pol_loops[spat_idx] = u[8 * (full_idx - VOLUME / 2) + 1];
      su3xsu3(u + 8 * (full_idx - VOLUME / 2), pol_loops + spat_idx,
              pol_loops + spat_idx);
    }
  }


  /* Loop and extend their lengths */
  for (it = 2; it < L0; it += 2) {
    for (spat_idx = 0; spat_idx < LSPATIAL; ++spat_idx) {
      full_idx = ipt[spat_idx + it * LSPATIAL];
      if (full_idx < VOLUME / 2) {
        su3xsu3(u + 8 * (iup[full_idx][0] - VOLUME / 2) + 1,
                pol_loops + spat_idx, pol_loops + spat_idx);
        su3xsu3(u + 8 * (iup[full_idx][0] - VOLUME / 2),
                pol_loops + spat_idx, pol_loops + spat_idx);
      } else {
        su3xsu3(u + 8 * (full_idx - VOLUME / 2) + 1, pol_loops + spat_idx,
                pol_loops + spat_idx);
        su3xsu3(u + 8 * (full_idx - VOLUME / 2), pol_loops + spat_idx,
                pol_loops + spat_idx);
      }
    }
  }

  if (phase_set_q == 1) {
    set_ud_phase();
  }

  /* Order communication in the temporal direction */
  if (NPROC0 > 1) {
    tag = mpi_tag();

    if (cpr[0] == (NPROC0 - 1)) {
      MPI_Send(pol_loops, 18 * LSPATIAL, MPI_DOUBLE, npr[0], tag,
               MPI_COMM_WORLD);
    } else if (cpr[0] == 0) {
      MPI_Recv(pol_buffers, 18 * LSPATIAL, MPI_DOUBLE, npr[1], tag,
               MPI_COMM_WORLD, &stat);

      for (spat_idx = 0; spat_idx < LSPATIAL; ++spat_idx) {
        su3xsu3(pol_buffers + spat_idx, pol_loops + spat_idx,
                pol_loops + spat_idx);
      }
    } else {
      MPI_Recv(pol_buffers, 18 * LSPATIAL, MPI_DOUBLE, npr[1], tag,
               MPI_COMM_WORLD, &stat);

      for (spat_idx = 0; spat_idx < LSPATIAL; ++spat_idx) {
        su3xsu3(pol_buffers + spat_idx, pol_loops + spat_idx,
                pol_loops + spat_idx);
      }

      MPI_Send(pol_loops, 18 * LSPATIAL, MPI_DOUBLE, npr[0], tag,
               MPI_COMM_WORLD);
    }
  }

  local_trace = 0.0;
  global_trace = 0.0;

  /* Trace the completed polyakov loops */
  if (cpr[0] == 0) {
    inv_lspat = 1.0 / (3. * LSPATIAL);

    for (spat_idx = 0; spat_idx < LSPATIAL; ++spat_idx) {
      local_trace += (pol_loops[spat_idx].c11.re + pol_loops[spat_idx].c22.re +
                      pol_loops[spat_idx].c33.re) *
                     inv_lspat;
    }
  }

  MPI_Allreduce(&local_trace, &global_trace, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  afree(pol_loops);

  if ((NPROC0 > 0) && (cpr[0] != (NPROC0 - 1))) {
    afree(pol_buffers);
  }


  return global_trace / (NPROC1 * NPROC2 * NPROC3);
}
