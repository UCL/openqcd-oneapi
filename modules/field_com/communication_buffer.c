
/*******************************************************************************
 *
 * File communication_buffer.c
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Storage buffer for the communication routines
 *
 * The externally accessible functions are
 *
 *   double *communication_buffer(void)
 *     Returns the buffer reserved for the communication routines. It is given
 *     in terms of a pointer to double and can thus be used to communicate both
 *     su3_dble and su3_alg_dble through a simple cast.
 *
 * Notes:
 *
 * The routine does communication and must therefore be called simultaneously on
 * all MPI processes.
 *
 *******************************************************************************/

#define COMMUNICATION_BUFFER_C

#include "field_com.h"
#include "global.h"
#include "lattice.h"
#include "mpi.h"

#include "com_macros.h"

static double *send_receive_buffer = NULL;

static void allocate_send_recieve_buffers(void)
{
  int mu, nuk, n;
  uidx_t *idx;

  idx = uidx();
  n = 0;

  /* Find the largest communication size */
  for (mu = 0; mu < 4; mu++) {
    nuk = idx[mu].nuk;

    if (nuk > n) {
      n = nuk;
    }
  }

  send_receive_buffer = (double *)amalloc(n * sizeof(su3_dble), ALIGN);

  error(send_receive_buffer == NULL, 1,
        "allocate_send_recieve_buffers [communication_buffer.c]",
        "Unable to allocate send recieve buffer");
}

double *communication_buffer(void)
{
  if (send_receive_buffer == NULL) {
    allocate_send_recieve_buffers();
  }

  return send_receive_buffer;
}
