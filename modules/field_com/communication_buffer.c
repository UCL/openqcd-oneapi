
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
