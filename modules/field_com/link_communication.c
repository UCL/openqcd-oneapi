#define LINK_COMMUNICATION_C

#include "field_com.h"
#include "flags.h"
#include "global.h"
#include "lattice.h"
#include "mpi.h"
#include "su3fcts.h"
#include <string.h>

#include "com_macros.h"

static int bc, np;
static uidx_t *idx;
static double *send_receive_buffer = NULL;

static const int positive_send_dir = 1;
static const int negative_send_dir = 0;

static void init_buffer(void)
{
  bc = bc_type();
  np = (cpr[0] + cpr[1] + cpr[2] + cpr[3]) & 0x1;
  idx = uidx();

  send_receive_buffer = communication_buffer();
}

/* Macro definitions of type two commiunication routines */
#define type_two_communication_routines(type)                                  \
                                                                               \
  static void _send_link_type_two_##type(int mu, int dir)                      \
  {                                                                            \
    int nuk, nbf;                                                              \
    int tag, saddr, raddr;                                                     \
    MPI_Status stat;                                                           \
                                                                               \
    nuk = idx[mu].nuk;                                                         \
                                                                               \
    if (nuk > 0) {                                                             \
      tag = mpi_tag();                                                         \
      saddr = npr[2 * mu + (dir & 0x1)];                                       \
      raddr = npr[2 * mu + ((dir + 1) & 0x1)];                                 \
      nbf = sizeof(type) * nuk / sizeof(double);                               \
                                                                               \
      if (np == 0) {                                                           \
        if ((mu > 0) || (bc == 3) ||                                           \
            ((dir == positive_send_dir) && (cpr[0] < (NPROC0 - 1))) ||         \
            ((dir == negative_send_dir) && (cpr[0] > 0)))                      \
          MPI_Send(_link_sbuf_##type, nbf, MPI_DOUBLE, saddr, tag,             \
                   MPI_COMM_WORLD);                                            \
                                                                               \
        if ((mu > 0) || (bc == 3) ||                                           \
            ((dir == negative_send_dir) && (cpr[0] < (NPROC0 - 1))) ||         \
            ((dir == positive_send_dir) && (cpr[0] > 0)))                      \
          MPI_Recv(_link_rbuf_##type, nbf, MPI_DOUBLE, raddr, tag,             \
                   MPI_COMM_WORLD, &stat);                                     \
      } else {                                                                 \
        if ((mu > 0) || (bc == 3) ||                                           \
            ((dir == negative_send_dir) && (cpr[0] < (NPROC0 - 1))) ||         \
            ((dir == positive_send_dir) && (cpr[0] > 0)))                      \
          MPI_Recv(_link_rbuf_##type, nbf, MPI_DOUBLE, raddr, tag,             \
                   MPI_COMM_WORLD, &stat);                                     \
                                                                               \
        if ((mu > 0) || (bc == 3) ||                                           \
            ((dir == positive_send_dir) && (cpr[0] < (NPROC0 - 1))) ||         \
            ((dir == negative_send_dir) && (cpr[0] > 0)))                      \
          MPI_Send(_link_sbuf_##type, nbf, MPI_DOUBLE, saddr, tag,             \
                   MPI_COMM_WORLD);                                            \
      }                                                                        \
                                                                               \
      if (dir == positive_send_dir)                                            \
        _link_sbuf_##type += nuk;                                              \
      else                                                                     \
        _link_rbuf_##type += nuk;                                              \
    }                                                                          \
  }                                                                            \
                                                                               \
  static void _pack_link_type_two_##type(type const *field, int mu)            \
  {                                                                            \
    int nuk, *iu, *ium;                                                        \
    type *u;                                                                   \
                                                                               \
    nuk = idx[mu].nuk;                                                         \
                                                                               \
    if ((nuk > 0) && ((mu > 0) || (cpr[0] > 0) || (bc == 3))) {                \
      u = _link_sbuf_##type;                                                   \
      iu = idx[mu].iuk;                                                        \
      ium = iu + nuk;                                                          \
                                                                               \
      for (; iu < ium; iu++, u++) {                                            \
        _copy_function_##type(u, field + (*iu));                               \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  static void _unpack_add_link_type_two_##type(type *field, int mu)            \
  {                                                                            \
    int nuk, *iu, *ium;                                                        \
    type const *u;                                                             \
                                                                               \
    nuk = idx[mu].nuk;                                                         \
                                                                               \
    if ((nuk > 0) && ((mu > 0) || (cpr[0] > 0) || (bc == 3))) {                \
      u = _link_rbuf_##type;                                                   \
      iu = idx[mu].iuk;                                                        \
      ium = iu + nuk;                                                          \
                                                                               \
      for (; iu < ium; iu++, u++) {                                            \
        _add_function_##type(field + (*iu), u);                                \
      }                                                                        \
    }                                                                          \
  }

/* Define the main communication routines */
/* clang-format off */
#define communication_routines(type)                                           \
                                                                               \
  communication_buffers(type);                                                 \
  type_one_communication_routines(type)                                        \
  type_two_communication_routines(type)                                        \
                                                                               \
  static void _copy_boundary_link_field_##type(type *field)                    \
  {                                                                            \
    int mu;                                                                    \
                                                                               \
    if (NPROC > 1) {                                                           \
      if (send_receive_buffer == NULL)                                         \
        init_buffer();                                                         \
                                                                               \
      _link_sbuf_##type = (type *)send_receive_buffer;                         \
      _link_rbuf_##type = field + 4 * VOLUME;                                  \
                                                                               \
      for (mu = 0; mu < 4; mu++) {                                             \
        _pack_link_type_one_##type(field, mu);                                 \
        _send_link_type_one_##type(mu, negative_send_dir);                     \
      }                                                                        \
                                                                               \
      for (mu = 0; mu < 4; mu++) {                                             \
        _pack_link_type_two_##type(field, mu);                                 \
        _send_link_type_two_##type(mu, negative_send_dir);                     \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  static void _add_boundary_link_field_##type(type *field)                     \
  {                                                                            \
    int mu;                                                                    \
                                                                               \
    if (NPROC > 1) {                                                           \
      if (send_receive_buffer == NULL)                                         \
        init_buffer();                                                         \
                                                                               \
      _link_rbuf_##type = (type *)send_receive_buffer;                         \
      _link_sbuf_##type = field + (4 * VOLUME) + (BNDRY / 4);                  \
                                                                               \
      for (mu = 0; mu < 4; mu++) {                                             \
        _send_link_type_two_##type(mu, positive_send_dir);                     \
        _unpack_add_link_type_two_##type(field, mu);                           \
      }                                                                        \
                                                                               \
      _link_sbuf_##type = field + (4 * VOLUME);                                \
                                                                               \
      for (mu = 0; mu < 4; mu++) {                                             \
        _send_link_type_one_##type(mu, positive_send_dir);                     \
        _unpack_add_link_type_one_##type(field, mu);                           \
      }                                                                        \
    }                                                                          \
  }

/*Instantiate comm functions for su3_dble */
communication_routines(su3_dble)

/*Instantiate comm functions for su3_alg_dble */
communication_routines(su3_alg_dble)

void copy_boundary_su3_field(su3_dble *su3_field)
{
  _copy_boundary_link_field_su3_dble(su3_field);
}

void add_boundary_su3_field(su3_dble *su3_field)
{
  _add_boundary_link_field_su3_dble(su3_field);
}

void copy_boundary_su3_alg_field(su3_alg_dble *su3_alg_field)
{
  _copy_boundary_link_field_su3_alg_dble(su3_alg_field);
}

void add_boundary_su3_alg_field(su3_alg_dble *su3_alg_field)
{
  _add_boundary_link_field_su3_alg_dble(su3_alg_field);
}
