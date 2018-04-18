
/*******************************************************************************
 *
 * File link_partial_communication.c
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Partial communication routines for "link like" problems
 *
 * The externally accessible functions are
 *
 *   void copy_partial_boundary_su3_field(su3_dble *su3_field, int const *dirs)
 *      Copy the boundaries of an su3_dble field to the neighbouring processes
 *      in directions -mu. However, depending on the contents of the dirs array
 *      it will ignore certain links. The dirs array is assumed to be a length 4
 *      array with boolean elements (0 or 1), and is so that if e.g. element nu
 *      is 1 all boundary links pointing in direction nu will be copied, they
 *      will be ignored otherwise.
 *
 *   void add_partial_boundary_su3_field(su3_dble *su3_field, int const *dirs)
 *      Add the current value of an su3_field's boundary to its neighbouring
 *      processes corresponding links in the +mu direction applying the dirs map
 *      as explained.
 *
 *   void copy_partial_boundary_su3_alg_field(su3_alg_dble *su3_alg_field,
 *                                            int const *dirs)
 *      Copy the boundaries of an su3_alg_dble field to the neighbouring
 *      processes in directions -mu applying the dirs map as explained.
 *
 *   void add_partial_boundary_su3_alg_field(su3_alg_dble *su3_alg_field,
 *                                           int const *dirs)
 *      Add the current value of an su3_alg_field's boundary to its neighbouring
 *      processes corresponding links in the +mu direction applying the dirs map
 *      as explained.
 *
 *   void copy_spatial_boundary_su3_field(su3_dble *su3_field)
 *      A special case of the more general partial variant where dirs={0,1,1,1}.
 *
 *   void add_spatial_boundary_su3_field(su3_dble *su3_field)
 *      A special case of the more general partial variant where dirs={0,1,1,1}.
 *
 *   void copy_spatial_boundary_su3_alg_field(su3_alg_dble *su3_alg_field)
 *      A special case of the more general partial variant where dirs={0,1,1,1}.
 *
 *   void add_spatial_boundary_su3_alg_field(su3_alg_dble *su3_alg_field)
 *      A special case of the more general partial variant where dirs={0,1,1,1}.
 *
 * Notes:
 *
 * The routines requires the allocation of one more memory buffer due to the
 * fact that the boundaries will no longer be contiguous memory. However it is a
 * tradeoff in return for smaller communication sizes for problems where certain
 * links might be left out. An example of this is the stout smearing routines
 * for parameters such that we do not smear the temporal links.
 *
 * All routines carries out communication and must therefore be call on all
 * processes simultaneously.
 *
 *******************************************************************************/

#define LINK_PARTIAL_COMMUNICATION_C

#include "field_com.h"
#include "flags.h"
#include "global.h"
#include "lattice.h"
#include "mpi.h"
#include "su3fcts.h"
#include <string.h>

#include "com_macros.h"

static int bc, np, my_rank;
static size_t offsets[4], num_links[4];
static double *send_receive_buffer = NULL, *extra_pack_buffer = NULL;
static uidx_t *idx;

static const int positive_send_dir = 1;
static const int negative_send_dir = 0;

static void init_buffer(void)
{
  int mu, nuk, n;

  bc = bc_type();
  np = (cpr[0] + cpr[1] + cpr[2] + cpr[3]) & 0x1;
  idx = uidx();
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  num_links[0] = 3 * FACE0;
  num_links[1] = 3 * FACE1;
  num_links[2] = 3 * FACE2;
  num_links[3] = 3 * FACE3;

  offsets[0] = 4 * VOLUME + BNDRY / 4;
  offsets[1] = offsets[0] + num_links[0];
  offsets[2] = offsets[1] + num_links[1];
  offsets[3] = offsets[2] + num_links[2];

  n = 0;

  /* Find the largest communication size */
  for (mu = 0; mu < 4; mu++) {
    nuk = idx[mu].nuk;

    if (nuk > n) {
      n = nuk;
    }
  }

  send_receive_buffer = communication_buffer();
  extra_pack_buffer = (double *)amalloc((n / 3) * 2 * sizeof(su3_dble), ALIGN);

  error(extra_pack_buffer == NULL, 1,
        "allocate_send_recieve_buffers [link_partial_communication.c]",
        "Unable to allocate send recieve buffer");
}

/* Reduce the length 4 dirs to a length 3 send_dirs signifying the available
 * directions of the current type two links */
static void fill_send_dirs(int mu, int const *dirs, int *send_dirs)
{
  int i, j;

  for (i = 0, j = 0; i < 4; ++i) {
    if (i == mu) {
      continue;
    }
    send_dirs[j] = dirs[i];
    ++j;
  }
}

/* Count the number of type two directions for current mu */
static int count_dirs(int mu, int const *dirs)
{
  int i, count = 0;

  for (i = 0; i < 4; ++i) {
    if (i == mu) {
      continue;
    }
    count += dirs[i];
  }

  return count;
}

/* Macro definitions of type two commiunication routines */
#define type_two_partial_communication_routines(type)                          \
                                                                               \
  static void _send_partial_link_type_two_##type(int mu, int com_dir,          \
                                                 int const *dirs)              \
  {                                                                            \
    int nuk, nbf;                                                              \
    int tag, saddr, raddr;                                                     \
    int send_count;                                                            \
    MPI_Status stat;                                                           \
                                                                               \
    send_count = count_dirs(mu, dirs);                                         \
    nuk = idx[mu].nuk;                                                         \
                                                                               \
    if (nuk > 0) {                                                             \
      tag = mpi_tag();                                                         \
      saddr = npr[2 * mu + (com_dir & 0x1)];                                   \
      raddr = npr[2 * mu + ((com_dir + 1) & 0x1)];                             \
      nbf = sizeof(type) * (send_count * (nuk / 3)) / sizeof(double);          \
                                                                               \
      if (np == 0) {                                                           \
        if ((mu > 0) || (bc == 3) ||                                           \
            ((com_dir == positive_send_dir) && (cpr[0] < (NPROC0 - 1))) ||     \
            ((com_dir == negative_send_dir) && (cpr[0] > 0)))                  \
          MPI_Send(_link_sbuf_##type, nbf, MPI_DOUBLE, saddr, tag,             \
                   MPI_COMM_WORLD);                                            \
                                                                               \
        if ((mu > 0) || (bc == 3) ||                                           \
            ((com_dir == negative_send_dir) && (cpr[0] < (NPROC0 - 1))) ||     \
            ((com_dir == positive_send_dir) && (cpr[0] > 0)))                  \
          MPI_Recv(_link_rbuf_##type, nbf, MPI_DOUBLE, raddr, tag,             \
                   MPI_COMM_WORLD, &stat);                                     \
      } else {                                                                 \
        if ((mu > 0) || (bc == 3) ||                                           \
            ((com_dir == negative_send_dir) && (cpr[0] < (NPROC0 - 1))) ||     \
            ((com_dir == positive_send_dir) && (cpr[0] > 0)))                  \
          MPI_Recv(_link_rbuf_##type, nbf, MPI_DOUBLE, raddr, tag,             \
                   MPI_COMM_WORLD, &stat);                                     \
                                                                               \
        if ((mu > 0) || (bc == 3) ||                                           \
            ((com_dir == positive_send_dir) && (cpr[0] < (NPROC0 - 1))) ||     \
            ((com_dir == negative_send_dir) && (cpr[0] > 0)))                  \
          MPI_Send(_link_sbuf_##type, nbf, MPI_DOUBLE, saddr, tag,             \
                   MPI_COMM_WORLD);                                            \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  static void _pack_extra_buffer_##type(int mu, int const *dirs,               \
                                        type const *field)                     \
  {                                                                            \
    type *v;                                                                   \
    type const *u, *um;                                                        \
    int nu;                                                                    \
    int send_dirs[3];                                                          \
                                                                               \
    fill_send_dirs(mu, dirs, send_dirs);                                       \
                                                                               \
    v = (type *)extra_pack_buffer;                                             \
    u = field + offsets[mu];                                                   \
    um = u + num_links[mu];                                                    \
                                                                               \
    for (; u < um;) {                                                          \
      for (nu = 0; nu < 3; ++nu, ++u) {                                        \
        if (send_dirs[nu]) {                                                   \
          _copy_function_##type(v, u);                                         \
          ++v;                                                                 \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  static void _unpack_extra_buffer_##type(int mu, int const *dirs,             \
                                          type *field)                         \
  {                                                                            \
    type const *v;                                                             \
    type *u, *um;                                                              \
    int nu;                                                                    \
    int send_dirs[3];                                                          \
                                                                               \
    fill_send_dirs(mu, dirs, send_dirs);                                       \
                                                                               \
    v = (type *)extra_pack_buffer;                                             \
    u = field + offsets[mu];                                                   \
    um = u + num_links[mu];                                                    \
                                                                               \
    for (; u < um;) {                                                          \
      for (nu = 0; nu < 3; ++nu, ++u) {                                        \
        if (send_dirs[nu]) {                                                   \
          _copy_function_##type(u, v);                                         \
          ++v;                                                                 \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  static void _pack_partial_link_type_two_##type(type const *field, int mu,    \
                                                 int const *dirs)              \
  {                                                                            \
    int nu, nuk, *iu, *ium;                                                    \
    int send_dirs[3];                                                          \
    type *u;                                                                   \
                                                                               \
    fill_send_dirs(mu, dirs, send_dirs);                                       \
    nuk = idx[mu].nuk;                                                         \
                                                                               \
    if ((nuk > 0) && ((mu > 0) || (cpr[0] > 0) || (bc == 3))) {                \
      u = _link_sbuf_##type;                                                   \
      iu = idx[mu].iuk;                                                        \
      ium = iu + nuk;                                                          \
                                                                               \
      for (; iu < ium; iu += 3) {                                              \
        for (nu = 0; nu < 3; ++nu) {                                           \
          if (send_dirs[nu]) {                                                 \
            _copy_function_##type(u, field + *(iu + nu));                      \
            ++u;                                                               \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  static void _unpack_add_partial_link_type_two_##type(type *field, int mu,    \
                                                       int const *dirs)        \
  {                                                                            \
    int nu, nuk, *iu, *ium;                                                    \
    int send_dirs[3];                                                          \
    type const *u;                                                             \
                                                                               \
    fill_send_dirs(mu, dirs, send_dirs);                                       \
    nuk = idx[mu].nuk;                                                         \
                                                                               \
    if ((nuk > 0) && ((mu > 0) || (cpr[0] > 0) || (bc == 3))) {                \
      u = _link_rbuf_##type;                                                   \
      iu = idx[mu].iuk;                                                        \
      ium = iu + nuk;                                                          \
                                                                               \
      for (; iu < ium; iu += 3) {                                              \
        for (nu = 0; nu < 3; ++nu) {                                           \
          if (send_dirs[nu]) {                                                 \
            _add_function_##type(field + *(iu + nu), u);                       \
            ++u;                                                               \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

/* Define the main communication routines */
/* clang-format off */
#define communication_routines(type)                                           \
                                                                               \
  communication_buffers(type);                                                 \
  type_one_communication_routines(type)                                       \
  type_two_partial_communication_routines(type)                               \
                                                                               \
  static void _copy_partial_boundary_link_field_##type(type *field,            \
                                                       int const *dirs)        \
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
        if (dirs[mu]) {                                                        \
          _pack_link_type_one_##type(field, mu);                               \
          _send_link_type_one_##type(mu, negative_send_dir);                   \
        } else {                                                               \
          _link_rbuf_##type += idx[mu].nu0;                                    \
        }                                                                      \
      }                                                                        \
                                                                               \
      for (mu = 0; mu < 4; mu++) {                                             \
        _pack_partial_link_type_two_##type(field, mu, dirs);                   \
        if (count_dirs(mu, dirs) == 3) {                                       \
          _link_rbuf_##type = field + offsets[mu];                             \
        } else {                                                               \
          _link_rbuf_##type = (type *)extra_pack_buffer;                       \
        }                                                                      \
        _send_partial_link_type_two_##type(mu, negative_send_dir, dirs);       \
        if (count_dirs(mu, dirs) != 3) {                                       \
          _unpack_extra_buffer_##type(mu, dirs, field);                        \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  static void _add_partial_boundary_link_field_##type(type *field,             \
                                                      int const *dirs)         \
  {                                                                            \
    int mu;                                                                    \
                                                                               \
    if (NPROC > 1) {                                                           \
      if (send_receive_buffer == NULL)                                         \
        init_buffer();                                                         \
                                                                               \
      _link_rbuf_##type = (type *)send_receive_buffer;                         \
                                                                               \
      for (mu = 0; mu < 4; mu++) {                                             \
        if (count_dirs(mu, dirs) == 3) {                                       \
          _link_sbuf_##type = field + offsets[mu];                             \
        } else {                                                               \
          _pack_extra_buffer_##type(mu, dirs, field);                          \
          _link_sbuf_##type = (type *)extra_pack_buffer;                       \
        }                                                                      \
        _send_partial_link_type_two_##type(mu, positive_send_dir, dirs);       \
        _unpack_add_partial_link_type_two_##type(field, mu, dirs);             \
      }                                                                        \
                                                                               \
      _link_sbuf_##type = field + (4 * VOLUME);                                \
                                                                               \
      for (mu = 0; mu < 4; mu++) {                                             \
        if (dirs[mu]) {                                                        \
          _send_link_type_one_##type(mu, positive_send_dir);                   \
          _unpack_add_link_type_one_##type(field, mu);                         \
        } else {                                                               \
          _link_sbuf_##type += idx[mu].nu0;                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

/*Instantiate comm functions for su3_dble */
communication_routines(su3_dble)

/*Instantiate comm functions for su3_alg_dble */
communication_routines(su3_alg_dble)

void copy_partial_boundary_su3_field(su3_dble *su3_field, int const *dirs)
{
  _copy_partial_boundary_link_field_su3_dble(su3_field, dirs);
}

void add_partial_boundary_su3_field(su3_dble *su3_field, int const *dirs)
{
  _add_partial_boundary_link_field_su3_dble(su3_field, dirs);
}

void copy_partial_boundary_su3_alg_field(su3_alg_dble *su3_alg_field,
                                         int const *dirs)
{
  _copy_partial_boundary_link_field_su3_alg_dble(su3_alg_field, dirs);
}

void add_partial_boundary_su3_alg_field(su3_alg_dble *su3_alg_field,
                                        int const *dirs)
{
  _add_partial_boundary_link_field_su3_alg_dble(su3_alg_field, dirs);
}

void copy_spatial_boundary_su3_field(su3_dble *su3_field)
{
  int dirs[4] = {0, 1, 1, 1};
  copy_partial_boundary_su3_field(su3_field, dirs);
}

void add_spatial_boundary_su3_field(su3_dble *su3_field)
{
  int dirs[4] = {0, 1, 1, 1};
  add_partial_boundary_su3_field(su3_field, dirs);
}

void copy_spatial_boundary_su3_alg_field(su3_alg_dble *su3_alg_field)
{
  int dirs[4] = {0, 1, 1, 1};
  copy_partial_boundary_su3_alg_field(su3_alg_field, dirs);
}

void add_spatial_boundary_su3_alg_field(su3_alg_dble *su3_alg_field)
{
  int dirs[4] = {0, 1, 1, 1};
  add_partial_boundary_su3_alg_field(su3_alg_field, dirs);
}
