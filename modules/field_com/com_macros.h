
/*
 * Communication macros shared between the different source files.
 */

/* Function used to copy an su3_dble */
#define _copy_function_su3_dble(x, y) cm3x3_assign(1, y, x)

/* Function used to copy an su3_alg_dble */
#define _copy_function_su3_alg_dble(x, y) *(x) = *(y)

/* Function used to add assign an su3_dble */
#define _add_function_su3_dble(x, y) cm3x3_add(y, x)

/* Function used to add assign an su3_alg_dble */
#define _add_function_su3_alg_dble(x, y)                                       \
  (x)->c1 += (y)->c1;                                                          \
  (x)->c2 += (y)->c2;                                                          \
  (x)->c3 += (y)->c3;                                                          \
  (x)->c4 += (y)->c4;                                                          \
  (x)->c5 += (y)->c5;                                                          \
  (x)->c6 += (y)->c6;                                                          \
  (x)->c7 += (y)->c7;                                                          \
  (x)->c8 += (y)->c8

/* Declaration of send and receive buffers */
#define communication_buffers(type)                                            \
  static type *_link_sbuf_##type = NULL, *_link_rbuf_##type = NULL

/* Macro definitions of generic commiunication of type one links */
#define type_one_communication_routines(type)                                  \
                                                                               \
  static void _send_link_type_one_##type(int mu, int dir)                      \
  {                                                                            \
    int nu0, nbf;                                                              \
    int tag, saddr, raddr;                                                     \
    MPI_Status stat;                                                           \
                                                                               \
    nu0 = idx[mu].nu0;                                                         \
                                                                               \
    if (nu0 > 0) {                                                             \
      tag = mpi_tag();                                                         \
      saddr = npr[2 * mu + (dir & 0x1)];                                       \
      raddr = npr[2 * mu + ((dir + 1) & 0x1)];                                 \
      nbf = (sizeof(type) / sizeof(double)) * nu0;                             \
                                                                               \
      if (np == 0) {                                                           \
        MPI_Send(_link_sbuf_##type, nbf, MPI_DOUBLE, saddr, tag,               \
                 MPI_COMM_WORLD);                                              \
        MPI_Recv(_link_rbuf_##type, nbf, MPI_DOUBLE, raddr, tag,               \
                 MPI_COMM_WORLD, &stat);                                       \
      } else {                                                                 \
        MPI_Recv(_link_rbuf_##type, nbf, MPI_DOUBLE, raddr, tag,               \
                 MPI_COMM_WORLD, &stat);                                       \
        MPI_Send(_link_sbuf_##type, nbf, MPI_DOUBLE, saddr, tag,               \
                 MPI_COMM_WORLD);                                              \
      }                                                                        \
                                                                               \
      if (dir == positive_send_dir)                                            \
        _link_sbuf_##type += nu0;                                              \
      else                                                                     \
        _link_rbuf_##type += nu0;                                              \
    }                                                                          \
  }                                                                            \
                                                                               \
  static void _pack_link_type_one_##type(type const *field, int mu)            \
  {                                                                            \
    int nu0, *iu, *ium;                                                        \
    type *u;                                                                   \
                                                                               \
    nu0 = idx[mu].nu0;                                                         \
                                                                               \
    if (nu0 > 0) {                                                             \
      u = _link_sbuf_##type;                                                   \
      iu = idx[mu].iu0;                                                        \
      ium = iu + nu0;                                                          \
                                                                               \
      for (; iu < ium; iu++, u++) {                                            \
        _copy_function_##type(u, field + (*iu));                               \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  static void _unpack_add_link_type_one_##type(type *field, int mu)            \
  {                                                                            \
    int nu0, *iu, *ium;                                                        \
    type const *u;                                                             \
                                                                               \
    nu0 = idx[mu].nu0;                                                         \
                                                                               \
    if (nu0 > 0) {                                                             \
      u = _link_rbuf_##type;                                                   \
      iu = idx[mu].iu0;                                                        \
      ium = iu + nu0;                                                          \
                                                                               \
      for (; iu < ium; iu++, u++) {                                            \
        _add_function_##type(field + (*iu), u);                                \
      }                                                                        \
    }                                                                          \
  }
