#define SMEARED_FIELDS_COMMUNICATION_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "stout_smearing.h"
#include "mpi.h"
#include "lattice.h"
#include "global.h"

static int np;
static su3_dble *su3_sbuf = NULL, *su3_rbuf = NULL;
static su3_alg_dble *su3_alg_sbuf = NULL, *su3_alg_rbuf = NULL;
static uidx_t *idx;

/* su3_dble send and recieve routines */

static void allocate_su3_send_recieve_buffers(void)
{
  int mu, nuk, n;

  np = (cpr[0] + cpr[1] + cpr[2] + cpr[3]) & 0x1;
  idx = uidx();
  n = 0;

  /* Find the largest communication size */
  for (mu = 0; mu < 4; mu++) {
    nuk = idx[mu].nuk;

    if (nuk > n)
      n = nuk;
  }

  su3_rbuf = amalloc(n * sizeof(*su3_rbuf), ALIGN);
  error(su3_rbuf == NULL, 1,
        "allocate_send_recieve_buffers [smeared_fields_communication.c]",
        "Unable to allocate recieve buffer");
}

static void send_su3_type_two(int mu)
{
  int nuk, nbf;
  int tag, saddr, raddr;
  MPI_Status stat;

  nuk = idx[mu].nuk;

  if (nuk > 0) {
    tag = mpi_tag();
    saddr = npr[2 * mu];
    raddr = npr[2 * mu + 1];
    nbf = 18 * nuk;

    if (np == 0) {
      MPI_Send(su3_sbuf, nbf, MPI_DOUBLE, saddr, tag, MPI_COMM_WORLD);
      MPI_Recv(su3_rbuf, nbf, MPI_DOUBLE, raddr, tag, MPI_COMM_WORLD, &stat);
    } else {
      MPI_Recv(su3_rbuf, nbf, MPI_DOUBLE, raddr, tag, MPI_COMM_WORLD, &stat);
      MPI_Send(su3_sbuf, nbf, MPI_DOUBLE, saddr, tag, MPI_COMM_WORLD);
    }

    su3_sbuf += nuk;
  }
}

static void send_su3_type_one(int mu)
{
  int nu0, nbf;
  int tag, saddr, raddr;
  MPI_Status stat;

  nu0 = idx[mu].nu0;

  if (nu0 > 0) {
    tag = mpi_tag();
    saddr = npr[2 * mu];
    raddr = npr[2 * mu + 1];
    nbf = 18 * nu0;

    if (np == 0) {
      MPI_Send(su3_sbuf, nbf, MPI_DOUBLE, saddr, tag, MPI_COMM_WORLD);
      MPI_Recv(su3_rbuf, nbf, MPI_DOUBLE, raddr, tag, MPI_COMM_WORLD, &stat);
    } else {
      MPI_Recv(su3_rbuf, nbf, MPI_DOUBLE, raddr, tag, MPI_COMM_WORLD, &stat);
      MPI_Send(su3_sbuf, nbf, MPI_DOUBLE, saddr, tag, MPI_COMM_WORLD);
    }

    su3_sbuf += nu0;
  }
}

static void unpack_su3_type_two(su3_dble *udb, int mu)
{
  int nuk, *iu, *ium;
  su3_dble *u;

  nuk = idx[mu].nuk;

  if (nuk > 0) {
    u = su3_rbuf;
    iu = idx[mu].iuk;
    ium = iu + nuk;

    for (; iu < ium; iu++) {
      cm3x3_add(u, udb + (*iu));
      u += 1;
    }
  }
}

static void unpack_su3_type_one(su3_dble *udb, int mu)
{
  int nu0, *iu, *ium;
  su3_dble *u;

  nu0 = idx[mu].nu0;

  if (nu0 > 0) {
    u = su3_rbuf;
    iu = idx[mu].iu0;
    ium = iu + nu0;

    for (; iu < ium; iu++) {
      cm3x3_add(u, udb + (*iu));
      u += 1;
    }
  }
}

/* su3_alg_dble send and recieve routines */

static void allocate_su3_alg_send_recieve_buffers(void)
{
  int mu, nuk, n;

  np = (cpr[0] + cpr[1] + cpr[2] + cpr[3]) & 0x1;
  idx = uidx();
  n = 0;

  /* Find the largest communication size */
  for (mu = 0; mu < 4; mu++) {
    nuk = idx[mu].nuk;

    if (nuk > n)
      n = nuk;
  }

  su3_alg_sbuf = amalloc(n * sizeof(*su3_alg_sbuf), ALIGN);
  error(su3_alg_sbuf == NULL, 1,
        "allocate_su3_alg_send_recieve_buffers [smeared_fields_communication.c]",
        "Unable to allocate send buffer");
}

static void pack_su3_alg_type_one(su3_alg_dble *algdb, int mu)
{
  int nu0, *iu, *ium;
  su3_alg_dble *ualg;

  nu0 = idx[mu].nu0;

  if (nu0 > 0) {
    ualg = su3_alg_sbuf;
    iu = idx[mu].iu0;
    ium = iu + nu0;

    for (; iu < ium; iu++) {
      (*ualg) = algdb[*iu];
      ualg += 1;
    }
  }
}

static void pack_su3_alg_type_two(su3_alg_dble *algdb, int mu)
{
  int nuk, *iu, *ium;
  su3_alg_dble *ualg;

  nuk = idx[mu].nuk;

  if (nuk > 0) {
    ualg = su3_alg_sbuf;
    iu = idx[mu].iuk;
    ium = iu + nuk;

    for (; iu < ium; iu++) {
      (*ualg) = algdb[*iu];
      ualg += 1;
    }
  }
}

static void send_su3_alg_type_one(int mu)
{
  int nu0, nbf;
  int tag, saddr, raddr;
  MPI_Status stat;

  nu0 = idx[mu].nu0;

  if (nu0 > 0) {
    tag = mpi_tag();
    saddr = npr[2 * mu];
    raddr = npr[2 * mu + 1];
    nbf = 8 * nu0;

    if (np == 0) {
      MPI_Send(su3_alg_sbuf, nbf, MPI_DOUBLE, saddr, tag, MPI_COMM_WORLD);
      MPI_Recv(su3_alg_rbuf, nbf, MPI_DOUBLE, raddr, tag, MPI_COMM_WORLD, &stat);
    } else {
      MPI_Recv(su3_alg_rbuf, nbf, MPI_DOUBLE, raddr, tag, MPI_COMM_WORLD, &stat);
      MPI_Send(su3_alg_sbuf, nbf, MPI_DOUBLE, saddr, tag, MPI_COMM_WORLD);
    }

    su3_alg_rbuf += nu0;
  }
}

static void send_su3_alg_type_two(int mu)
{
  int nuk, nbf;
  int tag, saddr, raddr;
  MPI_Status stat;

  nuk = idx[mu].nuk;

  if (nuk > 0) {
    tag = mpi_tag();
    saddr = npr[2 * mu];
    raddr = npr[2 * mu + 1];
    nbf = 8 * nuk;

    if (np == 0) {
        MPI_Send(su3_alg_sbuf, nbf, MPI_DOUBLE, saddr, tag, MPI_COMM_WORLD);
        MPI_Recv(su3_alg_rbuf, nbf, MPI_DOUBLE, raddr, tag, MPI_COMM_WORLD, &stat);
    } else {
        MPI_Recv(su3_alg_rbuf, nbf, MPI_DOUBLE, raddr, tag, MPI_COMM_WORLD, &stat);
        MPI_Send(su3_alg_sbuf, nbf, MPI_DOUBLE, saddr, tag, MPI_COMM_WORLD);
    }

    su3_alg_rbuf += nuk;
  }
}

/* Public communication routined */

void add_boundary_su3_field(su3_dble *su3_field)
{
  int mu;

  if (NPROC > 1) {
    if (su3_rbuf == NULL)
      allocate_su3_send_recieve_buffers();

    su3_sbuf = su3_field + (4 * VOLUME) + (BNDRY / 4);

    for (mu = 0; mu < 4; mu++) {
      send_su3_type_two(mu);
      unpack_su3_type_two(su3_field, mu);
    }

    su3_sbuf = su3_field + (4 * VOLUME);

    for (mu = 0; mu < 4; mu++) {
      send_su3_type_one(mu);
      unpack_su3_type_one(su3_field, mu);
    }
  }
}

void communicate_boundary_su3_alg_field(su3_alg_dble *alg_field)
{
  int mu;

  if (NPROC > 1) {
    if (su3_alg_sbuf == NULL)
      allocate_su3_alg_send_recieve_buffers();

    su3_alg_rbuf = alg_field + 4 * VOLUME;

    for (mu = 0; mu < 4; mu++) {
      pack_su3_alg_type_one(alg_field, mu);
      send_su3_alg_type_one(mu);
    }

    for (mu = 0; mu < 4; mu++) {
      pack_su3_alg_type_two(alg_field, mu);
      send_su3_alg_type_two(mu);
    }
  }
}
