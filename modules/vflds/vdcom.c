
/*******************************************************************************
 *
 * File vdcom.c
 *
 * Copyright (C) 2007, 2011, 2013 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Communication functions for the global double-precision vector fields.
 *
 *   void cpvd_int_bnd(complex_dble *vd)
 *     Copies the components of the field vd on the interior boundary of
 *     the local block lattice to the corresponding field components at
 *     the exterior boundaries of the block lattices on the neighbouring
 *     MPI processes.
 *
 *   void cpvd_ext_bnd(complex_dble *vd)
 *     *Adds* the components of the field v on the exterior boundary of
 *     the local block lattice to the corresponding field components on
 *     the interior boundaries of the block lattices on the neighbouring
 *     MPI processes.
 *
 * Notes:
 *
 * The fields passed to cpvd_int_bnd() and cpvd_ext_bnd() are interpreted as
 * elements of the deflation subspace spanned by the Ns local modes in the
 * DFL_BLOCKS block grid. They must have at least Ns*(nb+nbb/2) elements,
 * where nb and nbb are the numbers blocks in the DFL_BLOCKS grid and its
 * exterior boundary (see dfl/dfl_geometry.c for further explanations).
 *
 * In the case of boundary conditions of type 0,1 and 2, the programs do not
 * copy any components of the fields across the boundaries of the lattice at
 * global time 0 and NPROC0*L0-1. The program cpvd_int_bnd() instead sets the
 * field at the exterior boundaries of the block lattice at these times to
 * zero.
 *
 * All these programs involve global communications and must be called on all
 * MPI processes simultaneously.
 *
 *******************************************************************************/

#define VDCOM_C
#define OPENQCD_INTERNAL

#include "dfl.h"
#include "flags.h"
#include "global.h"
#include "mpi.h"
#include "vflds.h"

static int bc, np, nmu[8];
static int Ns, nb, nbb;
static int nbbe[8], nbbo[8], obbe[8], obbo[8], *ipp;
static int nsnd, sfc[8], sflg[8];
static complex_dble *snd_buf_int[8], *rcv_buf_int[8];
static complex_dble *snd_buf_ext[8], *rcv_buf_ext[8], *wb = NULL;
static MPI_Request snd_req_int[8], rcv_req_int[8];
static MPI_Request snd_req_ext[8], rcv_req_ext[8];

static void alloc_vdbufs(void)
{
  int ifc, tag, saddr, raddr;
  complex_dble *w;
  dfl_parms_t dfl;
  dfl_grid_t dgr;

  bc = bc_type();
  np = (cpr[0] + cpr[1] + cpr[2] + cpr[3]) & 0x1;
  dfl = dfl_parms();
  Ns = dfl.Ns;

  error_root(Ns == 0, 1, "alloc_vdbufs [vdcom.c]",
             "Deflation subspace parameters are not set");

  dgr = dfl_geometry();
  nb = dgr.nb;
  nbb = dgr.nbb;
  nsnd = 0;

  for (ifc = 0; ifc < 8; ifc++) {
    nmu[ifc] = cpr[ifc / 2] & 0x1;
    nbbe[ifc] = dgr.nbbe[ifc];
    nbbo[ifc] = dgr.nbbo[ifc];
    obbe[ifc] = dgr.obbe[ifc];
    obbo[ifc] = dgr.obbo[ifc];

    if (nbbe[ifc] + nbbo[ifc]) {
      sfc[nsnd] = ifc;
      nsnd += 1;
    }

    sflg[ifc] = ((ifc > 1) || ((ifc == 0) && (cpr[0] != 0)) ||
                 ((ifc == 1) && (cpr[0] != (NPROC0 - 1))) || (bc == 3));
  }

  ipp = dgr.ipp;

  wb = amalloc(Ns * nbb * sizeof(*wb), ALIGN);
  error(wb == NULL, 1, "alloc_vdbufs [vcom.c]",
        "Unable to allocate communication buffers");
  set_vd2zero(Ns * nbb, wb);
  w = wb;

  for (ifc = 0; ifc < 8; ifc++) {
    snd_buf_int[ifc] = w;
    w += Ns * nbbo[ifc];
    rcv_buf_int[ifc] = w;
    w += Ns * nbbe[ifc ^ 0x1];

    tag = mpi_permanent_tag();
    saddr = npr[ifc];
    raddr = npr[ifc ^ 0x1];

    MPI_Send_init(snd_buf_int[ifc], 2 * Ns * nbbo[ifc], MPI_DOUBLE, saddr, tag,
                  MPI_COMM_WORLD, &snd_req_int[ifc]);
    MPI_Recv_init(rcv_buf_int[ifc], 2 * Ns * nbbe[ifc ^ 0x1], MPI_DOUBLE, raddr,
                  tag, MPI_COMM_WORLD, &rcv_req_int[ifc]);
  }

  w = wb;

  for (ifc = 0; ifc < 8; ifc++) {
    snd_buf_ext[ifc] = w;
    w += Ns * nbbe[ifc];
    rcv_buf_ext[ifc] = w;
    w += Ns * nbbo[ifc ^ 0x1];

    tag = mpi_permanent_tag();
    saddr = npr[ifc];
    raddr = npr[ifc ^ 0x1];

    MPI_Send_init(snd_buf_ext[ifc], 2 * Ns * nbbe[ifc], MPI_DOUBLE, saddr, tag,
                  MPI_COMM_WORLD, &snd_req_ext[ifc]);
    MPI_Recv_init(rcv_buf_ext[ifc], 2 * Ns * nbbo[ifc ^ 0x1], MPI_DOUBLE, raddr,
                  tag, MPI_COMM_WORLD, &rcv_req_ext[ifc]);
  }
}

static void get_int(int n, int *imb, complex_dble *v, complex_dble *w)
{
  int *imm;
  complex_dble *vv, *vm;

  imm = imb + n;

  for (; imb < imm; imb++) {
    vv = v + Ns * (*imb);
    vm = vv + Ns;

    for (; vv < vm; vv += 2) {
      w[0] = vv[0];
      w[1] = vv[1];
      w += 2;
    }
  }
}

static void send_bufs_int(int ifc, int eo)
{
  int io;

  io = (ifc ^ nmu[ifc]);

  if (sflg[io]) {
    if (np == eo) {
      if (nbbo[io])
        MPI_Start(&snd_req_int[io]);
    } else {
      if (nbbe[io])
        MPI_Start(&rcv_req_int[io ^ 0x1]);
    }
  }
}

static void wait_bufs_int(int ifc, int eo)
{
  int io;
  MPI_Status stat_snd, stat_rcv;

  io = (ifc ^ nmu[ifc]);

  if (sflg[io]) {
    if (np == eo) {
      if (nbbo[io])
        MPI_Wait(&snd_req_int[io], &stat_snd);
    } else {
      if (nbbe[io])
        MPI_Wait(&rcv_req_int[io ^ 0x1], &stat_rcv);
    }
  }
}

void cpvd_int_bnd(complex_dble *vd)
{
  int ifc, io;
  int n, m, eo;
  complex_dble *vb;

  if (NPROC == 1)
    return;

  if (wb == NULL)
    alloc_vdbufs();

  m = 0;
  eo = 0;
  vb = vd + Ns * nb;

  for (n = 0; n < nsnd; n++) {
    if (n > 0)
      send_bufs_int(sfc[m], eo);

    ifc = sfc[n];
    io = ifc ^ nmu[ifc];

    if (sflg[io])
      get_int(nbbo[io], ipp + obbo[io], vd, snd_buf_int[io]);

    if (n > 0) {
      wait_bufs_int(sfc[m], eo);
      m += eo;
      eo ^= 0x1;
    }
  }

  for (n = 0; n < 2; n++) {
    send_bufs_int(sfc[m], eo);
    wait_bufs_int(sfc[m], eo);
    m += eo;
    eo ^= 0x1;
  }

  for (n = 0; n < nsnd; n++) {
    if (m < nsnd)
      send_bufs_int(sfc[m], eo);

    ifc = sfc[n];
    io = (ifc ^ nmu[ifc]) ^ 0x1;

    if (sflg[io ^ 0x1])
      assign_vd2vd(Ns * nbbe[io ^ 0x1], rcv_buf_int[io],
                   vb + Ns * obbe[io ^ 0x1]);
    else
      set_vd2zero(Ns * nbbe[io ^ 0x1], vb + Ns * obbe[io ^ 0x1]);

    if (m < nsnd) {
      wait_bufs_int(sfc[m], eo);
      m += eo;
      eo ^= 0x1;
    }
  }
}

static void add_ext(int n, int *imb, complex_dble *w, complex_dble *v)
{
  int *imm;
  complex_dble *vv, *vm;

  imm = imb + n;

  for (; imb < imm; imb++) {
    vv = v + Ns * (*imb);
    vm = vv + Ns;

    for (; vv < vm; vv += 2) {
      vv[0].re += w[0].re;
      vv[0].im += w[0].im;
      vv[1].re += w[1].re;
      vv[1].im += w[1].im;
      w += 2;
    }
  }
}

static void send_bufs_ext(int ifc, int eo)
{
  int io;

  io = (ifc ^ nmu[ifc]);

  if (sflg[io]) {
    if (np == eo) {
      if (nbbe[io])
        MPI_Start(&snd_req_ext[io]);
    } else {
      if (nbbo[io])
        MPI_Start(&rcv_req_ext[io ^ 0x1]);
    }
  }
}

static void wait_bufs_ext(int ifc, int eo)
{
  int io;
  MPI_Status stat_snd, stat_rcv;

  io = (ifc ^ nmu[ifc]);

  if (sflg[io]) {
    if (np == eo) {
      if (nbbe[io])
        MPI_Wait(&snd_req_ext[io], &stat_snd);
    } else {
      if (nbbo[io])
        MPI_Wait(&rcv_req_ext[io ^ 0x1], &stat_rcv);
    }
  }
}

void cpvd_ext_bnd(complex_dble *vd)
{
  int ifc, io;
  int n, m, eo;
  complex_dble *vb;

  if (NPROC == 1)
    return;

  if (wb == NULL)
    alloc_vdbufs();

  m = 0;
  eo = 0;
  vb = vd + Ns * nb;

  for (n = 0; n < nsnd; n++) {
    if (n > 0)
      send_bufs_ext(sfc[m], eo);

    ifc = sfc[n];
    io = ifc ^ nmu[ifc];

    if (sflg[io])
      assign_vd2vd(Ns * nbbe[io], vb + Ns * obbe[io], snd_buf_ext[io]);

    if (n > 0) {
      wait_bufs_ext(sfc[m], eo);
      m += eo;
      eo ^= 0x1;
    }
  }

  for (n = 0; n < 2; n++) {
    send_bufs_ext(sfc[m], eo);
    wait_bufs_ext(sfc[m], eo);
    m += eo;
    eo ^= 0x1;
  }

  for (n = 0; n < nsnd; n++) {
    if (m < nsnd)
      send_bufs_ext(sfc[m], eo);

    ifc = sfc[n];
    io = (ifc ^ nmu[ifc]) ^ 0x1;

    if (sflg[io ^ 0x1])
      add_ext(nbbo[io ^ 0x1], ipp + obbo[io ^ 0x1], rcv_buf_ext[io], vd);

    if (m < nsnd) {
      wait_bufs_ext(sfc[m], eo);
      m += eo;
      eo ^= 0x1;
    }
  }
}
