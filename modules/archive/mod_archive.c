

/*******************************************************************************
 *
 * Felix Ziegler (2020)
 *
 * Modification to the link archive function
 *
 * Notes
 *
 *  export_cnfg routine is modified
 *
 *
*******************************************************************************/

/*******************************************************************************
 *
 * File archive.c
 *
 * Copyright (C) 2005, 2007, 2009-2014 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Programs to read and write gauge-field configurations.
 *
 * The externally accessible functions are
 *
 *   void write_cnfg(char const *out)
 *     Writes the lattice sizes, the process grid sizes, the coordinates
 *     of the calling process, the state of the random number generators,
 *     the local plaquette sum and the local double-precision gauge field
 *     to the file "out".
 *
 *   void read_cnfg(char const *in)
 *     Reads the local double-precision gauge field from the file "in",
 *     assuming it was written to the file by the program write_cnfg().
 *     The program then resets the random number generator and checks
 *     that the restored field is compatible with the chosen boundary
 *     conditions.
 *
 *   void export_cnfg(char const *out)
 *     Writes the lattice sizes and the global double-precision gauge
 *     field to the file "out" from process 0 in the universal format
 *     specified below (see the notes).
 *
 *   void import_cnfg(char const *in)
 *     Reads the global double-precision gauge field from the file "in"
 *     on process 0, assuming the field was written to the file in the
 *     universal format. The field is periodically extended if needed
 *     and the program then checks that the configuration is compatible
 *     with the chosen boundary conditions (see the notes).
 *
 * Notes:
 *
 * The program export_cnfg() first writes the lattice sizes and the average
 * of the plaquette Re(tr{U(p)}) to the output file. Then follow the 8 link
 * variables in the directions +0,-0,...,+3,-3 at the first odd point, the
 * second odd point, and so on. The order of the point (x0,x1,x2,x3) with
 * Cartesian coordinates in the range 0<=x0<N0,...,0<=x3<N3 is determined by
 * the index
 *
 *   ix=x3+N3*x2+N2*N3*x1+N1*N2*N3*x0,
 *
 * where N0,N1,N2,N3 are the global lattice sizes (N0=NPROC0*L0, etc.). The
 * average plaquette is calculated by summing the plaquette values over all
 * plaquettes in the lattice, including the space-like ones at time N0 if
 * SF or open-SF boundary conditions are chosen, and dividing the sum by
 * 6*N0*N1*N2*N3.
 *
 * Independently of the machine, the export function writes the data to the
 * output file in little-endian byte order. Integers and double-precision
 * numbers on the output file occupy 4 and 8 bytes, respectively, the latter
 * being formatted according to the IEEE-754 standard. The import function
 * assumes the data on the input file to be little endian and converts them
 * to big-endian order if the machine is big endian. Exported configurations
 * can thus be safely exchanged between different machines.
 *
 * If the current lattice sizes N0,..,N3 are larger than the lattice sizes
 * n0,..,n3 read from the configuration file, and if N0,..,N3 are integer
 * multiples of n0,..,n3, the program import_cnfg() periodically extends the
 * imported field. An extension in the time direction is only possible with
 * periodic boundary conditions. Note that the boundary values of the link
 * variables (rather than the angles characterizing them) must match in the
 * case of SF and open-SF boundary conditions (see doc/gauge_action.pdf).
 *
 * Compatibility of a configuration with the chosen boundary conditions is
 * established by calling check_bc() [lattice/bcnds.c], with a tolerance on
 * the boundary link variables of 64.0*DBL_EPSILON, and by checking that the
 * average plaquette coincides with the value read from the configuration
 * file. On exit both read_cnfg() and import_cnfg() set the boundary values
 * of the field (if any) to the ones stored in the parameter data base so
 * as to guarantee that they are bit-identical to the latter.
 *
 * All programs in this module may involve global communications and must be
 * called simultaneously on all processes.
 *
 *******************************************************************************/

#define MOD_ARCHIVE_C
#define OPENQCD_INTERNAL

#include "archive.h"
#include "flags.h"
#include "global.h"
#include "lattice.h"
#include "mpi.h"
#include "random.h"
#include "uflds.h"

#define N0 (NPROC0 * L0)
#define N1 (NPROC1 * L1)
#define N2 (NPROC2 * L2)
#define N3 (NPROC3 * L3)

static int endian;
static su3_dble *ubuf = NULL, *vbuf, *udb;

static void check_machine(void)
{
  error_root(sizeof(stdint_t) != 4, 1, "check_machine [archive.c]",
             "Size of a stdint_t integer is not 4");
  error_root(sizeof(double) != 8, 1, "check_machine [archive.c]",
             "Size of a double is not 8");

  endian = endianness();
  error_root(endian == openqcd_utils__UNKNOWN_ENDIAN, 1, "check_machine [archive.c]",
             "Unkown endianness");
}

static void alloc_ubuf(int my_rank)
{
  if (my_rank == 0) {
    ubuf = amalloc(4 * (L3 + N3) * sizeof(su3_dble), ALIGN);
    vbuf = ubuf + 4 * L3;
  } else {
    ubuf = amalloc(4 * L3 * sizeof(su3_dble), ALIGN);
    vbuf = NULL;
  }

  error(ubuf == NULL, 1, "alloc_ubuf [archive.c]",
        "Unable to allocate auxiliary array");
}

static void get_links(int iy)
{
  int y3, ifc;
  su3_dble *u, *v;

  v = ubuf;
  iy *= L3;

  if (ipt[iy] < (VOLUME / 2)) {
    iy += 1;
  }

  for (y3 = 0; y3 < L3; y3 += 2) {
    u = udb + 8 * (ipt[iy + y3] - (VOLUME / 2));

    for (ifc = 0; ifc < 8; ifc++) {
      v[0] = u[0];
      v += 1;
      u += 1;
    }
  }
}

void mod_export_cnfg(char const *out)
{
  int my_rank, np[4], n, iw;
  int iwa, dmy, tag0, tag1;
  int x0, x1, x2, x3, y0, y1, y2, ix, iy;
  stdint_t lsize[4];
  double nplaq, plaq;
  MPI_Status stat;
  FILE *fout = NULL;

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  error_root(query_flags(UD_IS_CLEAN) == 0, 1, "export_cnfg [archive.c]",
             "Attempt to export a modified (phase|smearing) gauge field");

  if (ubuf == NULL) {
    check_machine();
    alloc_ubuf(my_rank);
  }

  dmy = 1;
  tag0 = mpi_tag();
  tag1 = mpi_tag();
  nplaq = (double)(6 * N0 * N1) * (double)(N2 * N3);
  plaq = plaq_sum_dble(1) / nplaq;

  if (my_rank == 0) {
    fout = fopen(out, "wb");
    error_root(fout == NULL, 1, "export_cnfg [archive.c]",
               "Unable to open output file");

    lsize[0] = (stdint_t)(N0);
    lsize[1] = (stdint_t)(N1);
    lsize[2] = (stdint_t)(N2);
    lsize[3] = (stdint_t)(N3);

    if (endian == openqcd_utils__BIG_ENDIAN) {
      bswap_int(4, lsize);
      bswap_double(1, &plaq);
    }

    /*
    iw = fwrite(lsize, sizeof(stdint_t), 4, fout);
    iw += fwrite(&plaq, sizeof(double), 1, fout);

    error_root(iw != 5, 1, "export_cnfg [archive.c]", "Incorrect write count");
    */
  }

  iwa = 0;
  udb = udfld();

  for (ix = 0; ix < (N0 * N1 * N2); ix++) {
    x0 = ix / (N1 * N2);
    x1 = (ix / N2) % N1;
    x2 = ix % N2;

    y0 = x0 % L0;
    y1 = x1 % L1;
    y2 = x2 % L2;
    iy = y2 + L2 * y1 + L1 * L2 * y0;

    np[0] = x0 / L0;
    np[1] = x1 / L1;
    np[2] = x2 / L2;

    for (x3 = 0; x3 < N3; x3 += L3) {
      np[3] = x3 / L3;
      n = ipr_global(np);
      if (my_rank == n) {
        get_links(iy);
      }

      if (n > 0) {
        if (my_rank == 0) {
          MPI_Send(&dmy, 1, MPI_INT, n, tag0, MPI_COMM_WORLD);
          MPI_Recv(ubuf, 4 * L3 * 18, MPI_DOUBLE, n, tag1, MPI_COMM_WORLD,
                   &stat);
        } else if (my_rank == n) {
          MPI_Recv(&dmy, 1, MPI_INT, 0, tag0, MPI_COMM_WORLD, &stat);
          MPI_Send(ubuf, 4 * L3 * 18, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD);
        }
      }

      if (my_rank == 0) {
        if (endian == openqcd_utils__BIG_ENDIAN) {
          bswap_double(4 * L3 * 18, ubuf);
        }
        iw = fwrite(ubuf, sizeof(su3_dble), 4 * L3, fout);
        iwa |= (iw != (4 * L3));
      }
    }
  }

  if (my_rank == 0) {
    error_root(iwa != 0, 1, "export_cnfg [archive.c]", "Incorrect write count");
    fclose(fout);
  }
}

