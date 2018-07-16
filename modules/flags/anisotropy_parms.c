
/*******************************************************************************
 *
 * File anisotropy_parms.c
 *
 * Authors (2017, 2018): Jonas Rylund Glesaaen, Benjamin Jäger
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Anisotropy parameter database.
 *
 * The externally accessible functions are
 *
 *   ani_params_t set_ani_parms(int use_tts, double nu, double xi, double cR,
 *                              double cT, double us, double ut, double ust,
 *                              double utt)
 *     Sets the anisotropy parameters and returns a structure containing them.
 *     See the documentation with respect to their meanings.
 *
 *   ani_params_t set_no_ani_parms(void)
 *     Sets the parameters as if we had an isotropic simulation.
 *
 *   ani_params_t ani_parms(void)
 *     Return a structure that contains the current anisotropy parameters.
 *
 *   void print_ani_parms(void)
 *     Prints the anisotropy parameters to stdout on MPI process 0.
 *
 *   int ani_params_initialised(void)
 *     Returns whether the parameters have been initialised or not.
 *
 *   void write_ani_parms(FILE *fdat)
 *     Writes the anisotropy parameters to the file fdat on MPI process 0.
 *
 *   void check_ani_parms(FILE *fdat)
 *     Compares the anisotropy parameters with the values stored on the file
 *     fdat on MPI process 0, assuming the latter were written to the file by
 *     the program write_ani_parms().
 *
 * Notes:
 *
 * Except for ani_parms() and print_ani_parms(), the programs in this module
 * perform global operations and must be called simultaneously on all MPI
 * processes.
 *
 *******************************************************************************/

#define ANISOTROPY_PARMS_C
#define OPENQCD_INTERNAL

#include "flags.h"
#include "global.h"
#include "mpi.h"

#define N0 (NPROC0 * L0)
#define N1 (NPROC1 * L1)
#define N2 (NPROC2 * L2)
#define N3 (NPROC3 * L3)

static int flg_ani = 0;
static ani_params_t ani = {0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

ani_params_t set_ani_parms(int use_tts, double nu, double xi, double cR,
                           double cT, double us_gauge, double ut_gauge,
                           double us_fermion, double ut_fermion)
{
  int iprms[2];
  double dprms[8];

  error(flg_ani != 0, 1, "set_ani_parms [lat_parms.c]",
        "Attempt to reset the anisotropy parameters");

  if ((is_equal_d(nu, 1.0)) && (is_equal_d(xi, 1.0)) && (is_equal_d(cR, 1.0)) &&
      (is_equal_d(cT, 1.0)) && (is_equal_d(us_gauge, 1.0)) &&
      (is_equal_d(ut_gauge, 1.0)) && (is_equal_d(us_fermion, 1.0)) &&
      (is_equal_d(ut_fermion, 1.0))) {
    iprms[0] = 0;
  } else {
    iprms[0] = 1;
  }

  if (use_tts) {
    iprms[1] = 1;
  } else {
    iprms[1] = 0;
  }

  dprms[0] = nu;
  dprms[1] = xi;
  dprms[2] = cR;
  dprms[3] = cT;
  dprms[4] = us_gauge;
  dprms[5] = ut_gauge;
  dprms[6] = us_fermion;
  dprms[7] = ut_fermion;

  if (NPROC > 1) {
    MPI_Bcast(iprms, 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(dprms, 8, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  ani.has_ani = iprms[0];
  ani.has_tts = iprms[1];
  ani.nu = dprms[0];
  ani.xi = dprms[1];
  ani.cR = dprms[2];
  ani.cT = dprms[3];
  ani.us_gauge = dprms[4];
  ani.ut_gauge = dprms[5];
  ani.us_fermion = dprms[6];
  ani.ut_fermion = dprms[7];

  flg_ani = 1;

  return ani;
}

ani_params_t set_no_ani_parms(void)
{
  return set_ani_parms(1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
}

ani_params_t ani_parms(void)
{
  return ani;
}

void print_ani_parms(void)
{
  int my_rank, n;

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    printf("Anisotropy parameters:\n");
    printf("use tts = %s\n", ani.has_tts ? "true" : "false");
    n = fdigits(ani.nu);
    printf("nu = %.*f\n", IMAX(n, 1), ani.nu);
    n = fdigits(ani.xi);
    printf("xi = %.*f\n", IMAX(n, 1), ani.xi);
    n = fdigits(ani.cR);
    printf("cR = %.*f\n", IMAX(n, 1), ani.cR);
    n = fdigits(ani.cT);
    printf("cT = %.*f\n", IMAX(n, 1), ani.cT);
    n = fdigits(ani.ut_gauge);
    printf("ut_gauge = %.*f\n", IMAX(n, 1), ani.ut_gauge);
    n = fdigits(ani.us_gauge);
    printf("us_gauge = %.*f\n", IMAX(n, 1), ani.us_gauge);
    n = fdigits(ani.us_fermion);
    printf("us_fermion = %.*f\n", IMAX(n, 1), ani.us_fermion);
    n = fdigits(ani.ut_fermion);
    printf("ut_fermion = %.*f\n\n", IMAX(n, 1), ani.ut_fermion);
  }
}

void write_ani_parms(FILE *fdat)
{
  int my_rank, endian, iw;
  stdint_t istd[2];
  double dstd[8];

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  endian = endianness();

  error(
      !ani_params_initialised(), 1, "write_ani_parms [lat_parms.c]",
      "Attempt to write the anisotropy parameters without setting them first");

  if (my_rank == 0) {

    istd[0] = ani.has_ani;
    istd[1] = ani.has_tts;

    dstd[0] = ani.nu;
    dstd[1] = ani.xi;
    dstd[2] = ani.cR;
    dstd[3] = ani.cT;
    dstd[4] = ani.us_gauge;
    dstd[5] = ani.ut_gauge;
    dstd[6] = ani.us_fermion;
    dstd[7] = ani.ut_fermion;

    if (endian == openqcd_utils__BIG_ENDIAN) {
      bswap_int(2, istd);
      bswap_double(8, dstd);
    }

    iw = fwrite(istd, sizeof(stdint_t), 2, fdat);
    iw += fwrite(dstd, sizeof(double), 8, fdat);

    error_root(iw != 10, 1, "write_ani_parms [lat_parms.c]",
               "Incorrect write count");
  }
}

void check_ani_parms(FILE *fdat)
{
  int my_rank, endian, ir, ie;
  stdint_t istd[2];
  double dstd[8];

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  endian = endianness();

  if (my_rank == 0) {
    ir = fread(istd, sizeof(stdint_t), 2, fdat);
    ir += fread(dstd, sizeof(double), 8, fdat);

    if (endian == openqcd_utils__BIG_ENDIAN) {
      bswap_int(2, istd);
      bswap_double(8, dstd);
    }

    ie = 0;
    ie |= (istd[0] != (stdint_t)(ani.has_ani));
    ie |= (istd[1] != (stdint_t)(ani.has_tts));

    ie |= (dstd[0] != ani.nu);
    ie |= (dstd[1] != ani.xi);
    ie |= (dstd[2] != ani.cR);
    ie |= (dstd[3] != ani.cT);
    ie |= (dstd[4] != ani.us_gauge);
    ie |= (dstd[5] != ani.ut_gauge);
    ie |= (dstd[6] != ani.us_fermion);
    ie |= (dstd[7] != ani.ut_fermion);

    error_root(ir != 10, 1, "check_ani_parms [lat_parms.c]",
               "Incorrect read count");

    error_root(ie != 0, 1, "check_ani_parms [lat_parms.c]",
               "Parameters do not match");
  }
}

int ani_params_initialised(void)
{
  return (flg_ani == 1);
}

ani_params_t reset_ani_parms(int use_tts, double nu, double xi, double cR,
                             double cT, double us_gauge, double ut_gauge,
                             double us_fermion, double ut_fermion)
{
  flg_ani = 0;
  set_ani_parms(use_tts, nu, xi, cR, cT, us_gauge, ut_gauge, us_fermion,
                ut_fermion);
  recompute_sea_quark_masses();
  return ani;
}
