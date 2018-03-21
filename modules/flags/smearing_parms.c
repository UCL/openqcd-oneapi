#define SMEARING_PARMS_C

#include "flags.h"
#include "global.h"
#include "mpi.h"

#define NUM_INT_PARMS 5
#define NUM_DOUBLE_PARMS 2

static int flg_smearing = 0;
static stout_smearing_params_t ssp = {0, 0, 0., 0, 0., 0, 0};

stout_smearing_params_t set_stout_smearing_parms(int n, double pt, double ps,
                                                 int smear_gauge,
                                                 int smear_fermion)
{
  int nprm[5];
  double ssprms[2];

  error(flg_smearing != 0, 1, "set_stout_smearing_parms [smearing_parms.c]",
        "Attempt to reset the stout smearing parameters");

  nprm[0] = n;
  nprm[1] = !(pt == 0.);
  nprm[2] = !(ps == 0.);
  nprm[3] = (smear_gauge) ? 1 : 0;
  nprm[4] = (smear_fermion) ? 1 : 0;

  ssprms[0] = pt;
  ssprms[1] = ps;

  /* Error if you only smear temporal, which doesn't act as one would "expect"
   * */
  error((nprm[0] == 1) && (nprm[1] == 1) && (nprm[2] == 0), 1,
        "set_stout_smearing_parms [smearing_parms.c]",
        "Setting temporal plaquette smearing on and spatial plaquette smearing "
        "off is probably not what you want to do.");

  /* Turn off smearing if both rho's are zero */
  if ((nprm[1] == 0) && (nprm[2] == 0)) {
    nprm[0] = 0;
    /* or if neither smear_gauge nor smear_fermion is on */
  } else if ((nprm[3] == 0) && (nprm[4] == 0)) {
    nprm[0] = 0;
  }

  if (NPROC > 1) {
    MPI_Bcast(nprm, 5, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(ssprms, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  ssp.num_smear = nprm[0];

  ssp.smear_temporal = nprm[1];
  ssp.rho_temporal = ssprms[0];

  ssp.smear_spatial = nprm[2];
  ssp.rho_spatial = ssprms[1];

  ssp.smear_gauge = nprm[3];
  ssp.smear_fermion = nprm[4];

  flg_smearing = 1;

  return ssp;
}

stout_smearing_params_t set_no_stout_smearing_parms(void)
{
  return set_stout_smearing_parms(0, 0.0, 0.0, 0, 0);
}

void reset_stout_smearing(void)
{
  flg_smearing = 0;
  set_no_stout_smearing_parms();
  flg_smearing = 0;
}

stout_smearing_params_t stout_smearing_parms() { return ssp; }

void print_stout_smearing_parms(void)
{
  int my_rank, n;

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    printf("Smearing parameters:\n");
    printf("n_smear = %d\n", ssp.num_smear);
    n = fdigits(ssp.rho_temporal);
    printf("rho_t = %.*f\n", IMAX(n, 1), ssp.rho_temporal);
    n = fdigits(ssp.rho_spatial);
    printf("rho_s = %.*f\n", IMAX(n, 1), ssp.rho_spatial);
    printf("gauge = %s\n", (ssp.smear_gauge == 1) ? "true" : "false");
    printf("fermion = %s\n\n", (ssp.smear_fermion == 1) ? "true" : "false");
  }
}

void write_stout_smearing_parms(FILE *fdat)
{
  int my_rank, endian, iw;
  stdint_t istd[NUM_INT_PARMS];
  double dstd[NUM_DOUBLE_PARMS];

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  endian = endianness();

  error(flg_smearing != 1, 1, "write_stout_smearing_parms [smearing_parms.c]",
        "Attempt to write the stout smearing parameters without setting them "
        "first");

  if (my_rank == 0) {

    istd[0] = ssp.num_smear;
    istd[1] = ssp.smear_temporal;
    istd[2] = ssp.smear_spatial;
    istd[3] = ssp.smear_gauge;
    istd[4] = ssp.smear_fermion;

    dstd[0] = ssp.rho_temporal;
    dstd[1] = ssp.rho_spatial;

    if (endian == BIG_ENDIAN) {
      bswap_int(NUM_INT_PARMS, istd);
      bswap_double(NUM_DOUBLE_PARMS, dstd);
    }

    iw = fwrite(istd, sizeof(stdint_t), NUM_INT_PARMS, fdat);
    iw += fwrite(dstd, sizeof(double), NUM_DOUBLE_PARMS, fdat);

    error_root(iw != (NUM_INT_PARMS + NUM_DOUBLE_PARMS), 1,
               "write_stout_smearing_parms [smearing_parms.c]",
               "Incorrect write count");
  }
}

void check_stout_smearing_parms(FILE *fdat)
{
  int my_rank, endian, ir, ie;
  stdint_t istd[NUM_INT_PARMS];
  double dstd[NUM_DOUBLE_PARMS];

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  endian = endianness();

  if (my_rank == 0) {
    ir = fread(istd, sizeof(stdint_t), NUM_INT_PARMS, fdat);
    ir += fread(dstd, sizeof(double), NUM_DOUBLE_PARMS, fdat);

    if (endian == BIG_ENDIAN) {
      bswap_int(NUM_INT_PARMS, istd);
      bswap_double(NUM_DOUBLE_PARMS, dstd);
    }

    ie = 0;
    ie |= (istd[0] != (stdint_t)(ssp.num_smear));
    ie |= (istd[1] != (stdint_t)(ssp.smear_temporal));
    ie |= (istd[2] != (stdint_t)(ssp.smear_spatial));
    ie |= (istd[3] != (stdint_t)(ssp.smear_gauge));
    ie |= (istd[4] != (stdint_t)(ssp.smear_fermion));

    ie |= (dstd[0] != ssp.rho_temporal);
    ie |= (dstd[1] != ssp.rho_spatial);

    error_root(ir != (NUM_INT_PARMS + NUM_DOUBLE_PARMS), 1,
               "check_stout_smearing_parms [smearing_parms.c]",
               "Incorrect read count");

    error_root(ie != 0, 1, "check_stout_smearing_parms [smearing_parms.c]",
               "Parameters do not match");
  }
}
