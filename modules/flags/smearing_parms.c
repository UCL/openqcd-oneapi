#define SMEARING_PARMS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "utils.h"
#include "flags.h"
#include "global.h"

static stout_smearing_params_t ssp = {0, 0, 0., 0, 0., 0, 0};

stout_smearing_params_t set_stout_smearing_parms(int n, double pt, double ps,
                                                 int smear_gauge,
                                                 int smear_fermion)
{
  int nprm[5];
  double ssprms[2];

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

  return ssp;
}

stout_smearing_params_t stout_smearing_parms() { return ssp; }

void print_smearing_parms(void)
{
  int my_rank, n;

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    printf("Smearing parameters:\n");
    printf("n_smear = %d\n", ssp.num_smear);
    n = fdigits(ssp.rho_temporal);
    printf("rho_t = %.*f\n", IMAX(n,1), ssp.rho_temporal);
    n = fdigits(ssp.rho_spatial);
    printf("rho_s = %.*f\n", IMAX(n,1), ssp.rho_spatial);
    printf("gauge = %s\n", (ssp.smear_gauge == 1) ? "true" : "false");
    printf("fermion = %s\n\n", (ssp.smear_fermion == 1) ? "true" : "false");
  }
}
