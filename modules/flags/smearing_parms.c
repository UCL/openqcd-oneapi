#define SMEARING_PARMS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "utils.h"
#include "flags.h"
#include "global.h"

static stout_smearing_params_t ssp = {0, 0, 0., 0, 0.};

stout_smearing_params_t set_stout_smearing_parms(int n, double pt, double ps)
{
  int nprm[3];
  double ssprms[2];

  nprm[0] = n;
  nprm[1] = !(pt == 0.);
  nprm[2] = !(ps == 0.);
  ssprms[0] = pt;
  ssprms[1] = ps;

  /* Error if you only smear temporal, which doesn't act as one would "expect"
   * */
  error((nprm[0] == 1) && (nprm[1] == 1) && (nprm[2] == 0), 1,
        "set_stout_smearing_parms [smearing_parms.c]",
        "Setting temporal plaquette smearing on and spatial plaquette smearing "
        "off is probably not what you want to do.");

  /* Turn off smearing if both rho's are zero */
  if ((nprm[1] == 0) && (nprm[2] == 0))
    nprm[0] = 0;

  if (NPROC > 1) {
    MPI_Bcast(nprm, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(ssprms, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  ssp.num_smear = nprm[0];

  ssp.smear_temporal = nprm[1];
  ssp.rho_temporal = ssprms[0];

  ssp.smear_spatial = nprm[2];
  ssp.rho_spatial = ssprms[1];

  return ssp;
}

stout_smearing_params_t sout_smearing_parms() { return ssp; }
