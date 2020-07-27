#include "dw_cuda_soa.h"
#include "read_in.h"
#include "su3.h"
#include "macros.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


int main(int argc, char *argv[])
{
    int L0, L1, L2, L3;
    int VOLUME;

    su3 *u;
    pauli *m;
    spinor *s, *r, *rdiag;
    int *piup, *pidn;

    char cnfg[256];
    char cnfg_dir[128];

    L0 = atoi(argv[1]);
    L1 = atoi(argv[2]);
    L2 = atoi(argv[3]);
    L3 = atoi(argv[4]);
    VOLUME = L0 * L1 * L2 * L3;
    sprintf(cnfg_dir, argv[5]);

    /***************************************************************************
     *
     * Allocate memory and load data from disk
     *
     **************************************************************************/

    u = (su3 *) malloc((4 * VOLUME) * sizeof(*u));
    m = (pauli *) malloc((2 * VOLUME) * sizeof(*m));
    piup = (int *) malloc((2 * VOLUME) * sizeof(*piup));
    pidn = (int *) malloc((2 * VOLUME) * sizeof(*pidn));
    s = (spinor *) malloc(VOLUME * sizeof(*s));
    r = (spinor *) malloc(VOLUME * sizeof(*r));
    rdiag = (spinor *) malloc(VOLUME * sizeof(*rdiag));

    sprintf(cnfg, "%s/sp-u-%d-%d-%d-%d", cnfg_dir, L0, L1, L2, L3);
    read_sp_u_from_file(cnfg, u, VOLUME);

    sprintf(cnfg, "%s/sp-m-%d-%d-%d-%d", cnfg_dir, L0, L1, L2, L3);
    read_sp_m_from_file(cnfg, m, VOLUME);

    sprintf(cnfg, "%s/piup-%d-%d-%d-%d", cnfg_dir, L0, L1, L2, L3);
    read_lt_from_file(cnfg, piup, VOLUME);

    sprintf(cnfg, "%s/pidn-%d-%d-%d-%d", cnfg_dir, L0, L1, L2, L3);
    read_lt_from_file(cnfg, pidn, VOLUME);

    sprintf(cnfg, "%s/sp-s-%d-%d-%d-%d", cnfg_dir, L0, L1, L2, L3);
    read_sp_spinor_from_file(cnfg, s, VOLUME);

    // sprintf(cnfg, "%s/sp-r-%d-%d-%d-%d", cnfg_dir, L0, L1, L2, L3);
    // read_sp_spinor_from_file(cnfg, r, VOLUME);

    sprintf(cnfg, "%s/sp-rdiag-%d-%d-%d-%d", cnfg_dir, L0, L1, L2, L3);
    read_sp_spinor_from_file(cnfg, rdiag, VOLUME);

    Dw_cuda_SoA(VOLUME, u, s, r, m, piup, pidn);

    // Compare
    float dev;
    int devcount;
    float *fpr, *fprdiag;
    fpr = (float*) r;
    fprdiag = (float*) rdiag;

    devcount = 0;
    for(int ix = 0; ix < 24 * VOLUME; ix++, fpr++, fprdiag++)
    {
      dev = fabs((*fpr) - (*fprdiag));
      if(dev > 1.0e-6)
      {
          devcount++;
          printf("dev = %.8e\n", dev);
      }

    }

    printf("devcount = %d out of %d\n", devcount, 24 * VOLUME);

    return 0;
}
