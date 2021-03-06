#include "dw_cuda.h"
#include "read_in.h"
#include "su3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


long my_memcmp(const void *Ptr1, const void *Ptr2, size_t Count)
{
    float *p1 = (float *)Ptr1;
    float *p2 = (float *)Ptr2;

    while (Count > 0)
    {
        int res = memcmp(p1, p2, sizeof(float));
        if (res != 0) {
            if (fabs(*p1 - *p2) > 0.001) {
                return 1;
            }
        }
        p1++;
        p2++;
        Count--;
    }

    return 0;
}

int main(int argc, char *argv[])
{
    int L0, L1, L2, L3;
    int VOLUME;

    su3 *u;
    pauli *m;
    spinor *s, *r, *rfinal;
    int *piup, *pidn;

    char cnfg[256];
    char cnfg_dir[128];

    // Parse command line arguments
    L0 = atoi(argv[1]);
    L1 = atoi(argv[2]);
    L2 = atoi(argv[3]);
    L3 = atoi(argv[4]);
    VOLUME = L0 * L1 * L2 * L3;
    sprintf(cnfg_dir, argv[5]);

    // Allocate memory
    u = (su3 *) malloc((4 * VOLUME) * sizeof(*u));
    m = (pauli *) malloc((2 * VOLUME) * sizeof(*m));
    piup = (int *) malloc((2 * VOLUME) * sizeof(*piup));
    pidn = (int *) malloc((2 * VOLUME) * sizeof(*pidn));
    s = (spinor *) malloc(VOLUME * sizeof(*s));
    r = (spinor *) malloc(VOLUME * sizeof(*r));
    rfinal = (spinor *) malloc(VOLUME * sizeof(*rfinal));

    // Read data from disk
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

    sprintf(cnfg, "%s/sp-r-%d-%d-%d-%d", cnfg_dir, L0, L1, L2, L3);
    read_sp_spinor_from_file(cnfg, rfinal, VOLUME);


    // Call CUDA version of Dw() with Arrays of Structures
    Dw_cuda_AoS(VOLUME, u, s, r, m, piup, pidn);

    // Compare spinors r
    int ret;
    int count = 0;
    for (int i = 0; i < VOLUME; ++i) {
        ret = my_memcmp(r+i, rfinal+i, sizeof(spinor)/sizeof(float));
        if (ret == 0) {
            count++;
        }
        else {
            printf("Values in spinor r are incorrect at: %d\n", i);
        }
    }
    if (count == VOLUME) {
        printf("Values in spinor r are correct\n");
    }

    return 0;
}
