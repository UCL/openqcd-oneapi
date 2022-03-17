#ifndef DW_CUDA_SOA_H
#define DW_CUDA_SOA_H

#include "su3.h"

#ifdef __cplusplus
extern "C" {
#endif

// void Dw_cuda_SoA();
void Dw_cuda_SoA(int VOLUME, su3 *u, spinor *s, spinor *r, pauli *m, int *piup, int *pidn);

#ifdef __cplusplus
}
#endif

#endif
