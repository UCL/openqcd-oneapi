#ifndef DW_SYCL_SOA_H
#define DW_SYCL_SOA_H

#include "su3.h"

#ifdef __cplusplus
extern "C" {
#endif

void Dw_sycl_SoA(int VOLUME, su3 *u, spinor *s, spinor *r, pauli *m, int *piup, int *pidn);

#ifdef __cplusplus
}
#endif

#endif
