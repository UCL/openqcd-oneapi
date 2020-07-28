#ifndef DIRAC_H
#define DIRAC_H

#ifndef SU3_H
#include "su3.h"
#endif

#include "pauli_math.h"

extern void Dw(float mu, spinor *s,spinor *r, su3* u, pauli *m, int * piup, int * pidn, int vol);
extern void Dw_diag(float mu, spinor *s, spinor *r, pauli *m, int vol);
extern void Dw_upto_doe(float mu, spinor *s, spinor *r, su3* u, pauli *m, int * piup, int * pidn, int vol);

#endif
