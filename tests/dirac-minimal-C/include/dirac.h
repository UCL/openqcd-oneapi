#ifndef DIRAC_H
#define DIRAC_H

#ifndef SU3_H
#include "su3.h"
#endif

#include "pauli_math.h"

extern void Dw(float mu, spinor *s,spinor *r, su3* u, pauli *m, int * piup, int * pidn, int vol);

#endif
