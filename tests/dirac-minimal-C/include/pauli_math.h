#ifndef PAULI_MATH_H
#define PAULI_MATH_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifndef SU3_H
#include "su3.h"
#endif

extern void mul_pauli(float mu,pauli *m,weyl *s,weyl *r);

extern void mul_pauli2(float mu,pauli *m,spinor *s,spinor *r);

extern void apply_sw(int vol,float mu,pauli *m,spinor *s,spinor *r);

#endif
