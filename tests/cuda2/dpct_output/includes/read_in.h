#ifndef READ_IN_H
#define READ_IN_H

#include "su3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

extern int read_sp_u_from_file(char *cnfg, su3 *u, int vol);
extern int read_sp_m_from_file(char *cnfg, pauli *m, int vol);
extern int read_sp_spinor_from_file(char *cnfg, spinor *s, int vol);
extern int read_lt_from_file(char *cnfg, int *piud, int vol);

#endif
