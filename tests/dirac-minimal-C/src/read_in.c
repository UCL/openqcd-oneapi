#include "read_in.h"

void read_sp_u_from_file(char * cnfg, su3 * u, int vol)
{

  FILE * fin;
  int ix;

  fin = fopen(cnfg, "rb");
  for(ix = 0; ix < 4 * vol; ix++, u+=1)
  {
    fread((float*) u, sizeof(float), 18, fin);
  }

  fclose(fin);

  /* u -= (4 * vol); */

}

void read_sp_m_from_file(char * cnfg, pauli * m, int vol)
{

  FILE * fin;
  int ix;

  fin = fopen(cnfg, "rb");
  for(ix = 0; ix < 2 * vol; ix++, m+=1)
  {
    fread((float*) m, sizeof(float), 36, fin);
  }

  fclose(fin);

  /* move pointer back */

  /*m -= (2 * vol);*/

}

void read_sp_spinor_from_file(char * cnfg, spinor * s, int vol)
{

  FILE * fin;
  int ix;

  fin = fopen(cnfg, "rb");
  for(ix = 0; ix < vol; ix++, s+=1)
  {
    fread((float*) s, sizeof(float), 24, fin);
  }

  fclose(fin);

  /* s -= vol; */

}

void read_lt_from_file(char * cnfg, int * piud, int vol)
{

  FILE * fin;
  int ix;

  fin = fopen(cnfg, "rb");
  for(ix = 0; ix < vol / 2; ix++, piud+=4)
  {
    fread(piud, sizeof(int), 4, fin);
  }

  fclose(fin);

  /* piud -= (2 * vol); */

}
