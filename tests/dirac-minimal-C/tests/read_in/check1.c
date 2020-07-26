/***************************************************************************
 *
 * Felix Ziegler (2020) for FASTSUM
 * Data checks
 *
 *
 *
 ***************************************************************************/

#define MAIN_PROGRAM

#include "read_in.h"
#include "dirac.h"

int main(int argc, char** argv)
{

  int L0, L1, L2, L3;
  int VOLUME, ix;

  su3 *u;
  pauli *m;
  spinor *s, *r, *rtest;
  int *piup, *pidn;

  char cnfg[256];
  char cnfg_dir[128];

  FILE * flog;

  int devcount;
  float dev;
  float *fpr, *fprtest;

  flog = freopen("check1.log", "w", stdout);

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
  rtest = (spinor *) malloc(VOLUME * sizeof(*rtest));

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
  read_sp_spinor_from_file(cnfg, r, VOLUME);

  /*for(ix = 0; ix < VOLUME; ix++, s++)
  {
    printf("ix = %d, s.c1.c1.re = %.8f\n", ix, (*s).c1.c1.re);
  }
  */

  /***************************************************************************
   *
   * Apply the Dirac operator followed by result comparison
   *
   **************************************************************************/

   Dw(0.0f, s, rtest, u, m, piup, pidn, VOLUME);

   devcount = 0;
   fpr = (float*) r;
   fprtest = (float*) rtest;

   for(ix = 0; ix < 24 * VOLUME; ix++, fpr++, fprtest++)
   {

     dev = fabs((*fpr) - (*fprtest));
     if(dev > 5.0e-7)
     {
      devcount++;
       printf("dev = %.8e\n", dev);
     }

   }

   printf("devcount = %d out of %d\n", devcount, 24 * VOLUME);

  /* Clean up */

  fclose(flog);
  /*free(s);*/

  return 0;
}
