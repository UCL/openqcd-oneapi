/*******************************************************************************
 *
 * Felix Ziegler (2020)
 *
 ******************************************************************************/

#define TESTPROGRAM

/* Everything gets add_assigned into r */
/* All arguments are structures of arrays */

static void doe(int piup_soa, int pidn_soa, su3_soa u, spinor_soa s, spinor_soa r)
{

   spinor_loc_soa sp;
   su3_vector_loc psi, chi;

   /* define thread index idx */
   int idx = ...;

   int upidx = piup_soa[idx]

/******************************* direction +0 *********************************/
/* Note: the other directions work the same */


 /* load the stuff into registers / cache first */

   sp.c1.c1.re = s.c1.c1.re[upidx];
   sp.c1.c1.im = s.c1.c1.im[upidx];

   /* ... all 24 components ... */

   /* projection: vector_add */
   /* _vector_add(psi,(*sp).c1,(*sp).c3); */

   psi.c1.re = sp.c1.c1.re + sp.c3.c1.re;
   psi.c1.im = sp.c1.c1.re + sp.c3.c1.im;
   psi.c2.re = sp.c1.c2.re + sp.c3.c2.re;
   psi.c2.im = sp.c1.c2.re + sp.c3.c2.im;
   psi.c3.re = sp.c1.c3.re + sp.c3.c3.re;
   psi.c3.im = sp.c1.c3.re + sp.c3.c3.im;

   /* multiply with link */
   /* _su3_multiply(rs.s.c1,*u,psi); */
   chi.c1.re = u.c11.re[idx] * psi.c1.re[idx] + ...;
   /* ... and so on */

   /* add_assignment of the result to r */
   r.c1.c1.re[idx] += chi.c1.re;
   r.c1.c1.im[idx] += chi.c1.im;
   /* ... */

   r.c3.c1.re[idx] += chi.c1.re;
   r.c3.c1.im[idx] += chi.c1.im;
   /* ... */

   /* Same for c2 and c4 spinor components*/

   /*
   _vector_add(psi,(*sp).c2,(*sp).c4);
   _su3_multiply(rs.s.c2,*u,psi);
   rs.s.c4=rs.s.c2;
  */

}
