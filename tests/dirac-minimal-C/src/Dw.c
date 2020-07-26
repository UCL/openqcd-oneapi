
/*******************************************************************************

*
* File Dw.c
*
* Felix Ziegler (2020) for FASTSUM, GPU Hackathon
*
*
* based on:
*
*******************************************************************************/


/*******************************************************************************

*
* File Dw.c
*
* Copyright (C) 2005, 2011-2013, 2016 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Application of the O(a)-improved Wilson-Dirac operator D (single-
* precision programs).
*
* The externally accessible functions are
*
*   void Dw(float mu,spinor *s,spinor *r)
*     Depending on whether the twisted-mass flag is set or not, this
*     program applies D+i*mu*gamma_5*1e or D+i*mu*gamma_5 to the field
*     s and assigns the result to the field r.
*
* Notes:
*
* The notation and normalization conventions are specified in the notes
* "Implementation of the lattice Dirac operator" (file doc/dirac.pdf).
*
* In all these programs, it is assumed that the SW term is in the proper
* condition and that the global spinor fields have NSPIN elements. The
* programs check whether the twisted-mass flag (see flags/lat_parms.c) is
* set and turn off the twisted-mass term on the odd lattice sites if it is.
* The input and output fields may not coincide in the case of the programs
* Dw(), Dwhat(), Dw_blk() and Dwhat_blk().
*
* When the input and output fields are different, the input field is not
* changed except possibly at the points at global time 0 and NPROC0*L0-1,
* where both fields are set to zero if so required by the chosen boundary
* conditions. Depending on the operator considered, the fields are zeroed
* only on the even or odd points at these times.
*
* The programs Dw(),..,Dwhat() perform global operations and must be called
* simultaneously on all processes.
*
*******************************************************************************/

#define DW_C

#include "dirac.h"

typedef union
{
   spinor s;
   weyl w[2];
} spin_t;

static float coe,ceo;
static spin_t rs;

#define _vector_mul_assign(r,c) \
   (r).c1.re*=(c); \
   (r).c1.im*=(c); \
   (r).c2.re*=(c); \
   (r).c2.im*=(c); \
   (r).c3.re*=(c); \
   (r).c3.im*=(c)


static void doe(int *piup,int *pidn,su3 *u,spinor *pk)
{
   spinor *sp,*sm;
   su3_vector psi,chi;

/******************************* direction +0 *********************************/

   sp=pk+(*(piup++));

   _vector_add(psi,(*sp).c1,(*sp).c3);
   _su3_multiply(rs.s.c1,*u,psi);
   rs.s.c3=rs.s.c1;

   _vector_add(psi,(*sp).c2,(*sp).c4);
   _su3_multiply(rs.s.c2,*u,psi);
   rs.s.c4=rs.s.c2;

/******************************* direction -0 *********************************/

   sm=pk+(*(pidn++));
   u+=1;

   _vector_sub(psi,(*sm).c1,(*sm).c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c1,chi);
   _vector_sub_assign(rs.s.c3,chi);

   _vector_sub(psi,(*sm).c2,(*sm).c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c2,chi);
   _vector_sub_assign(rs.s.c4,chi);

/******************************* direction +1 *********************************/

   sp=pk+(*(piup++));
   u+=1;

   _vector_i_add(psi,(*sp).c1,(*sp).c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c1,chi);
   _vector_i_sub_assign(rs.s.c4,chi);

   _vector_i_add(psi,(*sp).c2,(*sp).c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c2,chi);
   _vector_i_sub_assign(rs.s.c3,chi);

/******************************* direction -1 *********************************/

   sm=pk+(*(pidn++));
   u+=1;

   _vector_i_sub(psi,(*sm).c1,(*sm).c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c1,chi);
   _vector_i_add_assign(rs.s.c4,chi);

   _vector_i_sub(psi,(*sm).c2,(*sm).c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c2,chi);
   _vector_i_add_assign(rs.s.c3,chi);

/******************************* direction +2 *********************************/

   sp=pk+(*(piup++));
   u+=1;

   _vector_add(psi,(*sp).c1,(*sp).c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c1,chi);
   _vector_add_assign(rs.s.c4,chi);

   _vector_sub(psi,(*sp).c2,(*sp).c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c2,chi);
   _vector_sub_assign(rs.s.c3,chi);

/******************************* direction -2 *********************************/

   sm=pk+(*(pidn++));
   u+=1;

   _vector_sub(psi,(*sm).c1,(*sm).c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c1,chi);
   _vector_sub_assign(rs.s.c4,chi);

   _vector_add(psi,(*sm).c2,(*sm).c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c2,chi);
   _vector_add_assign(rs.s.c3,chi);

/******************************* direction +3 *********************************/

   sp=pk+(*(piup));
   u+=1;

   _vector_i_add(psi,(*sp).c1,(*sp).c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c1,chi);
   _vector_i_sub_assign(rs.s.c3,chi);

   _vector_i_sub(psi,(*sp).c2,(*sp).c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c2,chi);
   _vector_i_add_assign(rs.s.c4,chi);

/******************************* direction -3 *********************************/

   sm=pk+(*(pidn));
   u+=1;

   _vector_i_sub(psi,(*sm).c1,(*sm).c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c1,chi);
   _vector_i_add_assign(rs.s.c3,chi);

   _vector_i_add(psi,(*sm).c2,(*sm).c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c2,chi);
   _vector_i_sub_assign(rs.s.c4,chi);

   _vector_mul_assign(rs.s.c1,coe);
   _vector_mul_assign(rs.s.c2,coe);
   _vector_mul_assign(rs.s.c3,coe);
   _vector_mul_assign(rs.s.c4,coe);
}


static void deo(int *piup,int *pidn,su3 *u,spinor *pl)
{
   spinor *sp,*sm;
   su3_vector psi,chi;

   _vector_mul_assign(rs.s.c1,ceo);
   _vector_mul_assign(rs.s.c2,ceo);
   _vector_mul_assign(rs.s.c3,ceo);
   _vector_mul_assign(rs.s.c4,ceo);

/******************************* direction +0 *********************************/

   sp=pl+(*(piup++));

   _vector_sub(psi,rs.s.c1,rs.s.c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c1,chi);
   _vector_sub_assign((*sp).c3,chi);

   _vector_sub(psi,rs.s.c2,rs.s.c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c2,chi);
   _vector_sub_assign((*sp).c4,chi);

/******************************* direction -0 *********************************/

   sm=pl+(*(pidn++));
   u+=1;

   _vector_add(psi,rs.s.c1,rs.s.c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c1,chi);
   _vector_add_assign((*sm).c3,chi);

   _vector_add(psi,rs.s.c2,rs.s.c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c2,chi);
   _vector_add_assign((*sm).c4,chi);

/******************************* direction +1 *********************************/

   sp=pl+(*(piup++));
   u+=1;

   _vector_i_sub(psi,rs.s.c1,rs.s.c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c1,chi);
   _vector_i_add_assign((*sp).c4,chi);

   _vector_i_sub(psi,rs.s.c2,rs.s.c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c2,chi);
   _vector_i_add_assign((*sp).c3,chi);

/******************************* direction -1 *********************************/

   sm=pl+(*(pidn++));
   u+=1;

   _vector_i_add(psi,rs.s.c1,rs.s.c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c1,chi);
   _vector_i_sub_assign((*sm).c4,chi);

   _vector_i_add(psi,rs.s.c2,rs.s.c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c2,chi);
   _vector_i_sub_assign((*sm).c3,chi);

/******************************* direction +2 *********************************/

   sp=pl+(*(piup++));
   u+=1;

   _vector_sub(psi,rs.s.c1,rs.s.c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c1,chi);
   _vector_sub_assign((*sp).c4,chi);

   _vector_add(psi,rs.s.c2,rs.s.c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c2,chi);
   _vector_add_assign((*sp).c3,chi);

/******************************* direction -2 *********************************/

   sm=pl+(*(pidn++));
   u+=1;

   _vector_add(psi,rs.s.c1,rs.s.c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c1,chi);
   _vector_add_assign((*sm).c4,chi);

   _vector_sub(psi,rs.s.c2,rs.s.c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c2,chi);
   _vector_sub_assign((*sm).c3,chi);

/******************************* direction +3 *********************************/

   sp=pl+(*(piup));
   u+=1;

   _vector_i_sub(psi,rs.s.c1,rs.s.c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c1,chi);
   _vector_i_add_assign((*sp).c3,chi);

   _vector_i_add(psi,rs.s.c2,rs.s.c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c2,chi);
   _vector_i_sub_assign((*sp).c4,chi);

/******************************* direction -3 *********************************/

   sm=pl+(*(pidn));
   u+=1;

   _vector_i_add(psi,rs.s.c1,rs.s.c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c1,chi);
   _vector_i_sub_assign((*sm).c3,chi);

   _vector_i_sub(psi,rs.s.c2,rs.s.c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c2,chi);
   _vector_i_add_assign((*sm).c4,chi);
}

void Dw(float mu, spinor *s, spinor *r, su3* u, pauli *m, int * piup, int * pidn, int vol)
{

   su3 *um;
   spin_t *so,*ro;

   apply_sw(vol / 2, mu, m, s, r);

   coe=-0.5f;
   ceo=-0.5f;

   so=(spin_t*)(s+(vol / 2));
   ro=(spin_t*)(r+(vol / 2));

   m += vol;

   um = u + 4 * vol;

   for (; u < um; u += 8, piup+=4, pidn+=4, so+=1, ro+=1, m+=2)
   {

      doe(piup,pidn,u,s);

      mul_pauli2(mu, m, &((*so).s), &((*ro).s));

      _vector_add_assign((*ro).s.c1,rs.s.c1);
      _vector_add_assign((*ro).s.c2,rs.s.c2);
      _vector_add_assign((*ro).s.c3,rs.s.c3);
      _vector_add_assign((*ro).s.c4,rs.s.c4);

      rs=(*so);

      deo(piup,pidn,u,r);

   }
}
