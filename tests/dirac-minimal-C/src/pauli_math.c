/*******************************************************************************
*
* Felix Ziegler (2020)
*
* Pauli math in single precision
*
* Notes:

*  only needs types defined in su3.h
*
* based on:
*
*******************************************************************************/

/*******************************************************************************
*
* File pauli.c
*
* Copyright (C) 2005, 2009, 2011, 2013 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Basic functions for single-precision Hermitian 6x6 matrices.
*
* The externally accessible functions are
*
*   void mul_pauli(float mu,pauli *m,weyl *s,weyl *r)
*     Multiplies the Weyl spinor s by the matrix m+i*mu and assigns
*     the result to the Weyl spinor r. The source spinor is overwritten
*     if r=s and otherwise left unchanged.
*
*   void mul_pauli2(float mu,pauli *m,spinor *s,spinor *r)
*     Multiplies the spinor s by the matrix m+i*mu*gamma_5 and assigns
*     the result to the spinor r. The source spinor is overwritten
*     if r=s and otherwise left unchanged.
*
*   void assign_pauli(int vol,pauli_dble *md,pauli *m)
*     Assigns the field md[vol] of double-precision matrices to the field
*     m[vol] of single-precision matrices.
*
*   void apply_sw(int vol,float mu,pauli *m,spinor *s,spinor *r)
*     Applies the matrix field m[2*vol]+i*mu*gamma_5 to the spinor field
*     s[vol] and assigns the result to the field r[vol]. The source field
*     is overwritten if r=s and otherwise left unchanged (the arrays may
*     not overlap in this case).
*
* Notes:
*
* The storage format for Hermitian 6x6 matrices is described in the notes
* "Implementation of the lattice Dirac operator" (file doc/dirac.pdf).
*
* The programs perform no communications and can be called locally. If SSE
* or AVX instructions are used, the Pauli matrices, Weyl and Dirac spinors
* must be aligned to a 16 byte boundary.
*
*******************************************************************************/

#define PAULI_MATH_C

#include "pauli_math.h"

typedef union
{
   spinor s;
   weyl w[2];
} spin_t;

static weyl rs;

void mul_pauli(float mu,pauli *m,weyl *s,weyl *r)
{
   float *u;

   u=(*m).u;

   rs.c1.c1.re=
      u[ 0]*(*s).c1.c1.re-   mu*(*s).c1.c1.im+
      u[ 6]*(*s).c1.c2.re-u[ 7]*(*s).c1.c2.im+
      u[ 8]*(*s).c1.c3.re-u[ 9]*(*s).c1.c3.im+
      u[10]*(*s).c2.c1.re-u[11]*(*s).c2.c1.im+
      u[12]*(*s).c2.c2.re-u[13]*(*s).c2.c2.im+
      u[14]*(*s).c2.c3.re-u[15]*(*s).c2.c3.im;

   rs.c1.c1.im=
      u[ 0]*(*s).c1.c1.im+   mu*(*s).c1.c1.re+
      u[ 6]*(*s).c1.c2.im+u[ 7]*(*s).c1.c2.re+
      u[ 8]*(*s).c1.c3.im+u[ 9]*(*s).c1.c3.re+
      u[10]*(*s).c2.c1.im+u[11]*(*s).c2.c1.re+
      u[12]*(*s).c2.c2.im+u[13]*(*s).c2.c2.re+
      u[14]*(*s).c2.c3.im+u[15]*(*s).c2.c3.re;

   rs.c1.c2.re=
      u[ 6]*(*s).c1.c1.re+u[ 7]*(*s).c1.c1.im+
      u[ 1]*(*s).c1.c2.re-   mu*(*s).c1.c2.im+
      u[16]*(*s).c1.c3.re-u[17]*(*s).c1.c3.im+
      u[18]*(*s).c2.c1.re-u[19]*(*s).c2.c1.im+
      u[20]*(*s).c2.c2.re-u[21]*(*s).c2.c2.im+
      u[22]*(*s).c2.c3.re-u[23]*(*s).c2.c3.im;

   rs.c1.c2.im=
      u[ 6]*(*s).c1.c1.im-u[ 7]*(*s).c1.c1.re+
      u[ 1]*(*s).c1.c2.im+   mu*(*s).c1.c2.re+
      u[16]*(*s).c1.c3.im+u[17]*(*s).c1.c3.re+
      u[18]*(*s).c2.c1.im+u[19]*(*s).c2.c1.re+
      u[20]*(*s).c2.c2.im+u[21]*(*s).c2.c2.re+
      u[22]*(*s).c2.c3.im+u[23]*(*s).c2.c3.re;

   rs.c1.c3.re=
      u[ 8]*(*s).c1.c1.re+u[ 9]*(*s).c1.c1.im+
      u[16]*(*s).c1.c2.re+u[17]*(*s).c1.c2.im+
      u[ 2]*(*s).c1.c3.re-   mu*(*s).c1.c3.im+
      u[24]*(*s).c2.c1.re-u[25]*(*s).c2.c1.im+
      u[26]*(*s).c2.c2.re-u[27]*(*s).c2.c2.im+
      u[28]*(*s).c2.c3.re-u[29]*(*s).c2.c3.im;

   rs.c1.c3.im=
      u[ 8]*(*s).c1.c1.im-u[ 9]*(*s).c1.c1.re+
      u[16]*(*s).c1.c2.im-u[17]*(*s).c1.c2.re+
      u[ 2]*(*s).c1.c3.im+   mu*(*s).c1.c3.re+
      u[24]*(*s).c2.c1.im+u[25]*(*s).c2.c1.re+
      u[26]*(*s).c2.c2.im+u[27]*(*s).c2.c2.re+
      u[28]*(*s).c2.c3.im+u[29]*(*s).c2.c3.re;

   rs.c2.c1.re=
      u[10]*(*s).c1.c1.re+u[11]*(*s).c1.c1.im+
      u[18]*(*s).c1.c2.re+u[19]*(*s).c1.c2.im+
      u[24]*(*s).c1.c3.re+u[25]*(*s).c1.c3.im+
      u[ 3]*(*s).c2.c1.re-   mu*(*s).c2.c1.im+
      u[30]*(*s).c2.c2.re-u[31]*(*s).c2.c2.im+
      u[32]*(*s).c2.c3.re-u[33]*(*s).c2.c3.im;

   rs.c2.c1.im=
      u[10]*(*s).c1.c1.im-u[11]*(*s).c1.c1.re+
      u[18]*(*s).c1.c2.im-u[19]*(*s).c1.c2.re+
      u[24]*(*s).c1.c3.im-u[25]*(*s).c1.c3.re+
      u[ 3]*(*s).c2.c1.im+   mu*(*s).c2.c1.re+
      u[30]*(*s).c2.c2.im+u[31]*(*s).c2.c2.re+
      u[32]*(*s).c2.c3.im+u[33]*(*s).c2.c3.re;

   rs.c2.c2.re=
      u[12]*(*s).c1.c1.re+u[13]*(*s).c1.c1.im+
      u[20]*(*s).c1.c2.re+u[21]*(*s).c1.c2.im+
      u[26]*(*s).c1.c3.re+u[27]*(*s).c1.c3.im+
      u[30]*(*s).c2.c1.re+u[31]*(*s).c2.c1.im+
      u[ 4]*(*s).c2.c2.re-   mu*(*s).c2.c2.im+
      u[34]*(*s).c2.c3.re-u[35]*(*s).c2.c3.im;

   rs.c2.c2.im=
      u[12]*(*s).c1.c1.im-u[13]*(*s).c1.c1.re+
      u[20]*(*s).c1.c2.im-u[21]*(*s).c1.c2.re+
      u[26]*(*s).c1.c3.im-u[27]*(*s).c1.c3.re+
      u[30]*(*s).c2.c1.im-u[31]*(*s).c2.c1.re+
      u[ 4]*(*s).c2.c2.im+   mu*(*s).c2.c2.re+
      u[34]*(*s).c2.c3.im+u[35]*(*s).c2.c3.re;

   rs.c2.c3.re=
      u[14]*(*s).c1.c1.re+u[15]*(*s).c1.c1.im+
      u[22]*(*s).c1.c2.re+u[23]*(*s).c1.c2.im+
      u[28]*(*s).c1.c3.re+u[29]*(*s).c1.c3.im+
      u[32]*(*s).c2.c1.re+u[33]*(*s).c2.c1.im+
      u[34]*(*s).c2.c2.re+u[35]*(*s).c2.c2.im+
      u[ 5]*(*s).c2.c3.re-   mu*(*s).c2.c3.im;

   rs.c2.c3.im=
      u[14]*(*s).c1.c1.im-u[15]*(*s).c1.c1.re+
      u[22]*(*s).c1.c2.im-u[23]*(*s).c1.c2.re+
      u[28]*(*s).c1.c3.im-u[29]*(*s).c1.c3.re+
      u[32]*(*s).c2.c1.im-u[33]*(*s).c2.c1.re+
      u[34]*(*s).c2.c2.im-u[35]*(*s).c2.c2.re+
      u[ 5]*(*s).c2.c3.im+   mu*(*s).c2.c3.re;

   (*r)=rs;
}


void mul_pauli2(float mu,pauli *m,spinor *s,spinor *r)
{
   spin_t *ps,*pr;

   ps=(spin_t*)(s);
   pr=(spin_t*)(r);

   mul_pauli(mu,m,(*ps).w,(*pr).w);
   mul_pauli(-mu,m+1,(*ps).w+1,(*pr).w+1);
}

void apply_sw(int vol,float mu,pauli *m,spinor *s,spinor *r)
{
   spinor *sm;

   sm=s+vol;

   for (;s<sm;s++)
   {
      mul_pauli2(mu,m,s,r);
      m+=2;
      r+=1;
   }
}
