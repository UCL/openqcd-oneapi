// #include "dw_cuda.h"
#include "su3.h"
#include "macros.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


long my_memcmp(const void *Ptr1, const void *Ptr2, size_t Count)
{
    float *p1 = (float *)Ptr1;
    float *p2 = (float *)Ptr2;

    while (Count > 0)
    {
        int res = memcmp(p1, p2, sizeof(float));
        if (res != 0) {
            if (fabs(*p1 - *p2) > 0.001) {
                return 1;
            }
        }
        p1++;
        p2++;
        Count--;
    }

    return 0;
}


__device__
spin_t doe_cuda(int *piup, int *pidn, su3 *u, spinor *pk,
                     float coe, float gamma_f, float one_over_gammaf, spin_t rs)
{
    spinor *sp, *sm;
    su3_vector psi, chi;

    /***************************** direction +0 *******************************/

    sp = pk + (*(piup++));

    _vector_add(psi, (*sp).c1, (*sp).c3);
    _su3_multiply(rs.s.c1, *u, psi);
    rs.s.c3 = rs.s.c1;

    _vector_add(psi, (*sp).c2, (*sp).c4);
    _su3_multiply(rs.s.c2, *u, psi);
    rs.s.c4 = rs.s.c2;

    /***************************** direction -0 *******************************/

    sm = pk + (*(pidn++));
    u += 1;

    _vector_sub(psi, (*sm).c1, (*sm).c3);
    _su3_inverse_multiply(chi, *u, psi);
    _vector_add_assign(rs.s.c1, chi);
    _vector_sub_assign(rs.s.c3, chi);

    _vector_sub(psi, (*sm).c2, (*sm).c4);
    _su3_inverse_multiply(chi, *u, psi);
    _vector_add_assign(rs.s.c2, chi);
    _vector_sub_assign(rs.s.c4, chi);

    _vector_mul_assign(rs.s.c1, gamma_f);
    _vector_mul_assign(rs.s.c2, gamma_f);
    _vector_mul_assign(rs.s.c3, gamma_f);
    _vector_mul_assign(rs.s.c4, gamma_f);

    /***************************** direction +1 *******************************/

    sp = pk + (*(piup++));
    u += 1;

    _vector_i_add(psi, (*sp).c1, (*sp).c4);
    _su3_multiply(chi, *u, psi);
    _vector_add_assign(rs.s.c1, chi);
    _vector_i_sub_assign(rs.s.c4, chi);

    _vector_i_add(psi, (*sp).c2, (*sp).c3);
    _su3_multiply(chi, *u, psi);
    _vector_add_assign(rs.s.c2, chi);
    _vector_i_sub_assign(rs.s.c3, chi);

    /***************************** direction -1 *******************************/

    sm = pk + (*(pidn++));
    u += 1;

    _vector_i_sub(psi, (*sm).c1, (*sm).c4);
    _su3_inverse_multiply(chi, *u, psi);
    _vector_add_assign(rs.s.c1, chi);
    _vector_i_add_assign(rs.s.c4, chi);

    _vector_i_sub(psi, (*sm).c2, (*sm).c3);
    _su3_inverse_multiply(chi, *u, psi);
    _vector_add_assign(rs.s.c2, chi);
    _vector_i_add_assign(rs.s.c3, chi);

    /***************************** direction +2 *******************************/

    sp = pk + (*(piup++));
    u += 1;

    _vector_add(psi, (*sp).c1, (*sp).c4);
    _su3_multiply(chi, *u, psi);
    _vector_add_assign(rs.s.c1, chi);
    _vector_add_assign(rs.s.c4, chi);

    _vector_sub(psi, (*sp).c2, (*sp).c3);
    _su3_multiply(chi, *u, psi);
    _vector_add_assign(rs.s.c2, chi);
    _vector_sub_assign(rs.s.c3, chi);

    /***************************** direction -2 *******************************/

    sm = pk + (*(pidn++));
    u += 1;

    _vector_sub(psi, (*sm).c1, (*sm).c4);
    _su3_inverse_multiply(chi, *u, psi);
    _vector_add_assign(rs.s.c1, chi);
    _vector_sub_assign(rs.s.c4, chi);

    _vector_add(psi, (*sm).c2, (*sm).c3);
    _su3_inverse_multiply(chi, *u, psi);
    _vector_add_assign(rs.s.c2, chi);
    _vector_add_assign(rs.s.c3, chi);

    /***************************** direction +3 *******************************/

    sp = pk + (*(piup));
    u += 1;

    _vector_i_add(psi, (*sp).c1, (*sp).c3);
    _su3_multiply(chi, *u, psi);
    _vector_add_assign(rs.s.c1, chi);
    _vector_i_sub_assign(rs.s.c3, chi);

    _vector_i_sub(psi, (*sp).c2, (*sp).c4);
    _su3_multiply(chi, *u, psi);
    _vector_add_assign(rs.s.c2, chi);
    _vector_i_add_assign(rs.s.c4, chi);

    /***************************** direction -3 *******************************/

    sm = pk + (*(pidn));
    u += 1;

    _vector_i_sub(psi, (*sm).c1, (*sm).c3);
    _su3_inverse_multiply(chi, *u, psi);
    _vector_add_assign(rs.s.c1, chi);
    _vector_i_add_assign(rs.s.c3, chi);

    _vector_i_add(psi, (*sm).c2, (*sm).c4);
    _su3_inverse_multiply(chi, *u, psi);
    _vector_add_assign(rs.s.c2, chi);
    _vector_i_sub_assign(rs.s.c4, chi);

    _vector_mul_assign(rs.s.c1, coe);
    _vector_mul_assign(rs.s.c2, coe);
    _vector_mul_assign(rs.s.c3, coe);
    _vector_mul_assign(rs.s.c4, coe);

    _vector_mul_assign(rs.s.c1, one_over_gammaf);
    _vector_mul_assign(rs.s.c2, one_over_gammaf);
    _vector_mul_assign(rs.s.c3, one_over_gammaf);
    _vector_mul_assign(rs.s.c4, one_over_gammaf);

    return rs;


}

__device__
static void deo_cuda(int *piup, int *pidn, su3 *u, spinor *pl,
                     float ceo, float one_over_gammaf, spin_t rs)
{
    spinor *sp, *sm;
    su3_vector psi, chi;

    _vector_mul_assign(rs.s.c1, ceo);
    _vector_mul_assign(rs.s.c2, ceo);
    _vector_mul_assign(rs.s.c3, ceo);
    _vector_mul_assign(rs.s.c4, ceo);

    /***************************** direction +0 *******************************/

    sp = pl + (*(piup++));

    _vector_sub(psi, rs.s.c1, rs.s.c3);
    _su3_inverse_multiply(chi, *u, psi);
    _vector_add_assign((*sp).c1, chi);
    _vector_sub_assign((*sp).c3, chi);

    _vector_sub(psi, rs.s.c2, rs.s.c4);
    _su3_inverse_multiply(chi, *u, psi);
    _vector_add_assign((*sp).c2, chi);
    _vector_sub_assign((*sp).c4, chi);

    /***************************** direction -0 *******************************/

    sm = pl + (*(pidn++));
    u += 1;

    _vector_add(psi, rs.s.c1, rs.s.c3);
    _su3_multiply(chi, *u, psi);
    _vector_add_assign((*sm).c1, chi);
    _vector_add_assign((*sm).c3, chi);

    _vector_add(psi, rs.s.c2, rs.s.c4);
    _su3_multiply(chi, *u, psi);
    _vector_add_assign((*sm).c2, chi);
    _vector_add_assign((*sm).c4, chi);

    /***************************** direction +1 *******************************/

    _vector_mul_assign(rs.s.c1, one_over_gammaf);
    _vector_mul_assign(rs.s.c2, one_over_gammaf);
    _vector_mul_assign(rs.s.c3, one_over_gammaf);
    _vector_mul_assign(rs.s.c4, one_over_gammaf);

    sp = pl + (*(piup++));
    u += 1;

    _vector_i_sub(psi, rs.s.c1, rs.s.c4);
    _su3_inverse_multiply(chi, *u, psi);
    _vector_add_assign((*sp).c1, chi);
    _vector_i_add_assign((*sp).c4, chi);

    _vector_i_sub(psi, rs.s.c2, rs.s.c3);
    _su3_inverse_multiply(chi, *u, psi);
    _vector_add_assign((*sp).c2, chi);
    _vector_i_add_assign((*sp).c3, chi);

    /***************************** direction -1 *******************************/

    sm = pl + (*(pidn++));
    u += 1;

    _vector_i_add(psi, rs.s.c1, rs.s.c4);
    _su3_multiply(chi, *u, psi);
    _vector_add_assign((*sm).c1, chi);
    _vector_i_sub_assign((*sm).c4, chi);

    _vector_i_add(psi, rs.s.c2, rs.s.c3);
    _su3_multiply(chi, *u, psi);
    _vector_add_assign((*sm).c2, chi);
    _vector_i_sub_assign((*sm).c3, chi);

    /***************************** direction +2 *******************************/

    sp = pl + (*(piup++));
    u += 1;

    _vector_sub(psi, rs.s.c1, rs.s.c4);
    _su3_inverse_multiply(chi, *u, psi);
    _vector_add_assign((*sp).c1, chi);
    _vector_sub_assign((*sp).c4, chi);

    _vector_add(psi, rs.s.c2, rs.s.c3);
    _su3_inverse_multiply(chi, *u, psi);
    _vector_add_assign((*sp).c2, chi);
    _vector_add_assign((*sp).c3, chi);

    /***************************** direction -2 *******************************/

    sm = pl + (*(pidn++));
    u += 1;

    _vector_add(psi, rs.s.c1, rs.s.c4);
    _su3_multiply(chi, *u, psi);
    _vector_add_assign((*sm).c1, chi);
    _vector_add_assign((*sm).c4, chi);

    _vector_sub(psi, rs.s.c2, rs.s.c3);
    _su3_multiply(chi, *u, psi);
    _vector_add_assign((*sm).c2, chi);
    _vector_sub_assign((*sm).c3, chi);

    /***************************** direction +3 *******************************/

    sp = pl + (*(piup));
    u += 1;

    _vector_i_sub(psi, rs.s.c1, rs.s.c3);
    _su3_inverse_multiply(chi, *u, psi);
    _vector_add_assign((*sp).c1, chi);
    _vector_i_add_assign((*sp).c3, chi);

    _vector_i_add(psi, rs.s.c2, rs.s.c4);
    _su3_inverse_multiply(chi, *u, psi);
    _vector_add_assign((*sp).c2, chi);
    _vector_i_sub_assign((*sp).c4, chi);

    /***************************** direction -3 *******************************/

    sm = pl + (*(pidn));
    u += 1;

    _vector_i_add(psi, rs.s.c1, rs.s.c3);
    _su3_multiply(chi, *u, psi);
    _vector_add_assign((*sm).c1, chi);
    _vector_i_sub_assign((*sm).c3, chi);

    _vector_i_sub(psi, rs.s.c2, rs.s.c4);
    _su3_multiply(chi, *u, psi);
    _vector_add_assign((*sm).c2, chi);
    _vector_i_add_assign((*sm).c4, chi);
}

__device__
void vector_add_assign(su3_vector *r, su3_vector s)
{
    atomicAdd(&((*r).c1.re),  s.c1.re);
    atomicAdd(&((*r).c1.im),  s.c1.im);
    atomicAdd(&((*r).c2.re),  s.c2.re);
    atomicAdd(&((*r).c2.im),  s.c2.im);
    atomicAdd(&((*r).c3.re),  s.c3.re);
    atomicAdd(&((*r).c3.im),  s.c3.im);
    // (r).c1.re += (s).c1.re;
    // (r).c1.im += (s).c1.im;
    // (r).c2.re += (s).c2.re;
    // (r).c2.im += (s).c2.im;
    // (r).c3.re += (s).c3.re;
    // (r).c3.im += (s).c3.im
}

__device__
void vector_sub_assign(su3_vector *r, su3_vector s)
{
    atomicAdd(&((*r).c1.re), -s.c1.re);
    atomicAdd(&((*r).c1.im), -s.c1.im);
    atomicAdd(&((*r).c2.re), -s.c2.re);
    atomicAdd(&((*r).c2.im), -s.c2.im);
    atomicAdd(&((*r).c3.re), -s.c3.re);
    atomicAdd(&((*r).c3.im), -s.c3.im);
    // (r).c1.re -= (s).c1.re;
    // (r).c1.im -= (s).c1.im;
    // (r).c2.re -= (s).c2.re;
    // (r).c2.im -= (s).c2.im;
    // (r).c3.re -= (s).c3.re;
    // (r).c3.im -= (s).c3.im
}

__device__
void vector_i_add_assign(su3_vector *r, su3_vector s)
{
    atomicAdd(&((*r).c1.re), -s.c1.im);
    atomicAdd(&((*r).c1.im),  s.c1.re);
    atomicAdd(&((*r).c2.re), -s.c2.im);
    atomicAdd(&((*r).c2.im),  s.c2.re);
    atomicAdd(&((*r).c3.re), -s.c3.im);
    atomicAdd(&((*r).c3.im),  s.c3.re);
    // (r).c1.re -= (s).c1.im;
    // (r).c1.im += (s).c1.re;
    // (r).c2.re -= (s).c2.im;
    // (r).c2.im += (s).c2.re;
    // (r).c3.re -= (s).c3.im;
    // (r).c3.im += (s).c3.re
}

__device__
void vector_i_sub_assign(su3_vector *r, su3_vector s)
{
    atomicAdd(&((*r).c1.re),  s.c1.im);
    atomicAdd(&((*r).c1.im), -s.c1.re);
    atomicAdd(&((*r).c2.re),  s.c2.im);
    atomicAdd(&((*r).c2.im), -s.c2.re);
    atomicAdd(&((*r).c3.re),  s.c3.im);
    atomicAdd(&((*r).c3.im), -s.c3.re);
    // (r).c1.re += (s).c1.im;
    // (r).c1.im -= (s).c1.re;
    // (r).c2.re += (s).c2.im;
    // (r).c2.im -= (s).c2.re;
    // (r).c3.re += (s).c3.im;
    // (r).c3.im -= (s).c3.re
}


__device__
static void deo_cuda_atomics(int *piup, int *pidn, su3 *u, spinor *pl,
                             float ceo, float one_over_gammaf, spin_t rs)
{
    spinor *sp, *sm;
    su3_vector psi, chi;

    _vector_mul_assign(rs.s.c1, ceo);
    _vector_mul_assign(rs.s.c2, ceo);
    _vector_mul_assign(rs.s.c3, ceo);
    _vector_mul_assign(rs.s.c4, ceo);

    /***************************** direction +0 *******************************/

    sp = pl + (*(piup++));

    _vector_sub(psi, rs.s.c1, rs.s.c3);
    _su3_inverse_multiply(chi, *u, psi);
    vector_add_assign(&((*sp).c1), chi);
    vector_sub_assign(&((*sp).c3), chi);

    _vector_sub(psi, rs.s.c2, rs.s.c4);
    _su3_inverse_multiply(chi, *u, psi);
    vector_add_assign(&((*sp).c2), chi);
    vector_sub_assign(&((*sp).c4), chi);

    /***************************** direction -0 *******************************/

    sm = pl + (*(pidn++));
    u += 1;

    _vector_add(psi, rs.s.c1, rs.s.c3);
    _su3_multiply(chi, *u, psi);
    vector_add_assign(&((*sm).c1), chi);
    vector_add_assign(&((*sm).c3), chi);

    _vector_add(psi, rs.s.c2, rs.s.c4);
    _su3_multiply(chi, *u, psi);
    vector_add_assign(&((*sm).c2), chi);
    vector_add_assign(&((*sm).c4), chi);

    /***************************** direction +1 *******************************/

    _vector_mul_assign(rs.s.c1, one_over_gammaf);
    _vector_mul_assign(rs.s.c2, one_over_gammaf);
    _vector_mul_assign(rs.s.c3, one_over_gammaf);
    _vector_mul_assign(rs.s.c4, one_over_gammaf);

    sp = pl + (*(piup++));
    u += 1;

    _vector_i_sub(psi, rs.s.c1, rs.s.c4);
    _su3_inverse_multiply(chi, *u, psi);
    vector_add_assign(&((*sp).c1), chi);
    vector_i_add_assign(&((*sp).c4), chi);

    _vector_i_sub(psi, rs.s.c2, rs.s.c3);
    _su3_inverse_multiply(chi, *u, psi);
    vector_add_assign(&((*sp).c2), chi);
    vector_i_add_assign(&((*sp).c3), chi);

    /***************************** direction -1 *******************************/

    sm = pl + (*(pidn++));
    u += 1;

    _vector_i_add(psi, rs.s.c1, rs.s.c4);
    _su3_multiply(chi, *u, psi);
    vector_add_assign(&((*sm).c1), chi);
    vector_i_sub_assign(&((*sm).c4), chi);

    _vector_i_add(psi, rs.s.c2, rs.s.c3);
    _su3_multiply(chi, *u, psi);
    vector_add_assign(&((*sm).c2), chi);
    vector_i_sub_assign(&((*sm).c3), chi);

    /***************************** direction +2 *******************************/

    sp = pl + (*(piup++));
    u += 1;

    _vector_sub(psi, rs.s.c1, rs.s.c4);
    _su3_inverse_multiply(chi, *u, psi);
    vector_add_assign(&((*sp).c1), chi);
    vector_sub_assign(&((*sp).c4), chi);

    _vector_add(psi, rs.s.c2, rs.s.c3);
    _su3_inverse_multiply(chi, *u, psi);
    vector_add_assign(&((*sp).c2), chi);
    vector_add_assign(&((*sp).c3), chi);

    /***************************** direction -2 *******************************/

    sm = pl + (*(pidn++));
    u += 1;

    _vector_add(psi, rs.s.c1, rs.s.c4);
    _su3_multiply(chi, *u, psi);
    vector_add_assign(&((*sm).c1), chi);
    vector_add_assign(&((*sm).c4), chi);

    _vector_sub(psi, rs.s.c2, rs.s.c3);
    _su3_multiply(chi, *u, psi);
    vector_add_assign(&((*sm).c2), chi);
    vector_sub_assign(&((*sm).c3), chi);

    /***************************** direction +3 *******************************/

    sp = pl + (*(piup));
    u += 1;

    _vector_i_sub(psi, rs.s.c1, rs.s.c3);
    _su3_inverse_multiply(chi, *u, psi);
    vector_add_assign(&((*sp).c1), chi);
    vector_i_add_assign(&((*sp).c3), chi);

    _vector_i_add(psi, rs.s.c2, rs.s.c4);
    _su3_inverse_multiply(chi, *u, psi);
    vector_add_assign(&((*sp).c2), chi);
    vector_i_sub_assign(&((*sp).c4), chi);

    /***************************** direction -3 *******************************/

    sm = pl + (*(pidn));
    u += 1;

    _vector_i_add(psi, rs.s.c1, rs.s.c3);
    _su3_multiply(chi, *u, psi);
    vector_add_assign(&((*sm).c1), chi);
    vector_i_sub_assign(&((*sm).c3), chi);

    _vector_i_sub(psi, rs.s.c2, rs.s.c4);
    _su3_multiply(chi, *u, psi);
    vector_add_assign(&((*sm).c2), chi);
    vector_i_add_assign(&((*sm).c4), chi);
}


__device__
void mul_pauli_cuda(float mu, pauli const *m, weyl const *s, weyl *r, weyl rt)
{
    float const *u;

    u = (*m).u;

    rt.c1.c1.re =
      u[0] * (*s).c1.c1.re - mu * (*s).c1.c1.im + u[6] * (*s).c1.c2.re -
      u[7] * (*s).c1.c2.im + u[8] * (*s).c1.c3.re - u[9] * (*s).c1.c3.im +
      u[10] * (*s).c2.c1.re - u[11] * (*s).c2.c1.im + u[12] * (*s).c2.c2.re -
      u[13] * (*s).c2.c2.im + u[14] * (*s).c2.c3.re - u[15] * (*s).c2.c3.im;

    rt.c1.c1.im =
      u[0] * (*s).c1.c1.im + mu * (*s).c1.c1.re + u[6] * (*s).c1.c2.im +
      u[7] * (*s).c1.c2.re + u[8] * (*s).c1.c3.im + u[9] * (*s).c1.c3.re +
      u[10] * (*s).c2.c1.im + u[11] * (*s).c2.c1.re + u[12] * (*s).c2.c2.im +
      u[13] * (*s).c2.c2.re + u[14] * (*s).c2.c3.im + u[15] * (*s).c2.c3.re;

    rt.c1.c2.re =
      u[6] * (*s).c1.c1.re + u[7] * (*s).c1.c1.im + u[1] * (*s).c1.c2.re -
      mu * (*s).c1.c2.im + u[16] * (*s).c1.c3.re - u[17] * (*s).c1.c3.im +
      u[18] * (*s).c2.c1.re - u[19] * (*s).c2.c1.im + u[20] * (*s).c2.c2.re -
      u[21] * (*s).c2.c2.im + u[22] * (*s).c2.c3.re - u[23] * (*s).c2.c3.im;

    rt.c1.c2.im =
      u[6] * (*s).c1.c1.im - u[7] * (*s).c1.c1.re + u[1] * (*s).c1.c2.im +
      mu * (*s).c1.c2.re + u[16] * (*s).c1.c3.im + u[17] * (*s).c1.c3.re +
      u[18] * (*s).c2.c1.im + u[19] * (*s).c2.c1.re + u[20] * (*s).c2.c2.im +
      u[21] * (*s).c2.c2.re + u[22] * (*s).c2.c3.im + u[23] * (*s).c2.c3.re;

    rt.c1.c3.re =
      u[8] * (*s).c1.c1.re + u[9] * (*s).c1.c1.im + u[16] * (*s).c1.c2.re +
      u[17] * (*s).c1.c2.im + u[2] * (*s).c1.c3.re - mu * (*s).c1.c3.im +
      u[24] * (*s).c2.c1.re - u[25] * (*s).c2.c1.im + u[26] * (*s).c2.c2.re -
      u[27] * (*s).c2.c2.im + u[28] * (*s).c2.c3.re - u[29] * (*s).c2.c3.im;

    rt.c1.c3.im =
      u[8] * (*s).c1.c1.im - u[9] * (*s).c1.c1.re + u[16] * (*s).c1.c2.im -
      u[17] * (*s).c1.c2.re + u[2] * (*s).c1.c3.im + mu * (*s).c1.c3.re +
      u[24] * (*s).c2.c1.im + u[25] * (*s).c2.c1.re + u[26] * (*s).c2.c2.im +
      u[27] * (*s).c2.c2.re + u[28] * (*s).c2.c3.im + u[29] * (*s).c2.c3.re;

    rt.c2.c1.re =
      u[10] * (*s).c1.c1.re + u[11] * (*s).c1.c1.im + u[18] * (*s).c1.c2.re +
      u[19] * (*s).c1.c2.im + u[24] * (*s).c1.c3.re + u[25] * (*s).c1.c3.im +
      u[3] * (*s).c2.c1.re - mu * (*s).c2.c1.im + u[30] * (*s).c2.c2.re -
      u[31] * (*s).c2.c2.im + u[32] * (*s).c2.c3.re - u[33] * (*s).c2.c3.im;

    rt.c2.c1.im =
      u[10] * (*s).c1.c1.im - u[11] * (*s).c1.c1.re + u[18] * (*s).c1.c2.im -
      u[19] * (*s).c1.c2.re + u[24] * (*s).c1.c3.im - u[25] * (*s).c1.c3.re +
      u[3] * (*s).c2.c1.im + mu * (*s).c2.c1.re + u[30] * (*s).c2.c2.im +
      u[31] * (*s).c2.c2.re + u[32] * (*s).c2.c3.im + u[33] * (*s).c2.c3.re;

    rt.c2.c2.re =
      u[12] * (*s).c1.c1.re + u[13] * (*s).c1.c1.im + u[20] * (*s).c1.c2.re +
      u[21] * (*s).c1.c2.im + u[26] * (*s).c1.c3.re + u[27] * (*s).c1.c3.im +
      u[30] * (*s).c2.c1.re + u[31] * (*s).c2.c1.im + u[4] * (*s).c2.c2.re -
      mu * (*s).c2.c2.im + u[34] * (*s).c2.c3.re - u[35] * (*s).c2.c3.im;

    rt.c2.c2.im =
      u[12] * (*s).c1.c1.im - u[13] * (*s).c1.c1.re + u[20] * (*s).c1.c2.im -
      u[21] * (*s).c1.c2.re + u[26] * (*s).c1.c3.im - u[27] * (*s).c1.c3.re +
      u[30] * (*s).c2.c1.im - u[31] * (*s).c2.c1.re + u[4] * (*s).c2.c2.im +
      mu * (*s).c2.c2.re + u[34] * (*s).c2.c3.im + u[35] * (*s).c2.c3.re;

    rt.c2.c3.re =
      u[14] * (*s).c1.c1.re + u[15] * (*s).c1.c1.im + u[22] * (*s).c1.c2.re +
      u[23] * (*s).c1.c2.im + u[28] * (*s).c1.c3.re + u[29] * (*s).c1.c3.im +
      u[32] * (*s).c2.c1.re + u[33] * (*s).c2.c1.im + u[34] * (*s).c2.c2.re +
      u[35] * (*s).c2.c2.im + u[5] * (*s).c2.c3.re - mu * (*s).c2.c3.im;

    rt.c2.c3.im =
      u[14] * (*s).c1.c1.im - u[15] * (*s).c1.c1.re + u[22] * (*s).c1.c2.im -
      u[23] * (*s).c1.c2.re + u[28] * (*s).c1.c3.im - u[29] * (*s).c1.c3.re +
      u[32] * (*s).c2.c1.im - u[33] * (*s).c2.c1.re + u[34] * (*s).c2.c2.im -
      u[35] * (*s).c2.c2.re + u[5] * (*s).c2.c3.im + mu * (*s).c2.c3.re;

    (*r) = rt;
}

__device__
void mul_pauli2_cuda(float mu, pauli const *m, spinor const *s, spinor *r, weyl rt)
{
    spin_t const *ps;
    spin_t *pr;

    ps = (spin_t const *)(s);
    pr = (spin_t *)(r);

    mul_pauli_cuda(mu, m, (*ps).w, (*pr).w, rt);
    mul_pauli_cuda(-mu, m + 1, (*ps).w + 1, (*pr).w + 1, rt);
}

extern "C" __global__
void Dw_cuda_kernel(int VOLUME, float mu, spinor *s, spinor *r,
                    int *piup, int *pidn, su3 *u, pauli *m,
                    float coe, float ceo, float gamma_f, float one_over_gammaf)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= VOLUME/2) return;

    weyl rt;
    spin_t rs;

    spin_t *so, *ro;
    so = (spin_t *)(s + (VOLUME / 2));
    ro = (spin_t *)(r + (VOLUME / 2));

    u += (idx)*8;
    piup += (idx)*4;
    pidn += (idx)*4;
    so += (idx)*1;
    ro += (idx)*1;
    m += (idx)*2;

    rs = doe_cuda(piup, pidn, u, s, coe, gamma_f, one_over_gammaf, rs);

    mul_pauli2_cuda(mu, m, &((*so).s), &((*ro).s), rt);

    _vector_add_assign((*ro).s.c1, rs.s.c1);
    _vector_add_assign((*ro).s.c2, rs.s.c2);
    _vector_add_assign((*ro).s.c3, rs.s.c3);
    _vector_add_assign((*ro).s.c4, rs.s.c4);

    // __syncthreads();

    // rs = (*so);
    // deo_cuda(piup, pidn, u, r, ceo, one_over_gammaf, rs); // This needs rework (Cannot be parallelized this way)
}

// // This kernel runs in 1 thread in the gpu, with a for loop to simulate
// // the original code and produce the correct results.
// // Of course it is garbage in terms of performance.
// extern "C" __global__
// void Dw_cuda_kernel2(int VOLUME, float mu, spinor *s, spinor *r,
//                      int *piup, int *pidn, su3 *u, pauli *m,
//                      float coe, float ceo, float gamma_f, float one_over_gammaf)
// {
//     int idx = blockIdx.x*blockDim.x + threadIdx.x;
//     if (idx >= 1) return;  // Only for tests and to produce the correct results
//
//     su3 *um;
//     um = u + 4 * VOLUME;
//
//     spin_t rs;
//
//     spin_t *so, *ro;
//     so = (spin_t *)(s + (VOLUME / 2));
//
//     for (; u < um; u += 8) {
//         rs = (*so);
//         deo_cuda(piup, pidn, u, r, ceo, one_over_gammaf, rs);
//
//         piup += 4;
//         pidn += 4;
//         so += 1;
//     }
// }


extern "C" __global__
void Dw_cuda_kernel_deo(int VOLUME, float mu, spinor *s, spinor *r,
                        int *piup, int *pidn, su3 *u, pauli *m,
                        float coe, float ceo, float gamma_f, float one_over_gammaf)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= VOLUME/2) return;

    spin_t rs;

    spin_t *so;
    so = (spin_t *)(s + (VOLUME / 2));

    u    += idx*8;
    piup += idx*4;
    pidn += idx*4;
    so   += idx*1;

    rs = (*so);

    deo_cuda_atomics(piup, pidn, u, r, ceo, one_over_gammaf, rs);
}

extern "C"
void Dw_cuda(spinor *r_cpu, char *SIZE)
{
    char buffer[100];
    size_t result;
    FILE *ptr;

    // Read again everyting before calling the cuda kernel
    // Read VOLUME
    int VOLUME;
    buffer[0] = '\0';
    strcat(buffer, SIZE);
    strcat(buffer, "/dataBefore/VOLUME.bin");
    ptr = fopen(buffer, "rb");
    result = fread(&VOLUME, sizeof(int), 1, ptr);
    fclose(ptr);

    // Read mu
    float mu;
    buffer[0] = '\0';
    strcat(buffer, SIZE);
    strcat(buffer, "/dataBefore/mu.bin");
    ptr = fopen(buffer, "rb");
    result = fread(&mu, sizeof(float), 1, ptr);
    fclose(ptr);

    // Read s
    spinor *s;
    s = (spinor*) malloc(VOLUME * sizeof(spinor));
    buffer[0] = '\0';
    strcat(buffer, SIZE);
    strcat(buffer, "/dataBefore/s.bin");
    ptr = fopen(buffer, "rb");
    result = fread(s, sizeof(spinor), VOLUME, ptr);
    fclose(ptr);

    // Read r
    spinor *r;
    r = (spinor*) malloc(VOLUME * sizeof(spinor));
    buffer[0] = '\0';
    strcat(buffer, SIZE);
    strcat(buffer, "/dataBefore/r.bin");
    ptr = fopen(buffer, "rb");
    result = fread(r, sizeof(spinor), VOLUME, ptr);
    fclose(ptr);

    float coe, ceo;
    float gamma_f, one_over_gammaf;


    // Read piup
    int *piup;
    piup = (int*) malloc(2 * VOLUME * sizeof(int));
    buffer[0] = '\0';
    strcat(buffer, SIZE);
    strcat(buffer, "/dataBefore/piup.bin");
    ptr = fopen(buffer, "rb");
    result = fread(piup, sizeof(int), 2 * VOLUME, ptr);
    fclose(ptr);

    // Read pidn
    int *pidn;
    pidn = (int*) malloc(2 * VOLUME * sizeof(int));
    buffer[0] = '\0';
    strcat(buffer, SIZE);
    strcat(buffer, "/dataBefore/pidn.bin");
    ptr = fopen(buffer, "rb");
    result = fread(pidn, sizeof(int), 2 * VOLUME, ptr);
    fclose(ptr);

    // Read u
    su3 *u;
    u = (su3*) malloc(4 * VOLUME * sizeof(su3));
    buffer[0] = '\0';
    strcat(buffer, SIZE);
    strcat(buffer, "/dataBefore/u.bin");
    ptr = fopen(buffer, "rb");
    result = fread(u, sizeof(su3), 4 * VOLUME, ptr);
    fclose(ptr);

    // Read m
    pauli *m;
    m = (pauli*) malloc(VOLUME * sizeof(pauli));
    buffer[0] = '\0';
    strcat(buffer, SIZE);
    strcat(buffer, "/dataBefore/m.bin");
    ptr = fopen(buffer, "rb");
    result = fread(m, sizeof(pauli), VOLUME, ptr);
    fclose(ptr);


    coe = -0.5f;
    ceo = -0.5f;

    gamma_f = 1.0f;
    one_over_gammaf = 1.0f;

    // Device copies
    spinor *d_s, *d_r;
    int *d_piup, *d_pidn;
    su3 *d_u;
    pauli *d_m;

    // Allocate space for device copies
    cudaMalloc((void **)&d_s, VOLUME * sizeof(spinor));
    cudaMalloc((void **)&d_r, VOLUME * sizeof(spinor));
    cudaMalloc((void **)&d_piup, 2 * VOLUME * sizeof(int));
    cudaMalloc((void **)&d_pidn, 2 * VOLUME * sizeof(int));
    cudaMalloc((void **)&d_u, 4 * VOLUME * sizeof(su3));
    cudaMalloc((void **)&d_m, VOLUME * sizeof(pauli));

    // Copy data from host to device
    cudaMemcpy(d_s, s, VOLUME * sizeof(spinor), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, r, VOLUME * sizeof(spinor), cudaMemcpyHostToDevice);
    cudaMemcpy(d_piup, piup, 2 * VOLUME * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pidn, pidn, 2 * VOLUME * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, 4 * VOLUME * sizeof(su3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, m, VOLUME * sizeof(pauli), cudaMemcpyHostToDevice);

    // Launch kernel on GPU
    int block_size = 128;
    int grid_size = ceil((VOLUME/2.0)/(float)block_size);
    Dw_cuda_kernel<<<grid_size, block_size>>>(VOLUME, mu, d_s, d_r,
                                              d_piup, d_pidn, d_u, d_m,
                                              coe, ceo, gamma_f, one_over_gammaf);

    // Dw_cuda_kernel2<<<grid_size, block_size>>>(VOLUME, mu, d_s, d_r,
    //                                            d_piup, d_pidn, d_u, d_m,
    //                                            coe, ceo, gamma_f, one_over_gammaf);
    Dw_cuda_kernel_deo<<<grid_size, block_size>>>(VOLUME, mu, d_s, d_r,
                                                  d_piup, d_pidn, d_u, d_m,
                                                  coe, ceo, gamma_f, one_over_gammaf);

    // Copy result back to the host
    cudaMemcpy(r, d_r, VOLUME * sizeof(spinor), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_s);
    cudaFree(d_r);
    cudaFree(d_piup);
    cudaFree(d_pidn);
    cudaFree(d_u);
    cudaFree(d_m);

    // Read r after from disk
    spinor *r_after;
    r_after = (spinor*) malloc(VOLUME * sizeof(spinor));
    buffer[0] = '\0';
    strcat(buffer, SIZE);
    strcat(buffer, "/dataAfter/r.bin");
    ptr = fopen(buffer, "rb");
    result = fread(r_after, sizeof(spinor), VOLUME, ptr);
    fclose(ptr);

    int ret;

    // // Compare value from disk with r in memory after execution of Dw_cuda()
    // ret = memcmp(r, r_after, VOLUME * sizeof(spinor));
    // if (ret == 0) {
    //     printf("Values in spinor r are correct after calling Dw_cuda()\n");
    // }
    // else {
    //     printf("Values in spinor r are incorrect after calling Dw_cuda()\n");
    // }

    // Compare value from cpu computations with r from gpu after execution of Dw_cuda()
    int count = 0;
    for (int i = 0; i < VOLUME; ++i) {
        ret = my_memcmp(r+i, r_cpu+i, sizeof(spinor)/sizeof(float));
        if (ret == 0) {
            count++;
        }
        else {
            printf("Values in spinor r are different between cpu and gpu: %d\n", i);
        }
    }
    if (count == VOLUME) {
        printf("Values in spinor r are the same between cpu and gpu\n");
    }

    // // Write to disk
    // ptr = fopen("r_after_cuda.bin", "w");
    // fwrite (r, sizeof(spinor), VOLUME, ptr);
    // fclose (ptr);

}
