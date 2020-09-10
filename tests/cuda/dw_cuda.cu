// #include "dw_cuda.h"
#include "su3.h"
#include "macros.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


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
void mulpauli_kernel(int vol, float mu, spinor *s, spinor *r, pauli *m)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= vol) return;

    weyl rt;

    spin_t *so, *ro;
    so = (spin_t *)(s + 0);
    ro = (spin_t *)(r + 0);

    so += (idx)*1;
    ro += (idx)*1;
    m += (idx)*2;

    mul_pauli2_cuda(mu, m, &((*so).s), &((*ro).s), rt);
}

extern "C" __global__
void doe_kernel(int vol, spinor *s, spinor *r, su3 *u,
                int *piup, int *pidn, float coe,
                float gamma_f, float one_over_gammaf)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= vol/2) return;

    spin_t rs;

    spin_t *ro;
    ro = (spin_t *)(r + (vol / 2));

    u += (idx)*8;
    piup += (idx)*4;
    pidn += (idx)*4;
    ro += (idx)*1;

    rs = doe_cuda(piup, pidn, u, s, coe, gamma_f, one_over_gammaf, rs);

    _vector_add_assign((*ro).s.c1, rs.s.c1);
    _vector_add_assign((*ro).s.c2, rs.s.c2);
    _vector_add_assign((*ro).s.c3, rs.s.c3);
    _vector_add_assign((*ro).s.c4, rs.s.c4);
}

extern "C" __global__
void deo_kernel(int vol, spinor *s, spinor *r, su3 *u,
                int *piup, int *pidn, float ceo,
                float one_over_gammaf)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= vol/2) return;

    spin_t rs;

    spin_t *so;
    so = (spin_t *)(s + (vol / 2));

    u    += idx*8;
    piup += idx*4;
    pidn += idx*4;
    so   += idx*1;

    rs = (*so);

    deo_cuda_atomics(piup, pidn, u, r, ceo, one_over_gammaf, rs);
}

extern "C"
void Dw_cuda_AoS(int VOLUME, su3 *u, spinor *s, spinor *r, pauli *m, int *piup, int *pidn)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    float mu, coe, ceo;
    float gamma_f, one_over_gammaf;

    mu = 0.0f;
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
    cudaMalloc((void **)&d_m, 2 * VOLUME * sizeof(pauli));

    // Copy data from host to device
    cudaEventRecord(start);
    cudaMemcpy(d_s, s, VOLUME * sizeof(spinor), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, r, VOLUME * sizeof(spinor), cudaMemcpyHostToDevice);
    cudaMemcpy(d_piup, piup, 2 * VOLUME * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pidn, pidn, 2 * VOLUME * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, 4 * VOLUME * sizeof(su3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, m, 2 * VOLUME * sizeof(pauli), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for cudaMemcpy H2D (ms): %.2f\n", milliseconds);

    int block_size, grid_size;

    // Launch kernels on GPU
    block_size = 128;
    grid_size = ceil(VOLUME/(float)block_size);
    cudaEventRecord(start);
    mulpauli_kernel<<<grid_size, block_size>>>(VOLUME, mu, d_s, d_r, d_m);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for kernel mul_pauli (ms): %.2f\n", milliseconds);


    block_size = 128;
    grid_size = ceil((VOLUME/2.0)/(float)block_size);
    cudaEventRecord(start);
    doe_kernel<<<grid_size, block_size>>>(VOLUME, d_s, d_r, d_u,
                                          d_piup, d_pidn, coe,
                                          gamma_f, one_over_gammaf);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for kernel doe (ms): %.2f\n", milliseconds);


    block_size = 128;
    grid_size = ceil((VOLUME/2.0)/(float)block_size);
    cudaEventRecord(start);
    deo_kernel<<<grid_size, block_size>>>(VOLUME, d_s, d_r, d_u,
                                          d_piup, d_pidn, ceo,
                                          one_over_gammaf);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for kernel deo (ms): %.2f\n", milliseconds);


    // Copy result back to the host
    cudaEventRecord(start);
    cudaMemcpy(r, d_r, VOLUME * sizeof(spinor), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for cudaMemcpy D2H (ms): %.2f\n", milliseconds);

    // Cleanup
    cudaFree(d_s);
    cudaFree(d_r);
    cudaFree(d_piup);
    cudaFree(d_pidn);
    cudaFree(d_u);
    cudaFree(d_m);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // // Write to disk
    // FILE *ptr;
    // ptr = fopen("r_after_cuda.bin", "w");
    // fwrite(r, sizeof(spinor), VOLUME, ptr);
    // fclose(ptr);

}
