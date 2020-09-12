// #include "dw_cuda_soa.h"
#include "su3.h"
#include "macros.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


__device__
void vector_add_assign(su3_vector_soa r, su3_vector s, int idx)
{
    atomicAdd(&(r.c1.re[idx]),  s.c1.re);
    atomicAdd(&(r.c1.im[idx]),  s.c1.im);
    atomicAdd(&(r.c2.re[idx]),  s.c2.re);
    atomicAdd(&(r.c2.im[idx]),  s.c2.im);
    atomicAdd(&(r.c3.re[idx]),  s.c3.re);
    atomicAdd(&(r.c3.im[idx]),  s.c3.im);
}

__device__
void vector_sub_assign(su3_vector_soa r, su3_vector s, int idx)
{
    atomicAdd(&(r.c1.re[idx]), -s.c1.re);
    atomicAdd(&(r.c1.im[idx]), -s.c1.im);
    atomicAdd(&(r.c2.re[idx]), -s.c2.re);
    atomicAdd(&(r.c2.im[idx]), -s.c2.im);
    atomicAdd(&(r.c3.re[idx]), -s.c3.re);
    atomicAdd(&(r.c3.im[idx]), -s.c3.im);
}

__device__
void vector_i_add_assign(su3_vector_soa r, su3_vector s, int idx)
{
    atomicAdd(&(r.c1.re[idx]), -s.c1.im);
    atomicAdd(&(r.c1.im[idx]),  s.c1.re);
    atomicAdd(&(r.c2.re[idx]), -s.c2.im);
    atomicAdd(&(r.c2.im[idx]),  s.c2.re);
    atomicAdd(&(r.c3.re[idx]), -s.c3.im);
    atomicAdd(&(r.c3.im[idx]),  s.c3.re);
}

__device__
void vector_i_sub_assign(su3_vector_soa r, su3_vector s, int idx)
{
    atomicAdd(&(r.c1.re[idx]),  s.c1.im);
    atomicAdd(&(r.c1.im[idx]), -s.c1.re);
    atomicAdd(&(r.c2.re[idx]),  s.c2.im);
    atomicAdd(&(r.c2.im[idx]), -s.c2.re);
    atomicAdd(&(r.c3.re[idx]),  s.c3.im);
    atomicAdd(&(r.c3.im[idx]), -s.c3.re);
}

pauli_soa allocPauli2Device(int vol)
{
    pauli_soa d_m;

    // Allocate memory on device
    cudaMalloc((void **)&(d_m.m1), 36 * vol * sizeof(float));
    cudaMalloc((void **)&(d_m.m2), 36 * vol * sizeof(float));

    return d_m;
}

su3_soa allocSu32Device(int vol)
{
    su3_soa d_u;

    cudaMalloc((void **)&(d_u.c11.re), 4 * vol * sizeof(float));
    cudaMalloc((void **)&(d_u.c11.im), 4 * vol * sizeof(float));
    cudaMalloc((void **)&(d_u.c12.re), 4 * vol * sizeof(float));
    cudaMalloc((void **)&(d_u.c12.im), 4 * vol * sizeof(float));
    cudaMalloc((void **)&(d_u.c13.re), 4 * vol * sizeof(float));
    cudaMalloc((void **)&(d_u.c13.im), 4 * vol * sizeof(float));
    cudaMalloc((void **)&(d_u.c21.re), 4 * vol * sizeof(float));
    cudaMalloc((void **)&(d_u.c21.im), 4 * vol * sizeof(float));
    cudaMalloc((void **)&(d_u.c22.re), 4 * vol * sizeof(float));
    cudaMalloc((void **)&(d_u.c22.im), 4 * vol * sizeof(float));
    cudaMalloc((void **)&(d_u.c23.re), 4 * vol * sizeof(float));
    cudaMalloc((void **)&(d_u.c23.im), 4 * vol * sizeof(float));
    cudaMalloc((void **)&(d_u.c31.re), 4 * vol * sizeof(float));
    cudaMalloc((void **)&(d_u.c31.im), 4 * vol * sizeof(float));
    cudaMalloc((void **)&(d_u.c32.re), 4 * vol * sizeof(float));
    cudaMalloc((void **)&(d_u.c32.im), 4 * vol * sizeof(float));
    cudaMalloc((void **)&(d_u.c33.re), 4 * vol * sizeof(float));
    cudaMalloc((void **)&(d_u.c33.im), 4 * vol * sizeof(float));

    return d_u;
}

spinor_soa allocSpinor2Device(int vol)
{
    spinor_soa d_s;

    cudaMalloc((void **)&(d_s.c1.c1.re), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c1.c1.im), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c1.c2.re), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c1.c2.im), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c1.c3.re), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c1.c3.im), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c2.c1.re), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c2.c1.im), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c2.c2.re), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c2.c2.im), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c2.c3.re), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c2.c3.im), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c3.c1.re), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c3.c1.im), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c3.c2.re), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c3.c2.im), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c3.c3.re), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c3.c3.im), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c4.c1.re), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c4.c1.im), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c4.c2.re), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c4.c2.im), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c4.c3.re), vol * sizeof(float));
    cudaMalloc((void **)&(d_s.c4.c3.im), vol * sizeof(float));

    return d_s;
}


void destroy_pauli_soa(pauli_soa obj)
{
    cudaFree(obj.m1);
    cudaFree(obj.m2);
}

void destroy_su3_soa(su3_soa obj)
{
    cudaFree(obj.c11.re);
    cudaFree(obj.c11.im);
    cudaFree(obj.c12.re);
    cudaFree(obj.c12.im);
    cudaFree(obj.c13.re);
    cudaFree(obj.c13.im);
    cudaFree(obj.c21.re);
    cudaFree(obj.c21.im);
    cudaFree(obj.c22.re);
    cudaFree(obj.c22.im);
    cudaFree(obj.c23.re);
    cudaFree(obj.c23.im);
    cudaFree(obj.c31.re);
    cudaFree(obj.c31.im);
    cudaFree(obj.c32.re);
    cudaFree(obj.c32.im);
    cudaFree(obj.c33.re);
    cudaFree(obj.c33.im);
}

void destroy_spinor_soa(spinor_soa obj)
{
    cudaFree(obj.c1.c1.re);
    cudaFree(obj.c1.c1.im);
    cudaFree(obj.c1.c2.re);
    cudaFree(obj.c1.c2.im);
    cudaFree(obj.c1.c3.re);
    cudaFree(obj.c1.c3.im);
    cudaFree(obj.c2.c1.re);
    cudaFree(obj.c2.c1.im);
    cudaFree(obj.c2.c2.re);
    cudaFree(obj.c2.c2.im);
    cudaFree(obj.c2.c3.re);
    cudaFree(obj.c2.c3.im);
    cudaFree(obj.c3.c1.re);
    cudaFree(obj.c3.c1.im);
    cudaFree(obj.c3.c2.re);
    cudaFree(obj.c3.c2.im);
    cudaFree(obj.c3.c3.re);
    cudaFree(obj.c3.c3.im);
    cudaFree(obj.c4.c1.re);
    cudaFree(obj.c4.c1.im);
    cudaFree(obj.c4.c2.re);
    cudaFree(obj.c4.c2.im);
    cudaFree(obj.c4.c3.re);
    cudaFree(obj.c4.c3.im);
}


// Kernel to convert pauli from AoS to SoA in GPU
extern "C" __global__
void pauli_AoS2SoA(int vol, pauli_soa mout, pauli *min)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= vol) return;

    int j;

    // A+
    for (j = 0; j < 36; ++j) {
        mout.m1[j*vol + idx] = (*(min+2*idx+0)).u[j];
    }

    // A-
    for (j = 0; j < 36; ++j) {
        mout.m2[j*vol + idx] = (*(min+2*idx+1)).u[j];
    }
}

// Kernel to convert su3 from AoS to SoA in GPU
extern "C" __global__
void su3_AoS2SoA(int vol, su3_soa uout, su3 *uin)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= vol/2) return;

    int idSoA, idAoS;

    for (int j = 0; j < 8; ++j) {
        idSoA = (vol/2)*j + idx;
        idAoS = 8*idx + j;
        uout.c11.re[idSoA] = (*(uin+idAoS)).c11.re;
        uout.c11.im[idSoA] = (*(uin+idAoS)).c11.im;
        uout.c12.re[idSoA] = (*(uin+idAoS)).c12.re;
        uout.c12.im[idSoA] = (*(uin+idAoS)).c12.im;
        uout.c13.re[idSoA] = (*(uin+idAoS)).c13.re;
        uout.c13.im[idSoA] = (*(uin+idAoS)).c13.im;
        uout.c21.re[idSoA] = (*(uin+idAoS)).c21.re;
        uout.c21.im[idSoA] = (*(uin+idAoS)).c21.im;
        uout.c22.re[idSoA] = (*(uin+idAoS)).c22.re;
        uout.c22.im[idSoA] = (*(uin+idAoS)).c22.im;
        uout.c23.re[idSoA] = (*(uin+idAoS)).c23.re;
        uout.c23.im[idSoA] = (*(uin+idAoS)).c23.im;
        uout.c31.re[idSoA] = (*(uin+idAoS)).c31.re;
        uout.c31.im[idSoA] = (*(uin+idAoS)).c31.im;
        uout.c32.re[idSoA] = (*(uin+idAoS)).c32.re;
        uout.c32.im[idSoA] = (*(uin+idAoS)).c32.im;
        uout.c33.re[idSoA] = (*(uin+idAoS)).c33.re;
        uout.c33.im[idSoA] = (*(uin+idAoS)).c33.im;
    }
}

// Kernel to convert spinor from AoS to SoA in GPU
extern "C" __global__
void spinor_AoS2SoA(int vol, spinor_soa rout, spinor *rin)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= vol) return;

    rout.c1.c1.re[idx] = (*(rin+idx)).c1.c1.re;
    rout.c1.c1.im[idx] = (*(rin+idx)).c1.c1.im;
    rout.c1.c2.re[idx] = (*(rin+idx)).c1.c2.re;
    rout.c1.c2.im[idx] = (*(rin+idx)).c1.c2.im;
    rout.c1.c3.re[idx] = (*(rin+idx)).c1.c3.re;
    rout.c1.c3.im[idx] = (*(rin+idx)).c1.c3.im;
    rout.c2.c1.re[idx] = (*(rin+idx)).c2.c1.re;
    rout.c2.c1.im[idx] = (*(rin+idx)).c2.c1.im;
    rout.c2.c2.re[idx] = (*(rin+idx)).c2.c2.re;
    rout.c2.c2.im[idx] = (*(rin+idx)).c2.c2.im;
    rout.c2.c3.re[idx] = (*(rin+idx)).c2.c3.re;
    rout.c2.c3.im[idx] = (*(rin+idx)).c2.c3.im;
    rout.c3.c1.re[idx] = (*(rin+idx)).c3.c1.re;
    rout.c3.c1.im[idx] = (*(rin+idx)).c3.c1.im;
    rout.c3.c2.re[idx] = (*(rin+idx)).c3.c2.re;
    rout.c3.c2.im[idx] = (*(rin+idx)).c3.c2.im;
    rout.c3.c3.re[idx] = (*(rin+idx)).c3.c3.re;
    rout.c3.c3.im[idx] = (*(rin+idx)).c3.c3.im;
    rout.c4.c1.re[idx] = (*(rin+idx)).c4.c1.re;
    rout.c4.c1.im[idx] = (*(rin+idx)).c4.c1.im;
    rout.c4.c2.re[idx] = (*(rin+idx)).c4.c2.re;
    rout.c4.c2.im[idx] = (*(rin+idx)).c4.c2.im;
    rout.c4.c3.re[idx] = (*(rin+idx)).c4.c3.re;
    rout.c4.c3.im[idx] = (*(rin+idx)).c4.c3.im;
}

// Kernel to convert spinor from SoA to AoS in GPU
extern "C" __global__
void spinor_SoA2AoS(int vol, spinor *rout, spinor_soa rin)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= vol) return;

    (*(rout+idx)).c1.c1.re = rin.c1.c1.re[idx];
    (*(rout+idx)).c1.c1.im = rin.c1.c1.im[idx];
    (*(rout+idx)).c1.c2.re = rin.c1.c2.re[idx];
    (*(rout+idx)).c1.c2.im = rin.c1.c2.im[idx];
    (*(rout+idx)).c1.c3.re = rin.c1.c3.re[idx];
    (*(rout+idx)).c1.c3.im = rin.c1.c3.im[idx];
    (*(rout+idx)).c2.c1.re = rin.c2.c1.re[idx];
    (*(rout+idx)).c2.c1.im = rin.c2.c1.im[idx];
    (*(rout+idx)).c2.c2.re = rin.c2.c2.re[idx];
    (*(rout+idx)).c2.c2.im = rin.c2.c2.im[idx];
    (*(rout+idx)).c2.c3.re = rin.c2.c3.re[idx];
    (*(rout+idx)).c2.c3.im = rin.c2.c3.im[idx];
    (*(rout+idx)).c3.c1.re = rin.c3.c1.re[idx];
    (*(rout+idx)).c3.c1.im = rin.c3.c1.im[idx];
    (*(rout+idx)).c3.c2.re = rin.c3.c2.re[idx];
    (*(rout+idx)).c3.c2.im = rin.c3.c2.im[idx];
    (*(rout+idx)).c3.c3.re = rin.c3.c3.re[idx];
    (*(rout+idx)).c3.c3.im = rin.c3.c3.im[idx];
    (*(rout+idx)).c4.c1.re = rin.c4.c1.re[idx];
    (*(rout+idx)).c4.c1.im = rin.c4.c1.im[idx];
    (*(rout+idx)).c4.c2.re = rin.c4.c2.re[idx];
    (*(rout+idx)).c4.c2.im = rin.c4.c2.im[idx];
    (*(rout+idx)).c4.c3.re = rin.c4.c3.re[idx];
    (*(rout+idx)).c4.c3.im = rin.c4.c3.im[idx];
}


// ---------------------------------------------------------------------------//
extern "C" __global__
void mulpauli_kernel(int vol, float mu, spinor_soa s, spinor_soa r, pauli_soa m)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= vol) return;

    float u[36];
    weyl sloc;

    // A+
    sloc.c1.c1.re = s.c1.c1.re[idx];
    sloc.c1.c1.im = s.c1.c1.im[idx];
    sloc.c1.c2.re = s.c1.c2.re[idx];
    sloc.c1.c2.im = s.c1.c2.im[idx];
    sloc.c1.c3.re = s.c1.c3.re[idx];
    sloc.c1.c3.im = s.c1.c3.im[idx];
    sloc.c2.c1.re = s.c2.c1.re[idx];
    sloc.c2.c1.im = s.c2.c1.im[idx];
    sloc.c2.c2.re = s.c2.c2.re[idx];
    sloc.c2.c2.im = s.c2.c2.im[idx];
    sloc.c2.c3.re = s.c2.c3.re[idx];
    sloc.c2.c3.im = s.c2.c3.im[idx];

    #pragma unroll
    for (int i = 0; i < 36; ++i) {
        u[i] = m.m1[i*vol + idx];
    }

    r.c1.c1.re[idx] =
      u[0]  * sloc.c1.c1.re - mu    * sloc.c1.c1.im + u[6]  * sloc.c1.c2.re -
      u[7]  * sloc.c1.c2.im + u[8]  * sloc.c1.c3.re - u[9]  * sloc.c1.c3.im +
      u[10] * sloc.c2.c1.re - u[11] * sloc.c2.c1.im + u[12] * sloc.c2.c2.re -
      u[13] * sloc.c2.c2.im + u[14] * sloc.c2.c3.re - u[15] * sloc.c2.c3.im;

    r.c1.c1.im[idx] =
      u[0]  * sloc.c1.c1.im + mu    * sloc.c1.c1.re + u[6]  * sloc.c1.c2.im +
      u[7]  * sloc.c1.c2.re + u[8]  * sloc.c1.c3.im + u[9]  * sloc.c1.c3.re +
      u[10] * sloc.c2.c1.im + u[11] * sloc.c2.c1.re + u[12] * sloc.c2.c2.im +
      u[13] * sloc.c2.c2.re + u[14] * sloc.c2.c3.im + u[15] * sloc.c2.c3.re;

    r.c1.c2.re[idx] =
      u[6]  * sloc.c1.c1.re + u[7]  * sloc.c1.c1.im + u[1]  * sloc.c1.c2.re -
      mu    * sloc.c1.c2.im + u[16] * sloc.c1.c3.re - u[17] * sloc.c1.c3.im +
      u[18] * sloc.c2.c1.re - u[19] * sloc.c2.c1.im + u[20] * sloc.c2.c2.re -
      u[21] * sloc.c2.c2.im + u[22] * sloc.c2.c3.re - u[23] * sloc.c2.c3.im;

    r.c1.c2.im[idx] =
      u[6]  * sloc.c1.c1.im - u[7]  * sloc.c1.c1.re + u[1]  * sloc.c1.c2.im +
      mu    * sloc.c1.c2.re + u[16] * sloc.c1.c3.im + u[17] * sloc.c1.c3.re +
      u[18] * sloc.c2.c1.im + u[19] * sloc.c2.c1.re + u[20] * sloc.c2.c2.im +
      u[21] * sloc.c2.c2.re + u[22] * sloc.c2.c3.im + u[23] * sloc.c2.c3.re;

    r.c1.c3.re[idx] =
      u[8]  * sloc.c1.c1.re + u[9]  * sloc.c1.c1.im + u[16] * sloc.c1.c2.re +
      u[17] * sloc.c1.c2.im + u[2]  * sloc.c1.c3.re - mu    * sloc.c1.c3.im +
      u[24] * sloc.c2.c1.re - u[25] * sloc.c2.c1.im + u[26] * sloc.c2.c2.re -
      u[27] * sloc.c2.c2.im + u[28] * sloc.c2.c3.re - u[29] * sloc.c2.c3.im;

    r.c1.c3.im[idx] =
      u[8]  * sloc.c1.c1.im - u[9]  * sloc.c1.c1.re + u[16] * sloc.c1.c2.im -
      u[17] * sloc.c1.c2.re + u[2]  * sloc.c1.c3.im + mu    * sloc.c1.c3.re +
      u[24] * sloc.c2.c1.im + u[25] * sloc.c2.c1.re + u[26] * sloc.c2.c2.im +
      u[27] * sloc.c2.c2.re + u[28] * sloc.c2.c3.im + u[29] * sloc.c2.c3.re;

    r.c2.c1.re[idx] =
      u[10] * sloc.c1.c1.re + u[11] * sloc.c1.c1.im + u[18] * sloc.c1.c2.re +
      u[19] * sloc.c1.c2.im + u[24] * sloc.c1.c3.re + u[25] * sloc.c1.c3.im +
      u[3]  * sloc.c2.c1.re - mu * sloc.c2.c1.im    + u[30] * sloc.c2.c2.re -
      u[31] * sloc.c2.c2.im + u[32] * sloc.c2.c3.re - u[33] * sloc.c2.c3.im;

    r.c2.c1.im[idx] =
      u[10] * sloc.c1.c1.im - u[11] * sloc.c1.c1.re + u[18] * sloc.c1.c2.im -
      u[19] * sloc.c1.c2.re + u[24] * sloc.c1.c3.im - u[25] * sloc.c1.c3.re +
      u[3]  * sloc.c2.c1.im + mu    * sloc.c2.c1.re + u[30] * sloc.c2.c2.im +
      u[31] * sloc.c2.c2.re + u[32] * sloc.c2.c3.im + u[33] * sloc.c2.c3.re;

    r.c2.c2.re[idx] =
      u[12] * sloc.c1.c1.re + u[13] * sloc.c1.c1.im + u[20] * sloc.c1.c2.re +
      u[21] * sloc.c1.c2.im + u[26] * sloc.c1.c3.re + u[27] * sloc.c1.c3.im +
      u[30] * sloc.c2.c1.re + u[31] * sloc.c2.c1.im + u[4]  * sloc.c2.c2.re -
      mu    * sloc.c2.c2.im + u[34] * sloc.c2.c3.re - u[35] * sloc.c2.c3.im;

    r.c2.c2.im[idx] =
      u[12] * sloc.c1.c1.im - u[13] * sloc.c1.c1.re + u[20] * sloc.c1.c2.im -
      u[21] * sloc.c1.c2.re + u[26] * sloc.c1.c3.im - u[27] * sloc.c1.c3.re +
      u[30] * sloc.c2.c1.im - u[31] * sloc.c2.c1.re + u[4]  * sloc.c2.c2.im +
      mu    * sloc.c2.c2.re + u[34] * sloc.c2.c3.im + u[35] * sloc.c2.c3.re;

    r.c2.c3.re[idx] =
      u[14] * sloc.c1.c1.re + u[15] * sloc.c1.c1.im + u[22] * sloc.c1.c2.re +
      u[23] * sloc.c1.c2.im + u[28] * sloc.c1.c3.re + u[29] * sloc.c1.c3.im +
      u[32] * sloc.c2.c1.re + u[33] * sloc.c2.c1.im + u[34] * sloc.c2.c2.re +
      u[35] * sloc.c2.c2.im + u[5]  * sloc.c2.c3.re - mu    * sloc.c2.c3.im;

    r.c2.c3.im[idx] =
      u[14] * sloc.c1.c1.im - u[15] * sloc.c1.c1.re + u[22] * sloc.c1.c2.im -
      u[23] * sloc.c1.c2.re + u[28] * sloc.c1.c3.im - u[29] * sloc.c1.c3.re +
      u[32] * sloc.c2.c1.im - u[33] * sloc.c2.c1.re + u[34] * sloc.c2.c2.im -
      u[35] * sloc.c2.c2.re + u[5]  * sloc.c2.c3.im + mu    * sloc.c2.c3.re;


      // A-
      sloc.c1.c1.re = s.c3.c1.re[idx];
      sloc.c1.c1.im = s.c3.c1.im[idx];
      sloc.c1.c2.re = s.c3.c2.re[idx];
      sloc.c1.c2.im = s.c3.c2.im[idx];
      sloc.c1.c3.re = s.c3.c3.re[idx];
      sloc.c1.c3.im = s.c3.c3.im[idx];
      sloc.c2.c1.re = s.c4.c1.re[idx];
      sloc.c2.c1.im = s.c4.c1.im[idx];
      sloc.c2.c2.re = s.c4.c2.re[idx];
      sloc.c2.c2.im = s.c4.c2.im[idx];
      sloc.c2.c3.re = s.c4.c3.re[idx];
      sloc.c2.c3.im = s.c4.c3.im[idx];

      #pragma unroll
      for (int i = 0; i < 36; ++i) {
          u[i] = m.m2[i*vol + idx];
      }

      mu = -mu;

      r.c3.c1.re[idx] =
        u[0]  * sloc.c1.c1.re - mu    * sloc.c1.c1.im + u[6]  * sloc.c1.c2.re -
        u[7]  * sloc.c1.c2.im + u[8]  * sloc.c1.c3.re - u[9]  * sloc.c1.c3.im +
        u[10] * sloc.c2.c1.re - u[11] * sloc.c2.c1.im + u[12] * sloc.c2.c2.re -
        u[13] * sloc.c2.c2.im + u[14] * sloc.c2.c3.re - u[15] * sloc.c2.c3.im;

      r.c3.c1.im[idx] =
        u[0]  * sloc.c1.c1.im + mu    * sloc.c1.c1.re + u[6]  * sloc.c1.c2.im +
        u[7]  * sloc.c1.c2.re + u[8]  * sloc.c1.c3.im + u[9]  * sloc.c1.c3.re +
        u[10] * sloc.c2.c1.im + u[11] * sloc.c2.c1.re + u[12] * sloc.c2.c2.im +
        u[13] * sloc.c2.c2.re + u[14] * sloc.c2.c3.im + u[15] * sloc.c2.c3.re;

      r.c3.c2.re[idx] =
        u[6]  * sloc.c1.c1.re + u[7]  * sloc.c1.c1.im + u[1]  * sloc.c1.c2.re -
        mu    * sloc.c1.c2.im + u[16] * sloc.c1.c3.re - u[17] * sloc.c1.c3.im +
        u[18] * sloc.c2.c1.re - u[19] * sloc.c2.c1.im + u[20] * sloc.c2.c2.re -
        u[21] * sloc.c2.c2.im + u[22] * sloc.c2.c3.re - u[23] * sloc.c2.c3.im;

      r.c3.c2.im[idx] =
        u[6]  * sloc.c1.c1.im - u[7]  * sloc.c1.c1.re + u[1]  * sloc.c1.c2.im +
        mu    * sloc.c1.c2.re + u[16] * sloc.c1.c3.im + u[17] * sloc.c1.c3.re +
        u[18] * sloc.c2.c1.im + u[19] * sloc.c2.c1.re + u[20] * sloc.c2.c2.im +
        u[21] * sloc.c2.c2.re + u[22] * sloc.c2.c3.im + u[23] * sloc.c2.c3.re;

      r.c3.c3.re[idx] =
        u[8]  * sloc.c1.c1.re + u[9]  * sloc.c1.c1.im + u[16] * sloc.c1.c2.re +
        u[17] * sloc.c1.c2.im + u[2]  * sloc.c1.c3.re - mu    * sloc.c1.c3.im +
        u[24] * sloc.c2.c1.re - u[25] * sloc.c2.c1.im + u[26] * sloc.c2.c2.re -
        u[27] * sloc.c2.c2.im + u[28] * sloc.c2.c3.re - u[29] * sloc.c2.c3.im;

      r.c3.c3.im[idx] =
        u[8]  * sloc.c1.c1.im - u[9]  * sloc.c1.c1.re + u[16] * sloc.c1.c2.im -
        u[17] * sloc.c1.c2.re + u[2]  * sloc.c1.c3.im + mu    * sloc.c1.c3.re +
        u[24] * sloc.c2.c1.im + u[25] * sloc.c2.c1.re + u[26] * sloc.c2.c2.im +
        u[27] * sloc.c2.c2.re + u[28] * sloc.c2.c3.im + u[29] * sloc.c2.c3.re;

      r.c4.c1.re[idx] =
        u[10] * sloc.c1.c1.re + u[11] * sloc.c1.c1.im + u[18] * sloc.c1.c2.re +
        u[19] * sloc.c1.c2.im + u[24] * sloc.c1.c3.re + u[25] * sloc.c1.c3.im +
        u[3]  * sloc.c2.c1.re - mu * sloc.c2.c1.im    + u[30] * sloc.c2.c2.re -
        u[31] * sloc.c2.c2.im + u[32] * sloc.c2.c3.re - u[33] * sloc.c2.c3.im;

      r.c4.c1.im[idx] =
        u[10] * sloc.c1.c1.im - u[11] * sloc.c1.c1.re + u[18] * sloc.c1.c2.im -
        u[19] * sloc.c1.c2.re + u[24] * sloc.c1.c3.im - u[25] * sloc.c1.c3.re +
        u[3]  * sloc.c2.c1.im + mu    * sloc.c2.c1.re + u[30] * sloc.c2.c2.im +
        u[31] * sloc.c2.c2.re + u[32] * sloc.c2.c3.im + u[33] * sloc.c2.c3.re;

      r.c4.c2.re[idx] =
        u[12] * sloc.c1.c1.re + u[13] * sloc.c1.c1.im + u[20] * sloc.c1.c2.re +
        u[21] * sloc.c1.c2.im + u[26] * sloc.c1.c3.re + u[27] * sloc.c1.c3.im +
        u[30] * sloc.c2.c1.re + u[31] * sloc.c2.c1.im + u[4]  * sloc.c2.c2.re -
        mu    * sloc.c2.c2.im + u[34] * sloc.c2.c3.re - u[35] * sloc.c2.c3.im;

      r.c4.c2.im[idx] =
        u[12] * sloc.c1.c1.im - u[13] * sloc.c1.c1.re + u[20] * sloc.c1.c2.im -
        u[21] * sloc.c1.c2.re + u[26] * sloc.c1.c3.im - u[27] * sloc.c1.c3.re +
        u[30] * sloc.c2.c1.im - u[31] * sloc.c2.c1.re + u[4]  * sloc.c2.c2.im +
        mu    * sloc.c2.c2.re + u[34] * sloc.c2.c3.im + u[35] * sloc.c2.c3.re;

      r.c4.c3.re[idx] =
        u[14] * sloc.c1.c1.re + u[15] * sloc.c1.c1.im + u[22] * sloc.c1.c2.re +
        u[23] * sloc.c1.c2.im + u[28] * sloc.c1.c3.re + u[29] * sloc.c1.c3.im +
        u[32] * sloc.c2.c1.re + u[33] * sloc.c2.c1.im + u[34] * sloc.c2.c2.re +
        u[35] * sloc.c2.c2.im + u[5]  * sloc.c2.c3.re - mu    * sloc.c2.c3.im;

      r.c4.c3.im[idx] =
        u[14] * sloc.c1.c1.im - u[15] * sloc.c1.c1.re + u[22] * sloc.c1.c2.im -
        u[23] * sloc.c1.c2.re + u[28] * sloc.c1.c3.im - u[29] * sloc.c1.c3.re +
        u[32] * sloc.c2.c1.im - u[33] * sloc.c2.c1.re + u[34] * sloc.c2.c2.im -
        u[35] * sloc.c2.c2.re + u[5]  * sloc.c2.c3.im + mu    * sloc.c2.c3.re;
}
// ---------------------------------------------------------------------------//


// ---------------------------------------------------------------------------//
extern "C" __global__
void doe_kernel(int vol, spinor_soa s, spinor_soa r, su3_soa u,
                int4 *piup, int4 *pidn, float coe,
                float gamma_f, float one_over_gammaf)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= vol/2) return;

    su3 uloc;
    spinor sloc, rloc;
    su3_vector psi, chi;
    int4 piuploc = piup[idx];
    int4 pidnloc = pidn[idx];
    int sidx, uidx;

    /***************************** direction +0 *******************************/

    sidx = piuploc.x;
    uidx = 0*(vol/2) + idx;
    _spinor_copy2struct(sloc, s, sidx);
    _su3_copy2struct(uloc, u, uidx);

    _vector_add(psi, sloc.c1, sloc.c3);
    _su3_multiply(rloc.c1, uloc, psi);
    rloc.c3 = rloc.c1;

    _vector_add(psi, sloc.c2, sloc.c4);
    _su3_multiply(rloc.c2, uloc, psi);
    rloc.c4 = rloc.c2;

    /***************************** direction -0 *******************************/

    sidx = pidnloc.x;
    uidx = 1*(vol/2) + idx;
    _spinor_copy2struct(sloc, s, sidx);
    _su3_copy2struct(uloc, u, uidx);

    _vector_sub(psi, sloc.c1, sloc.c3);
    _su3_inverse_multiply(chi, uloc, psi);
    _vector_add_assign(rloc.c1, chi);
    _vector_sub_assign(rloc.c3, chi);

    _vector_sub(psi, sloc.c2, sloc.c4);
    _su3_inverse_multiply(chi, uloc, psi);
    _vector_add_assign(rloc.c2, chi);
    _vector_sub_assign(rloc.c4, chi);

    _vector_mul_assign(rloc.c1, gamma_f);
    _vector_mul_assign(rloc.c2, gamma_f);
    _vector_mul_assign(rloc.c3, gamma_f);
    _vector_mul_assign(rloc.c4, gamma_f);

    /***************************** direction +1 *******************************/

    sidx = piuploc.y;
    uidx = 2*(vol/2) + idx;
    _spinor_copy2struct(sloc, s, sidx);
    _su3_copy2struct(uloc, u, uidx);

    _vector_i_add(psi, sloc.c1, sloc.c4);
    _su3_multiply(chi, uloc, psi);
    _vector_add_assign(rloc.c1, chi);
    _vector_i_sub_assign(rloc.c4, chi);

    _vector_i_add(psi, sloc.c2, sloc.c3);
    _su3_multiply(chi, uloc, psi);
    _vector_add_assign(rloc.c2, chi);
    _vector_i_sub_assign(rloc.c3, chi);

    /***************************** direction -1 *******************************/

    sidx = pidnloc.y;
    uidx = 3*(vol/2) + idx;
    _spinor_copy2struct(sloc, s, sidx);
    _su3_copy2struct(uloc, u, uidx);

    _vector_i_sub(psi, sloc.c1, sloc.c4);
    _su3_inverse_multiply(chi, uloc, psi);
    _vector_add_assign(rloc.c1, chi);
    _vector_i_add_assign(rloc.c4, chi);

    _vector_i_sub(psi, sloc.c2, sloc.c3);
    _su3_inverse_multiply(chi, uloc, psi);
    _vector_add_assign(rloc.c2, chi);
    _vector_i_add_assign(rloc.c3, chi);

    /***************************** direction +2 *******************************/

    sidx = piuploc.z;
    uidx = 4*(vol/2) + idx;
    _spinor_copy2struct(sloc, s, sidx);
    _su3_copy2struct(uloc, u, uidx);

    _vector_add(psi, sloc.c1, sloc.c4);
    _su3_multiply(chi, uloc, psi);
    _vector_add_assign(rloc.c1, chi);
    _vector_add_assign(rloc.c4, chi);

    _vector_sub(psi, sloc.c2, sloc.c3);
    _su3_multiply(chi, uloc, psi);
    _vector_add_assign(rloc.c2, chi);
    _vector_sub_assign(rloc.c3, chi);

    /***************************** direction -2 *******************************/

    sidx = pidnloc.z;
    uidx = 5*(vol/2) + idx;
    _spinor_copy2struct(sloc, s, sidx);
    _su3_copy2struct(uloc, u, uidx);

    _vector_sub(psi, sloc.c1, sloc.c4);
    _su3_inverse_multiply(chi, uloc, psi);
    _vector_add_assign(rloc.c1, chi);
    _vector_sub_assign(rloc.c4, chi);

    _vector_add(psi, sloc.c2, sloc.c3);
    _su3_inverse_multiply(chi, uloc, psi);
    _vector_add_assign(rloc.c2, chi);
    _vector_add_assign(rloc.c3, chi);

    /***************************** direction +3 *******************************/

    sidx = piuploc.w;
    uidx = 6*(vol/2) + idx;
    _spinor_copy2struct(sloc, s, sidx);
    _su3_copy2struct(uloc, u, uidx);

    _vector_i_add(psi, sloc.c1, sloc.c3);
    _su3_multiply(chi, uloc, psi);
    _vector_add_assign(rloc.c1, chi);
    _vector_i_sub_assign(rloc.c3, chi);

    _vector_i_sub(psi, sloc.c2, sloc.c4);
    _su3_multiply(chi, uloc, psi);
    _vector_add_assign(rloc.c2, chi);
    _vector_i_add_assign(rloc.c4, chi);

    /***************************** direction -3 *******************************/

    sidx = pidnloc.w;
    uidx = 7*(vol/2) + idx;
    _spinor_copy2struct(sloc, s, sidx);
    _su3_copy2struct(uloc, u, uidx);

    _vector_i_sub(psi, sloc.c1, sloc.c3);
    _su3_inverse_multiply(chi, uloc, psi);
    _vector_add_assign(rloc.c1, chi);
    _vector_i_add_assign(rloc.c3, chi);

    _vector_i_add(psi, sloc.c2, sloc.c4);
    _su3_inverse_multiply(chi, uloc, psi);
    _vector_add_assign(rloc.c2, chi);
    _vector_i_sub_assign(rloc.c4, chi);

    _vector_mul_assign(rloc.c1, coe);
    _vector_mul_assign(rloc.c2, coe);
    _vector_mul_assign(rloc.c3, coe);
    _vector_mul_assign(rloc.c4, coe);

    _vector_mul_assign(rloc.c1, one_over_gammaf);
    _vector_mul_assign(rloc.c2, one_over_gammaf);
    _vector_mul_assign(rloc.c3, one_over_gammaf);
    _vector_mul_assign(rloc.c4, one_over_gammaf);

    // Copy from registers to global memory
    sidx = vol/2 + idx;
    _spinor_add2arrays(r, rloc, sidx);
}
// ---------------------------------------------------------------------------//


// ---------------------------------------------------------------------------//
extern "C" __global__
void deo_kernel(int vol, spinor_soa s, spinor_soa r, su3_soa u,
                int4 *piup, int4 *pidn, float ceo,
                float one_over_gammaf)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= vol/2) return;

    su3 uloc;
    spinor sloc;
    su3_vector psi, chi;
    int4 piuploc = piup[idx];
    int4 pidnloc = pidn[idx];
    int sidx, uidx;


    sidx = vol/2 + idx;
    _spinor_copy2struct(sloc, s, sidx);

    _vector_mul_assign(sloc.c1, ceo);
    _vector_mul_assign(sloc.c2, ceo);
    _vector_mul_assign(sloc.c3, ceo);
    _vector_mul_assign(sloc.c4, ceo);

    /***************************** direction +0 *******************************/

    sidx = piuploc.x;
    uidx = 0*(vol/2) + idx;
    _su3_copy2struct(uloc, u, uidx);

    _vector_sub(psi, sloc.c1, sloc.c3);
    _su3_inverse_multiply(chi, uloc, psi);
    vector_add_assign(r.c1, chi, sidx);
    vector_sub_assign(r.c3, chi, sidx);

    _vector_sub(psi, sloc.c2, sloc.c4);
    _su3_inverse_multiply(chi, uloc, psi);
    vector_add_assign(r.c2, chi, sidx);
    vector_sub_assign(r.c4, chi, sidx);

    /***************************** direction -0 *******************************/

    sidx = pidnloc.x;
    uidx = 1*(vol/2) + idx;
    _su3_copy2struct(uloc, u, uidx);

    _vector_add(psi, sloc.c1, sloc.c3);
    _su3_multiply(chi, uloc, psi);
    vector_add_assign(r.c1, chi, sidx);
    vector_add_assign(r.c3, chi, sidx);

    _vector_add(psi, sloc.c2, sloc.c4);
    _su3_multiply(chi, uloc, psi);
    vector_add_assign(r.c2, chi, sidx);
    vector_add_assign(r.c4, chi, sidx);

    /***************************** direction +1 *******************************/

    _vector_mul_assign(sloc.c1, one_over_gammaf);
    _vector_mul_assign(sloc.c2, one_over_gammaf);
    _vector_mul_assign(sloc.c3, one_over_gammaf);
    _vector_mul_assign(sloc.c4, one_over_gammaf);

    sidx = piuploc.y;
    uidx = 2*(vol/2) + idx;
    _su3_copy2struct(uloc, u, uidx);

    _vector_i_sub(psi, sloc.c1, sloc.c4);
    _su3_inverse_multiply(chi, uloc, psi);
    vector_add_assign(r.c1, chi, sidx);
    vector_i_add_assign(r.c4, chi, sidx);

    _vector_i_sub(psi, sloc.c2, sloc.c3);
    _su3_inverse_multiply(chi, uloc, psi);
    vector_add_assign(r.c2, chi, sidx);
    vector_i_add_assign(r.c3, chi, sidx);

    /***************************** direction -1 *******************************/

    sidx = pidnloc.y;
    uidx = 3*(vol/2) + idx;
    _su3_copy2struct(uloc, u, uidx);

    _vector_i_add(psi, sloc.c1, sloc.c4);
    _su3_multiply(chi, uloc, psi);
    vector_add_assign(r.c1, chi, sidx);
    vector_i_sub_assign(r.c4, chi, sidx);

    _vector_i_add(psi, sloc.c2, sloc.c3);
    _su3_multiply(chi, uloc, psi);
    vector_add_assign(r.c2, chi, sidx);
    vector_i_sub_assign(r.c3, chi, sidx);

    /***************************** direction +2 *******************************/

    sidx = piuploc.z;
    uidx = 4*(vol/2) + idx;
    _su3_copy2struct(uloc, u, uidx);

    _vector_sub(psi, sloc.c1, sloc.c4);
    _su3_inverse_multiply(chi, uloc, psi);
    vector_add_assign(r.c1, chi, sidx);
    vector_sub_assign(r.c4, chi, sidx);

    _vector_add(psi, sloc.c2, sloc.c3);
    _su3_inverse_multiply(chi, uloc, psi);
    vector_add_assign(r.c2, chi, sidx);
    vector_add_assign(r.c3, chi, sidx);

    /***************************** direction -2 *******************************/

    sidx = pidnloc.z;
    uidx = 5*(vol/2) + idx;
    _su3_copy2struct(uloc, u, uidx);

    _vector_add(psi, sloc.c1, sloc.c4);
    _su3_multiply(chi, uloc, psi);
    vector_add_assign(r.c1, chi, sidx);
    vector_add_assign(r.c4, chi, sidx);

    _vector_sub(psi, sloc.c2, sloc.c3);
    _su3_multiply(chi, uloc, psi);
    vector_add_assign(r.c2, chi, sidx);
    vector_sub_assign(r.c3, chi, sidx);

    /***************************** direction +3 *******************************/

    sidx = piuploc.w;
    uidx = 6*(vol/2) + idx;
    _su3_copy2struct(uloc, u, uidx);

    _vector_i_sub(psi, sloc.c1, sloc.c3);
    _su3_inverse_multiply(chi, uloc, psi);
    vector_add_assign(r.c1, chi, sidx);
    vector_i_add_assign(r.c3, chi, sidx);

    _vector_i_add(psi, sloc.c2, sloc.c4);
    _su3_inverse_multiply(chi, uloc, psi);
    vector_add_assign(r.c2, chi, sidx);
    vector_i_sub_assign(r.c4, chi, sidx);

    /***************************** direction -3 *******************************/

    sidx = pidnloc.w;
    uidx = 7*(vol/2) + idx;
    _su3_copy2struct(uloc, u, uidx);

    _vector_i_add(psi, sloc.c1, sloc.c3);
    _su3_multiply(chi, uloc, psi);
    vector_add_assign(r.c1, chi, sidx);
    vector_i_sub_assign(r.c3, chi, sidx);

    _vector_i_sub(psi, sloc.c2, sloc.c4);
    _su3_multiply(chi, uloc, psi);
    vector_add_assign(r.c2, chi, sidx);
    vector_i_add_assign(r.c4, chi, sidx);
}
// ---------------------------------------------------------------------------//




extern "C"
void Dw_cuda_SoA(int VOLUME, su3 *u, spinor *s, spinor *r, pauli *m, int *piup, int *pidn)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    int block_size, grid_size;

    float mu, coe, ceo;
    float gamma_f, one_over_gammaf;

    mu = 0.0f;
    coe = -0.5f;
    ceo = -0.5f;

    gamma_f = 1.0f;
    one_over_gammaf = 1.0f;


    // Copy pauli m from host to device and convert from Aos to SoA in GPU
    pauli_soa d_m_soa = allocPauli2Device(VOLUME);                              // Allocate SoA in device
    pauli *d_m_aos;
    cudaMalloc((void **)&d_m_aos, 2 * VOLUME * sizeof(pauli));                  // Allocate AoS in device
    cudaEventRecord(start);                                                     // Start the timer
    cudaMemcpy(d_m_aos, m, 2 * VOLUME * sizeof(pauli), cudaMemcpyHostToDevice); // Mem copy AoS H2D
    block_size = 128;
    grid_size = ceil(VOLUME/(float)block_size);
    pauli_AoS2SoA<<<grid_size, block_size>>>(VOLUME, d_m_soa, d_m_aos);
    cudaEventRecord(stop);                                                      // Stop the timer
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for AoS to SoA for pauli m +H2D (GPU) (ms): %.2f\n", milliseconds);
    cudaFree(d_m_aos);                                                          // Free AoS in GPU


    // Copy su3 u from host to device and convert from Aos to SoA in GPU
    su3_soa d_u_soa = allocSu32Device(VOLUME);                                  // Allocate SoA in device
    su3 *d_u_aos;
    cudaMalloc((void **)&d_u_aos, 4 * VOLUME * sizeof(su3));                    // Allocate AoS in device
    cudaEventRecord(start);                                                     // Start the timer
    cudaMemcpy(d_u_aos, u, 4 * VOLUME * sizeof(su3), cudaMemcpyHostToDevice);   // Mem copy AoS H2D
    block_size = 128;
    grid_size = ceil((VOLUME/2.0)/(float)block_size);
    su3_AoS2SoA<<<grid_size, block_size>>>(VOLUME, d_u_soa, d_u_aos);
    cudaEventRecord(stop);                                                      // Stop the timer
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for AoS to SoA for su3 u +H2D (GPU) (ms): %.2f\n", milliseconds);
    cudaFree(d_u_aos);                                                          // Free AoS in GPU


    // Copy spinor s from host to device and convert from Aos to SoA in GPU
    spinor_soa d_s_soa = allocSpinor2Device(VOLUME);                            // Allocate SoA in device
    spinor *d_s_aos;
    cudaMalloc((void **)&d_s_aos, VOLUME * sizeof(spinor));                     // Allocate AoS in device
    cudaEventRecord(start);                                                     // Start the timer
    cudaMemcpy(d_s_aos, s, VOLUME * sizeof(spinor), cudaMemcpyHostToDevice);    // Mem copy AoS H2D
    block_size = 128;
    grid_size = ceil(VOLUME/(float)block_size);
    spinor_AoS2SoA<<<grid_size, block_size>>>(VOLUME, d_s_soa, d_s_aos);
    cudaEventRecord(stop);                                                      // Stop the timer
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for AoS to SoA for spinor s +H2D (GPU) (ms): %.2f\n", milliseconds);
    cudaFree(d_s_aos);


    // Allocate memory on device for lookup tables and spinor r
    int4 *d_piup, *d_pidn;
    cudaMalloc((void **)&d_piup, 2 * VOLUME * sizeof(int));
    cudaMalloc((void **)&d_pidn, 2 * VOLUME * sizeof(int));
    spinor_soa d_r_soa = allocSpinor2Device(VOLUME);

    // Copy lookup tables from host to device
    cudaEventRecord(start);
    cudaMemcpy(d_piup, piup, 2 * VOLUME * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pidn, pidn, 2 * VOLUME * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for cudaMemcpy H2D of lookup tables (ms): %.2f\n", milliseconds);


    // Launch kernels on GPU
    block_size = 128;
    grid_size = ceil(VOLUME/(float)block_size);
    cudaEventRecord(start);
    mulpauli_kernel<<<grid_size, block_size>>>(VOLUME, mu, d_s_soa, d_r_soa, d_m_soa);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for kernel mul_pauli (ms): %.2f\n", milliseconds);


    block_size = 128;
    grid_size = ceil((VOLUME/2.0)/(float)block_size);
    cudaEventRecord(start);
    doe_kernel<<<grid_size, block_size>>>(VOLUME, d_s_soa, d_r_soa, d_u_soa,
                                          d_piup, d_pidn, coe, gamma_f, one_over_gammaf);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for kernel doe (ms): %.2f\n", milliseconds);


    block_size = 128;
    grid_size = ceil((VOLUME/2.0)/(float)block_size);
    cudaEventRecord(start);
    deo_kernel<<<grid_size, block_size>>>(VOLUME, d_s_soa, d_r_soa, d_u_soa,
                                          d_piup, d_pidn, ceo, one_over_gammaf);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for kernel deo (ms): %.2f\n", milliseconds);


    // Convert from SoA to AoS in GPU
    spinor *d_r_aos;
    cudaMalloc((void **)&d_r_aos, VOLUME * sizeof(spinor));
    block_size = 128;
    grid_size = ceil(VOLUME/(float)block_size);
    cudaEventRecord(start);
    spinor_SoA2AoS<<<grid_size, block_size>>>(VOLUME, d_r_aos, d_r_soa);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for SoA to AoS (GPU) (ms): %.2f\n", milliseconds);


    // Copy result back to the host
    cudaEventRecord(start);
    cudaMemcpy(r, d_r_aos, VOLUME * sizeof(spinor), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for cudaMemcpy D2H (ms): %.2f\n", milliseconds);



    // Free GPU memory
    destroy_pauli_soa(d_m_soa);
    destroy_su3_soa(d_u_soa);
    destroy_spinor_soa(d_s_soa);
    destroy_spinor_soa(d_r_soa);
    cudaFree(d_piup);
    cudaFree(d_pidn);
    cudaFree(d_r_aos);


    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
