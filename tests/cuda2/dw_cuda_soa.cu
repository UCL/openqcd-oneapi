// #include "dw_cuda_soa.h"
#include "su3.h"
#include "macros.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

pauli_soa allocPauli2Device(int vol)
{
    pauli_soa d_m;

    // Allocate memory on device
    cudaMalloc((void **)&(d_m.m1), 36 * vol * sizeof(float));
    cudaMalloc((void **)&(d_m.m2), 36 * vol * sizeof(float));

    return d_m;
}

void copyPauliHost2Device(pauli_soa d_m, pauli_soa *m, int vol)
{
    // Copy data from host to device
    cudaMemcpy(d_m.m1, (*m).m1, 36 * vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m.m2, (*m).m2, 36 * vol * sizeof(float), cudaMemcpyHostToDevice);
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

void copySpinorHost2Device(spinor_soa d_s, spinor_soa *s, int vol)
{
    cudaMemcpy(d_s.c1.c1.re, (*s).c1.c1.re, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c1.c1.im, (*s).c1.c1.im, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c1.c2.re, (*s).c1.c2.re, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c1.c2.im, (*s).c1.c2.im, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c1.c3.re, (*s).c1.c3.re, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c1.c3.im, (*s).c1.c3.im, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c2.c1.re, (*s).c2.c1.re, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c2.c1.im, (*s).c2.c1.im, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c2.c2.re, (*s).c2.c2.re, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c2.c2.im, (*s).c2.c2.im, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c2.c3.re, (*s).c2.c3.re, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c2.c3.im, (*s).c2.c3.im, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c3.c1.re, (*s).c3.c1.re, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c3.c1.im, (*s).c3.c1.im, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c3.c2.re, (*s).c3.c2.re, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c3.c2.im, (*s).c3.c2.im, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c3.c3.re, (*s).c3.c3.re, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c3.c3.im, (*s).c3.c3.im, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c4.c1.re, (*s).c4.c1.re, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c4.c1.im, (*s).c4.c1.im, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c4.c2.re, (*s).c4.c2.re, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c4.c2.im, (*s).c4.c2.im, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c4.c3.re, (*s).c4.c3.re, vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.c4.c3.im, (*s).c4.c3.im, vol * sizeof(float), cudaMemcpyHostToDevice);
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

void copySu3Host2Device(su3_soa d_u, su3_soa *u, int vol)
{
    cudaMemcpy(d_u.c11.re, (*u).c11.re, 4 * vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u.c11.im, (*u).c11.im, 4 * vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u.c12.re, (*u).c12.re, 4 * vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u.c12.im, (*u).c12.im, 4 * vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u.c13.re, (*u).c13.re, 4 * vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u.c13.im, (*u).c13.im, 4 * vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u.c21.re, (*u).c21.re, 4 * vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u.c21.im, (*u).c21.im, 4 * vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u.c22.re, (*u).c22.re, 4 * vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u.c22.im, (*u).c22.im, 4 * vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u.c23.re, (*u).c23.re, 4 * vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u.c23.im, (*u).c23.im, 4 * vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u.c31.re, (*u).c31.re, 4 * vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u.c31.im, (*u).c31.im, 4 * vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u.c32.re, (*u).c32.re, 4 * vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u.c32.im, (*u).c32.im, 4 * vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u.c33.re, (*u).c33.re, 4 * vol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u.c33.im, (*u).c33.im, 4 * vol * sizeof(float), cudaMemcpyHostToDevice);
}

pauli_soa* create_pauli_soa(int vol)
{
    pauli_soa* obj = (pauli_soa*) malloc(sizeof(pauli_soa));

    (*obj).m1 = (float*) malloc(36 * vol * sizeof(float));
    (*obj).m2 = (float*) malloc(36 * vol * sizeof(float));

    return obj;
}

void destroy_pauli_soa(pauli_soa* obj)
{
    free((*obj).m1);
    free((*obj).m2);

    free(obj);
}

spinor_soa* create_spinor_soa(int vol)
{
    spinor_soa* obj = (spinor_soa*) malloc(sizeof(spinor_soa));

    (*obj).c1.c1.re = (float*) malloc(vol * sizeof(float));
    (*obj).c1.c1.im = (float*) malloc(vol * sizeof(float));
    (*obj).c1.c2.re = (float*) malloc(vol * sizeof(float));
    (*obj).c1.c2.im = (float*) malloc(vol * sizeof(float));
    (*obj).c1.c3.re = (float*) malloc(vol * sizeof(float));
    (*obj).c1.c3.im = (float*) malloc(vol * sizeof(float));
    (*obj).c2.c1.re = (float*) malloc(vol * sizeof(float));
    (*obj).c2.c1.im = (float*) malloc(vol * sizeof(float));
    (*obj).c2.c2.re = (float*) malloc(vol * sizeof(float));
    (*obj).c2.c2.im = (float*) malloc(vol * sizeof(float));
    (*obj).c2.c3.re = (float*) malloc(vol * sizeof(float));
    (*obj).c2.c3.im = (float*) malloc(vol * sizeof(float));
    (*obj).c3.c1.re = (float*) malloc(vol * sizeof(float));
    (*obj).c3.c1.im = (float*) malloc(vol * sizeof(float));
    (*obj).c3.c2.re = (float*) malloc(vol * sizeof(float));
    (*obj).c3.c2.im = (float*) malloc(vol * sizeof(float));
    (*obj).c3.c3.re = (float*) malloc(vol * sizeof(float));
    (*obj).c3.c3.im = (float*) malloc(vol * sizeof(float));
    (*obj).c4.c1.re = (float*) malloc(vol * sizeof(float));
    (*obj).c4.c1.im = (float*) malloc(vol * sizeof(float));
    (*obj).c4.c2.re = (float*) malloc(vol * sizeof(float));
    (*obj).c4.c2.im = (float*) malloc(vol * sizeof(float));
    (*obj).c4.c3.re = (float*) malloc(vol * sizeof(float));
    (*obj).c4.c3.im = (float*) malloc(vol * sizeof(float));

    return obj;
}

void destroy_spinor_soa(spinor_soa* obj)
{
    free((*obj).c1.c1.re);
    free((*obj).c1.c1.im);
    free((*obj).c1.c2.re);
    free((*obj).c1.c2.im);
    free((*obj).c1.c3.re);
    free((*obj).c1.c3.im);
    free((*obj).c2.c1.re);
    free((*obj).c2.c1.im);
    free((*obj).c2.c2.re);
    free((*obj).c2.c2.im);
    free((*obj).c2.c3.re);
    free((*obj).c2.c3.im);
    free((*obj).c3.c1.re);
    free((*obj).c3.c1.im);
    free((*obj).c3.c2.re);
    free((*obj).c3.c2.im);
    free((*obj).c3.c3.re);
    free((*obj).c3.c3.im);
    free((*obj).c4.c1.re);
    free((*obj).c4.c1.im);
    free((*obj).c4.c2.re);
    free((*obj).c4.c2.im);
    free((*obj).c4.c3.re);
    free((*obj).c4.c3.im);

    free(obj);
}

su3_soa* create_su3_soa(int vol)
{
    su3_soa* obj = (su3_soa*) malloc(sizeof(su3_soa));

    (*obj).c11.re = (float*) malloc(4 * vol * sizeof(float));
    (*obj).c11.im = (float*) malloc(4 * vol * sizeof(float));
    (*obj).c12.re = (float*) malloc(4 * vol * sizeof(float));
    (*obj).c12.im = (float*) malloc(4 * vol * sizeof(float));
    (*obj).c13.re = (float*) malloc(4 * vol * sizeof(float));
    (*obj).c13.im = (float*) malloc(4 * vol * sizeof(float));
    (*obj).c21.re = (float*) malloc(4 * vol * sizeof(float));
    (*obj).c21.im = (float*) malloc(4 * vol * sizeof(float));
    (*obj).c22.re = (float*) malloc(4 * vol * sizeof(float));
    (*obj).c22.im = (float*) malloc(4 * vol * sizeof(float));
    (*obj).c23.re = (float*) malloc(4 * vol * sizeof(float));
    (*obj).c23.im = (float*) malloc(4 * vol * sizeof(float));
    (*obj).c31.re = (float*) malloc(4 * vol * sizeof(float));
    (*obj).c31.im = (float*) malloc(4 * vol * sizeof(float));
    (*obj).c32.re = (float*) malloc(4 * vol * sizeof(float));
    (*obj).c32.im = (float*) malloc(4 * vol * sizeof(float));
    (*obj).c33.re = (float*) malloc(4 * vol * sizeof(float));
    (*obj).c33.im = (float*) malloc(4 * vol * sizeof(float));

    return obj;
}

void destroy_su3_soa(su3_soa* obj)
{
    free((*obj).c11.re);
    free((*obj).c11.im);
    free((*obj).c12.re);
    free((*obj).c12.im);
    free((*obj).c13.re);
    free((*obj).c13.im);
    free((*obj).c21.re);
    free((*obj).c21.im);
    free((*obj).c22.re);
    free((*obj).c22.im);
    free((*obj).c23.re);
    free((*obj).c23.im);
    free((*obj).c31.re);
    free((*obj).c31.im);
    free((*obj).c32.re);
    free((*obj).c32.im);
    free((*obj).c33.re);
    free((*obj).c33.im);

    free(obj);
}

void copy_pauli_aos2soa(pauli_soa* m_soa, pauli* m, int vol)
{
    int i, j, idx;

    idx = 0;
    for (i = 0; i < 2*vol; i += 2) {
        for (j = 0; j < 36; ++j) {
            (*m_soa).m1[j*vol + idx] = (*(m+i)).u[j];
        }
        idx++;
    }

    idx = 0;
    for (i = 1; i < 2*vol; i += 2) {
        for (j = 0; j < 36; ++j) {
            (*m_soa).m2[j*vol + idx] = (*(m+i)).u[j];
        }
        idx++;
    }
}

// void copy_pauli_soa2aos(pauli* m, pauli_soa* m_soa, int vol)
// {
//     int i, j, idx;
//
//     idx = 0;
//     for (i = 0; i < 2*vol; i += 2) {
//         for (j = 0; j < 36; ++j) {
//             (*(m+i)).u[j] = (*m_soa).m1[j*vol + idx];
//         }
//         idx++;
//     }
//
//     idx = 0;
//     for (i = 1; i < 2*vol; i += 2) {
//         for (j = 0; j < 36; ++j) {
//             (*(m+i)).u[j] = (*m_soa).m2[j*vol + idx];
//         }
//         idx++;
//     }
// }

void copy_spinor_aos2soa(spinor_soa* s_soa, spinor* s, int vol)
{
    for (int i = 0; i < vol; ++i) {
        (*s_soa).c1.c1.re[i] = (*(s+i)).c1.c1.re;
        (*s_soa).c1.c1.im[i] = (*(s+i)).c1.c1.im;
        (*s_soa).c1.c2.re[i] = (*(s+i)).c1.c2.re;
        (*s_soa).c1.c2.im[i] = (*(s+i)).c1.c2.im;
        (*s_soa).c1.c3.re[i] = (*(s+i)).c1.c3.re;
        (*s_soa).c1.c3.im[i] = (*(s+i)).c1.c3.im;
        (*s_soa).c2.c1.re[i] = (*(s+i)).c2.c1.re;
        (*s_soa).c2.c1.im[i] = (*(s+i)).c2.c1.im;
        (*s_soa).c2.c2.re[i] = (*(s+i)).c2.c2.re;
        (*s_soa).c2.c2.im[i] = (*(s+i)).c2.c2.im;
        (*s_soa).c2.c3.re[i] = (*(s+i)).c2.c3.re;
        (*s_soa).c2.c3.im[i] = (*(s+i)).c2.c3.im;
        (*s_soa).c3.c1.re[i] = (*(s+i)).c3.c1.re;
        (*s_soa).c3.c1.im[i] = (*(s+i)).c3.c1.im;
        (*s_soa).c3.c2.re[i] = (*(s+i)).c3.c2.re;
        (*s_soa).c3.c2.im[i] = (*(s+i)).c3.c2.im;
        (*s_soa).c3.c3.re[i] = (*(s+i)).c3.c3.re;
        (*s_soa).c3.c3.im[i] = (*(s+i)).c3.c3.im;
        (*s_soa).c4.c1.re[i] = (*(s+i)).c4.c1.re;
        (*s_soa).c4.c1.im[i] = (*(s+i)).c4.c1.im;
        (*s_soa).c4.c2.re[i] = (*(s+i)).c4.c2.re;
        (*s_soa).c4.c2.im[i] = (*(s+i)).c4.c2.im;
        (*s_soa).c4.c3.re[i] = (*(s+i)).c4.c3.re;
        (*s_soa).c4.c3.im[i] = (*(s+i)).c4.c3.im;
    }
}

void copy_spinor_soa2aos(spinor* s, spinor_soa* s_soa, int vol)
{
    for (int i = 0; i < vol; ++i) {
        (*(s+i)).c1.c1.re = (*s_soa).c1.c1.re[i];
        (*(s+i)).c1.c1.im = (*s_soa).c1.c1.im[i];
        (*(s+i)).c1.c2.re = (*s_soa).c1.c2.re[i];
        (*(s+i)).c1.c2.im = (*s_soa).c1.c2.im[i];
        (*(s+i)).c1.c3.re = (*s_soa).c1.c3.re[i];
        (*(s+i)).c1.c3.im = (*s_soa).c1.c3.im[i];
        (*(s+i)).c2.c1.re = (*s_soa).c2.c1.re[i];
        (*(s+i)).c2.c1.im = (*s_soa).c2.c1.im[i];
        (*(s+i)).c2.c2.re = (*s_soa).c2.c2.re[i];
        (*(s+i)).c2.c2.im = (*s_soa).c2.c2.im[i];
        (*(s+i)).c2.c3.re = (*s_soa).c2.c3.re[i];
        (*(s+i)).c2.c3.im = (*s_soa).c2.c3.im[i];
        (*(s+i)).c3.c1.re = (*s_soa).c3.c1.re[i];
        (*(s+i)).c3.c1.im = (*s_soa).c3.c1.im[i];
        (*(s+i)).c3.c2.re = (*s_soa).c3.c2.re[i];
        (*(s+i)).c3.c2.im = (*s_soa).c3.c2.im[i];
        (*(s+i)).c3.c3.re = (*s_soa).c3.c3.re[i];
        (*(s+i)).c3.c3.im = (*s_soa).c3.c3.im[i];
        (*(s+i)).c4.c1.re = (*s_soa).c4.c1.re[i];
        (*(s+i)).c4.c1.im = (*s_soa).c4.c1.im[i];
        (*(s+i)).c4.c2.re = (*s_soa).c4.c2.re[i];
        (*(s+i)).c4.c2.im = (*s_soa).c4.c2.im[i];
        (*(s+i)).c4.c3.re = (*s_soa).c4.c3.re[i];
        (*(s+i)).c4.c3.im = (*s_soa).c4.c3.im[i];
    }
}

void copy_su3_aos2soa(su3_soa* u_soa, su3* u, int vol)
{
    for (int i = 0; i < 4*vol; ++i) {
        (*u_soa).c11.re[i] = (*(u+i)).c11.re;
        (*u_soa).c11.im[i] = (*(u+i)).c11.im;
        (*u_soa).c12.re[i] = (*(u+i)).c12.re;
        (*u_soa).c12.im[i] = (*(u+i)).c12.im;
        (*u_soa).c13.re[i] = (*(u+i)).c13.re;
        (*u_soa).c13.im[i] = (*(u+i)).c13.im;
        (*u_soa).c21.re[i] = (*(u+i)).c21.re;
        (*u_soa).c21.im[i] = (*(u+i)).c21.im;
        (*u_soa).c22.re[i] = (*(u+i)).c22.re;
        (*u_soa).c22.im[i] = (*(u+i)).c22.im;
        (*u_soa).c23.re[i] = (*(u+i)).c23.re;
        (*u_soa).c23.im[i] = (*(u+i)).c23.im;
        (*u_soa).c31.re[i] = (*(u+i)).c31.re;
        (*u_soa).c31.im[i] = (*(u+i)).c31.im;
        (*u_soa).c32.re[i] = (*(u+i)).c32.re;
        (*u_soa).c32.im[i] = (*(u+i)).c32.im;
        (*u_soa).c33.re[i] = (*(u+i)).c33.re;
        (*u_soa).c33.im[i] = (*(u+i)).c33.im;
    }
}

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

extern "C"
void Dw_cuda_SoA(int VOLUME, su3 *u, spinor *s, spinor *r, pauli *m, int *piup, int *pidn)
{
    float mu, coe, ceo;
    float gamma_f, one_over_gammaf;

    mu = 0.0f;
    coe = -0.5f;
    ceo = -0.5f;

    gamma_f = 1.0f;
    one_over_gammaf = 1.0f;

    // Create structure of arrays
    pauli_soa *m_soa = create_pauli_soa(VOLUME);
    spinor_soa *s_soa = create_spinor_soa(VOLUME);
    spinor_soa *r_soa = create_spinor_soa(VOLUME);
    su3_soa *u_soa = create_su3_soa(VOLUME);

    // Copy data from AoS to SoA
    copy_pauli_aos2soa(m_soa, m, VOLUME);
    copy_spinor_aos2soa(s_soa, s, VOLUME);
    copy_su3_aos2soa(u_soa, u, VOLUME);

    // Allocate memory on device
    pauli_soa d_m_soa = allocPauli2Device(VOLUME);
    spinor_soa d_s_soa = allocSpinor2Device(VOLUME);
    spinor_soa d_r_soa = allocSpinor2Device(VOLUME);
    su3_soa d_u_soa = allocSu32Device(VOLUME);

    // Copy from host to device
    copyPauliHost2Device(d_m_soa, m_soa, VOLUME);
    copySpinorHost2Device(d_s_soa, s_soa, VOLUME);
    copySu3Host2Device(d_u_soa, u_soa, VOLUME);

    int block_size, grid_size;
    // Launch kernel on GPU
    block_size = 128;
    grid_size = ceil(VOLUME/(float)block_size);
    mulpauli_kernel<<<grid_size, block_size>>>(VOLUME, mu, d_s_soa, d_r_soa, d_m_soa);


    // Copy data from device to host
    cudaMemcpy((*r_soa).c1.c1.re, d_r_soa.c1.c1.re, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c1.c1.im, d_r_soa.c1.c1.im, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c1.c2.re, d_r_soa.c1.c2.re, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c1.c2.im, d_r_soa.c1.c2.im, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c1.c3.re, d_r_soa.c1.c3.re, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c1.c3.im, d_r_soa.c1.c3.im, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c2.c1.re, d_r_soa.c2.c1.re, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c2.c1.im, d_r_soa.c2.c1.im, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c2.c2.re, d_r_soa.c2.c2.re, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c2.c2.im, d_r_soa.c2.c2.im, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c2.c3.re, d_r_soa.c2.c3.re, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c2.c3.im, d_r_soa.c2.c3.im, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c3.c1.re, d_r_soa.c3.c1.re, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c3.c1.im, d_r_soa.c3.c1.im, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c3.c2.re, d_r_soa.c3.c2.re, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c3.c2.im, d_r_soa.c3.c2.im, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c3.c3.re, d_r_soa.c3.c3.re, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c3.c3.im, d_r_soa.c3.c3.im, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c4.c1.re, d_r_soa.c4.c1.re, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c4.c1.im, d_r_soa.c4.c1.im, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c4.c2.re, d_r_soa.c4.c2.re, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c4.c2.im, d_r_soa.c4.c2.im, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c4.c3.re, d_r_soa.c4.c3.re, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((*r_soa).c4.c3.im, d_r_soa.c4.c3.im, VOLUME * sizeof(float), cudaMemcpyDeviceToHost);


    // Free GPU memory
    // Create destroy functions


    // Convert from SoA to AoS
    copy_spinor_soa2aos(r, r_soa, VOLUME);

    // Free SoA
    destroy_pauli_soa(m_soa);
    destroy_spinor_soa(s_soa);
    destroy_spinor_soa(r_soa);
    destroy_su3_soa(u_soa);
}
