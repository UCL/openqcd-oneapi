// #include "dw_cuda_soa.h"
#include "su3.h"
#include "macros.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


pauli_soa* create_pauli_soa(int vol)
{
    pauli_soa* obj = (pauli_soa*) malloc(sizeof(pauli_soa));

    (*obj).l1 = (float*) malloc(36 * (vol/2) * sizeof(float));
    (*obj).l2 = (float*) malloc(36 * (vol/2) * sizeof(float));
    (*obj).r1 = (float*) malloc(36 * (vol/2) * sizeof(float));
    (*obj).r2 = (float*) malloc(36 * (vol/2) * sizeof(float));

    return obj;
}

void destroy_pauli_soa(pauli_soa* obj)
{
    free((*obj).l1);
    free((*obj).l2);
    free((*obj).r1);
    free((*obj).r2);

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

void copy_pauli_aos2soa(pauli* m, pauli_soa* m_soa, int vol)
{
    int i, j, idx;

    idx = 0;
    for (i = 0; i < vol; i += 2) {
        for (j = 0; j < 36; ++j) {
            (*m_soa).l1[j*(vol/2) + idx] = (*(m+i)).u[j];
            (*m_soa).r1[j*(vol/2) + idx] = (*(m+vol+i)).u[j];
        }
        idx++;
    }

    idx = 0;
    for (i = 1; i < vol; i += 2) {
        for (j = 0; j < 36; ++j) {
            (*m_soa).l2[j*(vol/2) + idx] = (*(m+i)).u[j];
            (*m_soa).r2[j*(vol/2) + idx] = (*(m+vol+i)).u[j];
        }
        idx++;
    }
}

void copy_pauli_soa2aos(pauli_soa* m_soa, pauli* m, int vol)
{
    int i, j, idx;

    idx = 0;
    for (i = 0; i < vol; i += 2) {
        for (j = 0; j < 36; ++j) {
            (*(m+i)).u[j]     = (*m_soa).l1[j*(vol/2) + idx];
            (*(m+vol+i)).u[j] = (*m_soa).r1[j*(vol/2) + idx];
        }
        idx++;
    }

    idx = 0;
    for (i = 1; i < vol; i += 2) {
        for (j = 0; j < 36; ++j) {
            (*(m+i)).u[j]     = (*m_soa).l2[j*(vol/2) + idx];
            (*(m+vol+i)).u[j] = (*m_soa).r2[j*(vol/2) + idx];
        }
        idx++;
    }
}

void copy_spinor_aos2soa(spinor* s, spinor_soa* s_soa, int vol)
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

void copy_spinor_soa2aos(spinor_soa* s_soa, spinor* s, int vol)
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

__device__
static void mul_pauli(int idx, int sidx, int halfvol, float mu,
                      spinor_soa const *s, spinor_soa *r,
                      float const *m1, float const *m2)
{
    float u[36];
    halfspinor sloc;

    sloc.c1.c1.re = (*s).c1.c1.re[sidx];
    sloc.c1.c1.im = (*s).c1.c1.im[sidx];
    sloc.c1.c2.re = (*s).c1.c2.re[sidx];
    sloc.c1.c2.im = (*s).c1.c2.im[sidx];
    sloc.c1.c3.re = (*s).c1.c3.re[sidx];
    sloc.c1.c3.im = (*s).c1.c3.im[sidx];
    sloc.c2.c1.re = (*s).c2.c1.re[sidx];
    sloc.c2.c1.im = (*s).c2.c1.im[sidx];
    sloc.c2.c2.re = (*s).c2.c2.re[sidx];
    sloc.c2.c2.im = (*s).c2.c2.im[sidx];
    sloc.c2.c3.re = (*s).c2.c3.re[sidx];
    sloc.c2.c3.im = (*s).c2.c3.im[sidx];

    for (int i = 0; i < halfvol; ++i) {
        u[i] = m1[i*halfvol + idx]
    }

    (*r).c1.c1.re[sidx] =
      u[0]  * sloc.c1.c1.re - mu    * sloc.c1.c1.im + u[6]  * sloc.c1.c2.re -
      u[7]  * sloc.c1.c2.im + u[8]  * sloc.c1.c3.re - u[9]  * sloc.c1.c3.im +
      u[10] * sloc.c2.c1.re - u[11] * sloc.c2.c1.im + u[12] * sloc.c2.c2.re -
      u[13] * sloc.c2.c2.im + u[14] * sloc.c2.c3.re - u[15] * sloc.c2.c3.im;

    (*r).c1.c1.im[sidx] =
      u[0]  * sloc.c1.c1.im + mu    * sloc.c1.c1.re + u[6]  * sloc.c1.c2.im +
      u[7]  * sloc.c1.c2.re + u[8]  * sloc.c1.c3.im + u[9]  * sloc.c1.c3.re +
      u[10] * sloc.c2.c1.im + u[11] * sloc.c2.c1.re + u[12] * sloc.c2.c2.im +
      u[13] * sloc.c2.c2.re + u[14] * sloc.c2.c3.im + u[15] * sloc.c2.c3.re;

    (*r).c1.c2.re[sidx] =
      u[6]  * sloc.c1.c1.re + u[7]  * sloc.c1.c1.im + u[1]  * sloc.c1.c2.re -
      mu    * sloc.c1.c2.im + u[16] * sloc.c1.c3.re - u[17] * sloc.c1.c3.im +
      u[18] * sloc.c2.c1.re - u[19] * sloc.c2.c1.im + u[20] * sloc.c2.c2.re -
      u[21] * sloc.c2.c2.im + u[22] * sloc.c2.c3.re - u[23] * sloc.c2.c3.im;

    (*r).c1.c2.im[sidx] =
      u[6]  * sloc.c1.c1.im - u[7]  * sloc.c1.c1.re + u[1]  * sloc.c1.c2.im +
      mu    * sloc.c1.c2.re + u[16] * sloc.c1.c3.im + u[17] * sloc.c1.c3.re +
      u[18] * sloc.c2.c1.im + u[19] * sloc.c2.c1.re + u[20] * sloc.c2.c2.im +
      u[21] * sloc.c2.c2.re + u[22] * sloc.c2.c3.im + u[23] * sloc.c2.c3.re;

    (*r).c1.c3.re[sidx] =
      u[8]  * sloc.c1.c1.re + u[9]  * sloc.c1.c1.im + u[16] * sloc.c1.c2.re +
      u[17] * sloc.c1.c2.im + u[2]  * sloc.c1.c3.re - mu    * sloc.c1.c3.im +
      u[24] * sloc.c2.c1.re - u[25] * sloc.c2.c1.im + u[26] * sloc.c2.c2.re -
      u[27] * sloc.c2.c2.im + u[28] * sloc.c2.c3.re - u[29] * sloc.c2.c3.im;

    (*r).c1.c3.im[sidx] =
      u[8]  * sloc.c1.c1.im - u[9]  * sloc.c1.c1.re + u[16] * sloc.c1.c2.im -
      u[17] * sloc.c1.c2.re + u[2]  * sloc.c1.c3.im + mu    * sloc.c1.c3.re +
      u[24] * sloc.c2.c1.im + u[25] * sloc.c2.c1.re + u[26] * sloc.c2.c2.im +
      u[27] * sloc.c2.c2.re + u[28] * sloc.c2.c3.im + u[29] * sloc.c2.c3.re;

    (*r).c2.c1.re[sidx] =
      u[10] * sloc.c1.c1.re + u[11] * sloc.c1.c1.im + u[18] * sloc.c1.c2.re +
      u[19] * sloc.c1.c2.im + u[24] * sloc.c1.c3.re + u[25] * sloc.c1.c3.im +
      u[3]  * sloc.c2.c1.re - mu * sloc.c2.c1.im    + u[30] * sloc.c2.c2.re -
      u[31] * sloc.c2.c2.im + u[32] * sloc.c2.c3.re - u[33] * sloc.c2.c3.im;

    (*r).c2.c1.im[sidx] =
      u[10] * sloc.c1.c1.im - u[11] * sloc.c1.c1.re + u[18] * sloc.c1.c2.im -
      u[19] * sloc.c1.c2.re + u[24] * sloc.c1.c3.im - u[25] * sloc.c1.c3.re +
      u[3]  * sloc.c2.c1.im + mu    * sloc.c2.c1.re + u[30] * sloc.c2.c2.im +
      u[31] * sloc.c2.c2.re + u[32] * sloc.c2.c3.im + u[33] * sloc.c2.c3.re;

    (*r).c2.c2.re[sidx] =
      u[12] * sloc.c1.c1.re + u[13] * sloc.c1.c1.im + u[20] * sloc.c1.c2.re +
      u[21] * sloc.c1.c2.im + u[26] * sloc.c1.c3.re + u[27] * sloc.c1.c3.im +
      u[30] * sloc.c2.c1.re + u[31] * sloc.c2.c1.im + u[4]  * sloc.c2.c2.re -
      mu    * sloc.c2.c2.im + u[34] * sloc.c2.c3.re - u[35] * sloc.c2.c3.im;

    (*r).c2.c2.im[sidx] =
      u[12] * sloc.c1.c1.im - u[13] * sloc.c1.c1.re + u[20] * sloc.c1.c2.im -
      u[21] * sloc.c1.c2.re + u[26] * sloc.c1.c3.im - u[27] * sloc.c1.c3.re +
      u[30] * sloc.c2.c1.im - u[31] * sloc.c2.c1.re + u[4]  * sloc.c2.c2.im +
      mu    * sloc.c2.c2.re + u[34] * sloc.c2.c3.im + u[35] * sloc.c2.c3.re;

    (*r).c2.c3.re[sidx] =
      u[14] * sloc.c1.c1.re + u[15] * sloc.c1.c1.im + u[22] * sloc.c1.c2.re +
      u[23] * sloc.c1.c2.im + u[28] * sloc.c1.c3.re + u[29] * sloc.c1.c3.im +
      u[32] * sloc.c2.c1.re + u[33] * sloc.c2.c1.im + u[34] * sloc.c2.c2.re +
      u[35] * sloc.c2.c2.im + u[5]  * sloc.c2.c3.re - mu    * sloc.c2.c3.im;

    (*r).c2.c3.im[sidx] =
      u[14] * sloc.c1.c1.im - u[15] * sloc.c1.c1.re + u[22] * sloc.c1.c2.im -
      u[23] * sloc.c1.c2.re + u[28] * sloc.c1.c3.im - u[29] * sloc.c1.c3.re +
      u[32] * sloc.c2.c1.im - u[33] * sloc.c2.c1.re + u[34] * sloc.c2.c2.im -
      u[35] * sloc.c2.c2.re + u[5]  * sloc.c2.c3.im + mu    * sloc.c2.c3.re;


      sloc.c1.c1.re = (*s).c3.c1.re[sidx];
      sloc.c1.c1.im = (*s).c3.c1.im[sidx];
      sloc.c1.c2.re = (*s).c3.c2.re[sidx];
      sloc.c1.c2.im = (*s).c3.c2.im[sidx];
      sloc.c1.c3.re = (*s).c3.c3.re[sidx];
      sloc.c1.c3.im = (*s).c3.c3.im[sidx];
      sloc.c2.c1.re = (*s).c4.c1.re[sidx];
      sloc.c2.c1.im = (*s).c4.c1.im[sidx];
      sloc.c2.c2.re = (*s).c4.c2.re[sidx];
      sloc.c2.c2.im = (*s).c4.c2.im[sidx];
      sloc.c2.c3.re = (*s).c4.c3.re[sidx];
      sloc.c2.c3.im = (*s).c4.c3.im[sidx];

      for (int i = 0; i < halfvol; ++i) {
          u[i] = m2[i*halfvol + idx]
      }

      (*r).c3.c1.re[sidx] =
        u[0]  * sloc.c1.c1.re - mu    * sloc.c1.c1.im + u[6]  * sloc.c1.c2.re -
        u[7]  * sloc.c1.c2.im + u[8]  * sloc.c1.c3.re - u[9]  * sloc.c1.c3.im +
        u[10] * sloc.c2.c1.re - u[11] * sloc.c2.c1.im + u[12] * sloc.c2.c2.re -
        u[13] * sloc.c2.c2.im + u[14] * sloc.c2.c3.re - u[15] * sloc.c2.c3.im;

      (*r).c3.c1.im[sidx] =
        u[0]  * sloc.c1.c1.im + mu    * sloc.c1.c1.re + u[6]  * sloc.c1.c2.im +
        u[7]  * sloc.c1.c2.re + u[8]  * sloc.c1.c3.im + u[9]  * sloc.c1.c3.re +
        u[10] * sloc.c2.c1.im + u[11] * sloc.c2.c1.re + u[12] * sloc.c2.c2.im +
        u[13] * sloc.c2.c2.re + u[14] * sloc.c2.c3.im + u[15] * sloc.c2.c3.re;

      (*r).c3.c2.re[sidx] =
        u[6]  * sloc.c1.c1.re + u[7]  * sloc.c1.c1.im + u[1]  * sloc.c1.c2.re -
        mu    * sloc.c1.c2.im + u[16] * sloc.c1.c3.re - u[17] * sloc.c1.c3.im +
        u[18] * sloc.c2.c1.re - u[19] * sloc.c2.c1.im + u[20] * sloc.c2.c2.re -
        u[21] * sloc.c2.c2.im + u[22] * sloc.c2.c3.re - u[23] * sloc.c2.c3.im;

      (*r).c3.c2.im[sidx] =
        u[6]  * sloc.c1.c1.im - u[7]  * sloc.c1.c1.re + u[1]  * sloc.c1.c2.im +
        mu    * sloc.c1.c2.re + u[16] * sloc.c1.c3.im + u[17] * sloc.c1.c3.re +
        u[18] * sloc.c2.c1.im + u[19] * sloc.c2.c1.re + u[20] * sloc.c2.c2.im +
        u[21] * sloc.c2.c2.re + u[22] * sloc.c2.c3.im + u[23] * sloc.c2.c3.re;

      (*r).c3.c3.re[sidx] =
        u[8]  * sloc.c1.c1.re + u[9]  * sloc.c1.c1.im + u[16] * sloc.c1.c2.re +
        u[17] * sloc.c1.c2.im + u[2]  * sloc.c1.c3.re - mu    * sloc.c1.c3.im +
        u[24] * sloc.c2.c1.re - u[25] * sloc.c2.c1.im + u[26] * sloc.c2.c2.re -
        u[27] * sloc.c2.c2.im + u[28] * sloc.c2.c3.re - u[29] * sloc.c2.c3.im;

      (*r).c3.c3.im[sidx] =
        u[8]  * sloc.c1.c1.im - u[9]  * sloc.c1.c1.re + u[16] * sloc.c1.c2.im -
        u[17] * sloc.c1.c2.re + u[2]  * sloc.c1.c3.im + mu    * sloc.c1.c3.re +
        u[24] * sloc.c2.c1.im + u[25] * sloc.c2.c1.re + u[26] * sloc.c2.c2.im +
        u[27] * sloc.c2.c2.re + u[28] * sloc.c2.c3.im + u[29] * sloc.c2.c3.re;

      (*r).c4.c1.re[sidx] =
        u[10] * sloc.c1.c1.re + u[11] * sloc.c1.c1.im + u[18] * sloc.c1.c2.re +
        u[19] * sloc.c1.c2.im + u[24] * sloc.c1.c3.re + u[25] * sloc.c1.c3.im +
        u[3]  * sloc.c2.c1.re - mu * sloc.c2.c1.im    + u[30] * sloc.c2.c2.re -
        u[31] * sloc.c2.c2.im + u[32] * sloc.c2.c3.re - u[33] * sloc.c2.c3.im;

      (*r).c4.c1.im[sidx] =
        u[10] * sloc.c1.c1.im - u[11] * sloc.c1.c1.re + u[18] * sloc.c1.c2.im -
        u[19] * sloc.c1.c2.re + u[24] * sloc.c1.c3.im - u[25] * sloc.c1.c3.re +
        u[3]  * sloc.c2.c1.im + mu    * sloc.c2.c1.re + u[30] * sloc.c2.c2.im +
        u[31] * sloc.c2.c2.re + u[32] * sloc.c2.c3.im + u[33] * sloc.c2.c3.re;

      (*r).c4.c2.re[sidx] =
        u[12] * sloc.c1.c1.re + u[13] * sloc.c1.c1.im + u[20] * sloc.c1.c2.re +
        u[21] * sloc.c1.c2.im + u[26] * sloc.c1.c3.re + u[27] * sloc.c1.c3.im +
        u[30] * sloc.c2.c1.re + u[31] * sloc.c2.c1.im + u[4]  * sloc.c2.c2.re -
        mu    * sloc.c2.c2.im + u[34] * sloc.c2.c3.re - u[35] * sloc.c2.c3.im;

      (*r).c4.c2.im[sidx] =
        u[12] * sloc.c1.c1.im - u[13] * sloc.c1.c1.re + u[20] * sloc.c1.c2.im -
        u[21] * sloc.c1.c2.re + u[26] * sloc.c1.c3.im - u[27] * sloc.c1.c3.re +
        u[30] * sloc.c2.c1.im - u[31] * sloc.c2.c1.re + u[4]  * sloc.c2.c2.im +
        mu    * sloc.c2.c2.re + u[34] * sloc.c2.c3.im + u[35] * sloc.c2.c3.re;

      (*r).c4.c3.re[sidx] =
        u[14] * sloc.c1.c1.re + u[15] * sloc.c1.c1.im + u[22] * sloc.c1.c2.re +
        u[23] * sloc.c1.c2.im + u[28] * sloc.c1.c3.re + u[29] * sloc.c1.c3.im +
        u[32] * sloc.c2.c1.re + u[33] * sloc.c2.c1.im + u[34] * sloc.c2.c2.re +
        u[35] * sloc.c2.c2.im + u[5]  * sloc.c2.c3.re - mu    * sloc.c2.c3.im;

      (*r).c4.c3.im[sidx] =
        u[14] * sloc.c1.c1.im - u[15] * sloc.c1.c1.re + u[22] * sloc.c1.c2.im -
        u[23] * sloc.c1.c2.re + u[28] * sloc.c1.c3.im - u[29] * sloc.c1.c3.re +
        u[32] * sloc.c2.c1.im - u[33] * sloc.c2.c1.re + u[34] * sloc.c2.c2.im -
        u[35] * sloc.c2.c2.re + u[5]  * sloc.c2.c3.im + mu    * sloc.c2.c3.re;
}

extern "C" __global__
void Dw_cuda_kernel_mulpauli_soa(int VOLUME, float mu, spinor_soa *s, spinor_soa *r, pauli_soa *m)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= VOLUME/2) return;

    int sidx;
    int halfvol = VOLUME/2;

    sidx = idx;
    mul_pauli(idx, sidx, halfvol, mu, s, r, &((*m).l1), &((*m).l2))

    sidx = halfvol + idx;
    mul_pauli(idx, sidx, halfvol, mu, s, r, &((*m).r1), &((*m).r2))
}

extern "C"
void Dw_cuda_SoA(spinor *r_cpu, char *SIZE)
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


    // Create structure of arrays
    pauli_soa *m_soa = create_pauli_soa(VOLUME);
    spinor_soa *s_soa = create_spinor_soa(VOLUME);
    spinor_soa *r_soa = create_spinor_soa(VOLUME);

    // Copy data from AoS to SoA
    copy_pauli_aos2soa(m, m_soa, VOLUME);
    copy_spinor_aos2soa(s, s_soa, VOLUME);
    copy_spinor_aos2soa(r, r_soa, VOLUME);

    // Copy data from host to device


    // Launch kernel on GPU
    int block_size = 128;
    int grid_size = ceil((VOLUME/2.0)/(float)block_size);
    Dw_cuda_kernel_mulpauli_soa<<<grid_size, block_size>>>(VOLUME, mu, d_s_soa, d_r_soa, d_m_soa);



    // Copy data from device to host


    // Free GPU memory


    // Free SoA
    destroy_pauli_soa(m_soa);
    destroy_spinor_soa(s_soa);
    destroy_spinor_soa(r_soa);


}
