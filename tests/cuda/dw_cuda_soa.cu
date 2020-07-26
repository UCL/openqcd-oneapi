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

    (*obj).r1 = (float*) malloc(36 * (vol/2) * sizeof(float));
    (*obj).r2 = (float*) malloc(36 * (vol/2) * sizeof(float));

    return obj;
}

void destroy_pauli_soa(pauli_soa* obj)
{
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
            (*m_soa).r1[j*(vol/2) + idx] = (*(m+i)).u[j];
        }
        idx++;
    }

    idx = 0;
    for (i = 1; i < vol; i += 2) {
        for (j = 0; j < 36; ++j) {
            (*m_soa).r2[j*(vol/2) + idx] = (*(m+i)).u[j];
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
            (*(m+i)).u[j] = (*m_soa).r1[j*(vol/2) + idx];
        }
        idx++;
    }

    idx = 0;
    for (i = 1; i < vol; i += 2) {
        for (j = 0; j < 36; ++j) {
            (*(m+i)).u[j] = (*m_soa).r2[j*(vol/2) + idx];
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

extern "C" __global__
void Dw_cuda_kernel_mulpauli_soa(int VOLUME, float mu, spinor_soa *s, spinor_soa *r, pauli_soa *m)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= VOLUME/2) return;

    int volhalf = VOLUME/2;
    int sidx = volhalf + idx;

    (*r).c1.c1.re[sidx] =
      (*m).r1[0*volhalf+idx]  * (*s).c1.c1.re[sidx] - mu                      * (*s).c1.c1.im[sidx] + (*m).r1[6*volhalf+idx]  * (*s).c1.c2.re[sidx] -
      (*m).r1[7*volhalf+idx]  * (*s).c1.c2.im[sidx] + (*m).r1[8*volhalf+idx]  * (*s).c1.c3.re[sidx] - (*m).r1[9*volhalf+idx]  * (*s).c1.c3.im[sidx] +
      (*m).r1[10*volhalf+idx] * (*s).c2.c1.re[sidx] - (*m).r1[11*volhalf+idx] * (*s).c2.c1.im[sidx] + (*m).r1[12*volhalf+idx] * (*s).c2.c2.re[sidx] -
      (*m).r1[13*volhalf+idx] * (*s).c2.c2.im[sidx] + (*m).r1[14*volhalf+idx] * (*s).c2.c3.re[sidx] - (*m).r1[15*volhalf+idx] * (*s).c2.c3.im[sidx];


    // the rest and then...





    (*r).c3.c1.re[sidx] =
      (*m).r1[0*volhalf+idx]  * (*s).c3.c1.re[sidx] - mu                      * (*s).c3.c1.im[sidx] + (*m).r1[6*volhalf+idx]  * (*s).c3.c2.re[sidx] -
      (*m).r1[7*volhalf+idx]  * (*s).c3.c2.im[sidx] + (*m).r1[8*volhalf+idx]  * (*s).c3.c3.re[sidx] - (*m).r1[9*volhalf+idx]  * (*s).c3.c3.im[sidx] +
      (*m).r1[10*volhalf+idx] * (*s).c4.c1.re[sidx] - (*m).r1[11*volhalf+idx] * (*s).c4.c1.im[sidx] + (*m).r1[12*volhalf+idx] * (*s).c4.c2.re[sidx] -
      (*m).r1[13*volhalf+idx] * (*s).c4.c2.im[sidx] + (*m).r1[14*volhalf+idx] * (*s).c4.c3.re[sidx] - (*m).r1[15*volhalf+idx] * (*s).c4.c3.im[sidx];

      // and the rest
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
