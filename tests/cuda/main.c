#include "dw_cuda.h"
#include "dw_cuda_soa.h"
#include "su3.h"
#include "macros.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


float coe, ceo;
float gamma_f, one_over_gammaf;

static weyl rt;
spin_t rs;


static void doe(int *piup, int *pidn, su3 *u, spinor *pk)
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
}

static void deo(int *piup, int *pidn, su3 *u, spinor *pl)
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

void mul_pauli(float mu, pauli const *m, weyl const *s, weyl *r)
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

void mul_pauli2(float mu, pauli const *m, spinor const *s, spinor *r)
{
    spin_t const *ps;
    spin_t *pr;

    ps = (spin_t const *)(s);
    pr = (spin_t *)(r);

    mul_pauli(mu, m, (*ps).w, (*pr).w);
    mul_pauli(-mu, m + 1, (*ps).w + 1, (*pr).w + 1);
}

void Dw(int VOLUME, float mu, spinor *s, spinor *r, char *SIZE)
{
    char buffer[100];
    size_t result;
    FILE *ptr;

    // Read piup
    int *piup;
    piup = (int*) malloc(2 * VOLUME * sizeof(int));
    buffer[0] = '\0';
    strcat(buffer, SIZE);
    strcat(buffer, "/dataBefore/piup.bin");
    ptr = fopen(buffer,"rb");
    result = fread(piup, sizeof(int), 2 * VOLUME, ptr);
    fclose(ptr);

    // Read pidn
    int *pidn;
    pidn = (int*) malloc(2 * VOLUME * sizeof(int));
    buffer[0] = '\0';
    strcat(buffer, SIZE);
    strcat(buffer, "/dataBefore/pidn.bin");
    ptr = fopen(buffer,"rb");
    result = fread(pidn, sizeof(int), 2 * VOLUME, ptr);
    fclose(ptr);

    // Read u
    su3 *u;
    u = (su3*) malloc(4 * VOLUME * sizeof(su3));
    buffer[0] = '\0';
    strcat(buffer, SIZE);
    strcat(buffer, "/dataBefore/u.bin");
    ptr = fopen(buffer,"rb");
    result = fread(u, sizeof(su3), 4 * VOLUME, ptr);
    fclose(ptr);

    // Read m
    pauli *m;
    m = (pauli*) malloc(VOLUME * sizeof(pauli));
    buffer[0] = '\0';
    strcat(buffer, SIZE);
    strcat(buffer, "/dataBefore/m.bin");
    ptr = fopen(buffer,"rb");
    result = fread(m, sizeof(pauli), VOLUME, ptr);
    fclose(ptr);

    su3 *um;
    spin_t *so, *ro;

    um = u + 4 * VOLUME;

    so = (spin_t *)(s + (VOLUME / 2));
    ro = (spin_t *)(r + (VOLUME / 2));

    coe = -0.5f;
    ceo = -0.5f;

    gamma_f = 1.0f;
    one_over_gammaf = 1.0f;

    for (; u < um; u += 8) {
        doe(piup, pidn, u, s);

        mul_pauli2(mu, m, &((*so).s), &((*ro).s));

        _vector_add_assign((*ro).s.c1, rs.s.c1);
        _vector_add_assign((*ro).s.c2, rs.s.c2);
        _vector_add_assign((*ro).s.c3, rs.s.c3);
        _vector_add_assign((*ro).s.c4, rs.s.c4);
        rs = (*so);

        deo(piup, pidn, u, r);

        piup += 4;
        pidn += 4;
        so += 1;
        ro += 1;
        m += 2;
    }
}

int main(int argc, char *argv[])
{
    char buffer[100];
    size_t result;
    FILE *ptr;

    // Read VOLUME
    int VOLUME;
    buffer[0] = '\0';
    strcat(buffer, argv[1]);
    strcat(buffer, "/dataBefore/VOLUME.bin");
    ptr = fopen(buffer,"rb");
    result = fread(&VOLUME, sizeof(int), 1, ptr);
    fclose(ptr);

    // Read mu
    float mu;
    buffer[0] = '\0';
    strcat(buffer, argv[1]);
    strcat(buffer, "/dataBefore/mu.bin");
    ptr = fopen(buffer,"rb");
    result = fread(&mu, sizeof(float), 1, ptr);
    fclose(ptr);

    // Read s
    spinor *s;
    s = (spinor*) malloc(VOLUME * sizeof(spinor));
    buffer[0] = '\0';
    strcat(buffer, argv[1]);
    strcat(buffer, "/dataBefore/s.bin");
    ptr = fopen(buffer,"rb");
    result = fread(s, sizeof(spinor), VOLUME, ptr);
    fclose(ptr);

    // Read r
    spinor *r;
    r = (spinor*) malloc(VOLUME * sizeof(spinor));
    buffer[0] = '\0';
    strcat(buffer, argv[1]);
    strcat(buffer, "/dataBefore/r.bin");
    ptr = fopen(buffer,"rb");
    result = fread(r, sizeof(spinor), VOLUME, ptr);
    fclose(ptr);

    // Call Dw()
    printf("---------------------CPU--------------------\n");
    Dw(VOLUME, mu, s, r, argv[1]);

    // Read r_after from disk
    spinor *r_after;
    r_after = (spinor*) malloc(VOLUME * sizeof(spinor));
    buffer[0] = '\0';
    strcat(buffer, argv[1]);
    strcat(buffer, "/dataAfter/r.bin");
    ptr = fopen(buffer,"rb");
    result = fread(r_after, sizeof(spinor), VOLUME, ptr);
    fclose(ptr);

    // Compare value from disk with r in memory after the execution of Dw()
    int ret = memcmp(r, r_after, VOLUME * sizeof(spinor));
    if (ret == 0) {
        printf("Values in spinor r are correct after calling Dw()\n");
    }
    else {
        printf("Values in spinor r are incorrect after calling Dw()\n");
    }


    // Call Dw_cuda()
    printf("\n---------------------CUDA---------------------\n");
    Dw_cuda(r, argv[1]);

    // Call Dw_cuda_SoA()
    printf("\n------------CUDA (SoA) Unfinished-------------\n");
    Dw_cuda_SoA(r, argv[1]);

    return 0;
}
