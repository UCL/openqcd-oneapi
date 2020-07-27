#ifndef SU3_H
#define SU3_H

typedef struct
{
    float re, im;
} complex;

typedef struct
{
    complex c1, c2, c3;
} su3_vector;

typedef struct
{
    complex c11, c12, c13, c21, c22, c23, c31, c32, c33;
} su3;

typedef struct
{
    su3_vector c1, c2;
} weyl;

typedef struct
{
    su3_vector c1, c2, c3, c4;
} spinor;

typedef struct
{
    float u[36];
} pauli;

typedef union
{
  spinor s;
  weyl w[2];
} spin_t;


typedef struct
{
    float *re;
    float *im;
} complex_soa;

typedef struct
{
    complex_soa c1, c2, c3;
} su3_vector_soa;

typedef struct
{
    su3_vector_soa c1, c2, c3, c4;
} spinor_soa;

typedef struct
{
    float *l1;
    float *l2;
    float *r1;
    float *r2;
} pauli_soa;

#endif
