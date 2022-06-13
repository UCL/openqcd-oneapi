#ifndef SU3_H
#define SU3_H

typedef struct dpct_type_586024
{
  float re, im;
} complex;

typedef struct dpct_type_955565
{
  complex c1, c2, c3;
} su3_vector;

typedef struct dpct_type_113657
{
  complex c11, c12, c13, c21, c22, c23, c31, c32, c33;
} su3;

typedef struct dpct_type_129040
{
  su3_vector c1, c2;
} weyl;

typedef struct dpct_type_157303
{
  su3_vector c1, c2, c3, c4;
} spinor;

typedef struct dpct_type_466052
{
  float u[36];
} pauli;

typedef union dpct_type_143653
{
  spinor s;
  weyl w[2];
} spin_t;

typedef struct dpct_type_491149
{
  float *re;
  float *im;
} complex_soa;

typedef struct dpct_type_175281
{
  complex_soa c1, c2, c3;
} su3_vector_soa;

typedef struct dpct_type_120970
{
  complex_soa c11, c12, c13, c21, c22, c23, c31, c32, c33;
} su3_soa;

typedef struct dpct_type_637720
{
  su3_vector_soa c1, c2, c3, c4;
} spinor_soa;

typedef struct dpct_type_784313
{
  float *m1; // For A+
  float *m2; // For A-
} pauli_soa;

#endif
