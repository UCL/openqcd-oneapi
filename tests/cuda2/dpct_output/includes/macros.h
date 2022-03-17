#ifndef MACROS_H
#define MACROS_H

#define _vector_mul_assign(r, c)                                               \
  (r).c1.re *= (c);                                                            \
  (r).c1.im *= (c);                                                            \
  (r).c2.re *= (c);                                                            \
  (r).c2.im *= (c);                                                            \
  (r).c3.re *= (c);                                                            \
  (r).c3.im *= (c)

#define _vector_add(r, s1, s2)                                                 \
  (r).c1.re = (s1).c1.re + (s2).c1.re;                                         \
  (r).c1.im = (s1).c1.im + (s2).c1.im;                                         \
  (r).c2.re = (s1).c2.re + (s2).c2.re;                                         \
  (r).c2.im = (s1).c2.im + (s2).c2.im;                                         \
  (r).c3.re = (s1).c3.re + (s2).c3.re;                                         \
  (r).c3.im = (s1).c3.im + (s2).c3.im

#define _vector_sub(r, s1, s2)                                                 \
  (r).c1.re = (s1).c1.re - (s2).c1.re;                                         \
  (r).c1.im = (s1).c1.im - (s2).c1.im;                                         \
  (r).c2.re = (s1).c2.re - (s2).c2.re;                                         \
  (r).c2.im = (s1).c2.im - (s2).c2.im;                                         \
  (r).c3.re = (s1).c3.re - (s2).c3.re;                                         \
  (r).c3.im = (s1).c3.im - (s2).c3.im

#define _vector_i_add(r, s1, s2)                                               \
  (r).c1.re = (s1).c1.re - (s2).c1.im;                                         \
  (r).c1.im = (s1).c1.im + (s2).c1.re;                                         \
  (r).c2.re = (s1).c2.re - (s2).c2.im;                                         \
  (r).c2.im = (s1).c2.im + (s2).c2.re;                                         \
  (r).c3.re = (s1).c3.re - (s2).c3.im;                                         \
  (r).c3.im = (s1).c3.im + (s2).c3.re

#define _vector_i_sub(r, s1, s2)                                               \
  (r).c1.re = (s1).c1.re + (s2).c1.im;                                         \
  (r).c1.im = (s1).c1.im - (s2).c1.re;                                         \
  (r).c2.re = (s1).c2.re + (s2).c2.im;                                         \
  (r).c2.im = (s1).c2.im - (s2).c2.re;                                         \
  (r).c3.re = (s1).c3.re + (s2).c3.im;                                         \
  (r).c3.im = (s1).c3.im - (s2).c3.re

#define _vector_add_assign(r, s)                                               \
  (r).c1.re += (s).c1.re;                                                      \
  (r).c1.im += (s).c1.im;                                                      \
  (r).c2.re += (s).c2.re;                                                      \
  (r).c2.im += (s).c2.im;                                                      \
  (r).c3.re += (s).c3.re;                                                      \
  (r).c3.im += (s).c3.im

#define _vector_sub_assign(r, s)                                               \
  (r).c1.re -= (s).c1.re;                                                      \
  (r).c1.im -= (s).c1.im;                                                      \
  (r).c2.re -= (s).c2.re;                                                      \
  (r).c2.im -= (s).c2.im;                                                      \
  (r).c3.re -= (s).c3.re;                                                      \
  (r).c3.im -= (s).c3.im

#define _vector_i_add_assign(r, s)                                             \
  (r).c1.re -= (s).c1.im;                                                      \
  (r).c1.im += (s).c1.re;                                                      \
  (r).c2.re -= (s).c2.im;                                                      \
  (r).c2.im += (s).c2.re;                                                      \
  (r).c3.re -= (s).c3.im;                                                      \
  (r).c3.im += (s).c3.re

#define _vector_i_sub_assign(r, s)                                             \
  (r).c1.re += (s).c1.im;                                                      \
  (r).c1.im -= (s).c1.re;                                                      \
  (r).c2.re += (s).c2.im;                                                      \
  (r).c2.im -= (s).c2.re;                                                      \
  (r).c3.re += (s).c3.im;                                                      \
  (r).c3.im -= (s).c3.re

#define _su3_multiply(r, u, s)                                                 \
  (r).c1.re = (u).c11.re * (s).c1.re - (u).c11.im * (s).c1.im +                \
              (u).c12.re * (s).c2.re - (u).c12.im * (s).c2.im +                \
              (u).c13.re * (s).c3.re - (u).c13.im * (s).c3.im;                 \
  (r).c1.im = (u).c11.re * (s).c1.im + (u).c11.im * (s).c1.re +                \
              (u).c12.re * (s).c2.im + (u).c12.im * (s).c2.re +                \
              (u).c13.re * (s).c3.im + (u).c13.im * (s).c3.re;                 \
  (r).c2.re = (u).c21.re * (s).c1.re - (u).c21.im * (s).c1.im +                \
              (u).c22.re * (s).c2.re - (u).c22.im * (s).c2.im +                \
              (u).c23.re * (s).c3.re - (u).c23.im * (s).c3.im;                 \
  (r).c2.im = (u).c21.re * (s).c1.im + (u).c21.im * (s).c1.re +                \
              (u).c22.re * (s).c2.im + (u).c22.im * (s).c2.re +                \
              (u).c23.re * (s).c3.im + (u).c23.im * (s).c3.re;                 \
  (r).c3.re = (u).c31.re * (s).c1.re - (u).c31.im * (s).c1.im +                \
              (u).c32.re * (s).c2.re - (u).c32.im * (s).c2.im +                \
              (u).c33.re * (s).c3.re - (u).c33.im * (s).c3.im;                 \
  (r).c3.im = (u).c31.re * (s).c1.im + (u).c31.im * (s).c1.re +                \
              (u).c32.re * (s).c2.im + (u).c32.im * (s).c2.re +                \
              (u).c33.re * (s).c3.im + (u).c33.im * (s).c3.re

#define _su3_inverse_multiply(r, u, s)                                         \
  (r).c1.re = (u).c11.re * (s).c1.re + (u).c11.im * (s).c1.im +                \
              (u).c21.re * (s).c2.re + (u).c21.im * (s).c2.im +                \
              (u).c31.re * (s).c3.re + (u).c31.im * (s).c3.im;                 \
  (r).c1.im = (u).c11.re * (s).c1.im - (u).c11.im * (s).c1.re +                \
              (u).c21.re * (s).c2.im - (u).c21.im * (s).c2.re +                \
              (u).c31.re * (s).c3.im - (u).c31.im * (s).c3.re;                 \
  (r).c2.re = (u).c12.re * (s).c1.re + (u).c12.im * (s).c1.im +                \
              (u).c22.re * (s).c2.re + (u).c22.im * (s).c2.im +                \
              (u).c32.re * (s).c3.re + (u).c32.im * (s).c3.im;                 \
  (r).c2.im = (u).c12.re * (s).c1.im - (u).c12.im * (s).c1.re +                \
              (u).c22.re * (s).c2.im - (u).c22.im * (s).c2.re +                \
              (u).c32.re * (s).c3.im - (u).c32.im * (s).c3.re;                 \
  (r).c3.re = (u).c13.re * (s).c1.re + (u).c13.im * (s).c1.im +                \
              (u).c23.re * (s).c2.re + (u).c23.im * (s).c2.im +                \
              (u).c33.re * (s).c3.re + (u).c33.im * (s).c3.im;                 \
  (r).c3.im = (u).c13.re * (s).c1.im - (u).c13.im * (s).c1.re +                \
              (u).c23.re * (s).c2.im - (u).c23.im * (s).c2.re +                \
              (u).c33.re * (s).c3.im - (u).c33.im * (s).c3.re

#define _su3_copy2struct(dst, src, id)                                         \
  (dst).c11.re = (src).c11.re[id];                                             \
  (dst).c11.im = (src).c11.im[id];                                             \
  (dst).c12.re = (src).c12.re[id];                                             \
  (dst).c12.im = (src).c12.im[id];                                             \
  (dst).c13.re = (src).c13.re[id];                                             \
  (dst).c13.im = (src).c13.im[id];                                             \
  (dst).c21.re = (src).c21.re[id];                                             \
  (dst).c21.im = (src).c21.im[id];                                             \
  (dst).c22.re = (src).c22.re[id];                                             \
  (dst).c22.im = (src).c22.im[id];                                             \
  (dst).c23.re = (src).c23.re[id];                                             \
  (dst).c23.im = (src).c23.im[id];                                             \
  (dst).c31.re = (src).c31.re[id];                                             \
  (dst).c31.im = (src).c31.im[id];                                             \
  (dst).c32.re = (src).c32.re[id];                                             \
  (dst).c32.im = (src).c32.im[id];                                             \
  (dst).c33.re = (src).c33.re[id];                                             \
  (dst).c33.im = (src).c33.im[id]

#define _spinor_copy2struct(dst, src, id)                                      \
  (dst).c1.c1.re = (src).c1.c1.re[id];                                         \
  (dst).c1.c1.im = (src).c1.c1.im[id];                                         \
  (dst).c1.c2.re = (src).c1.c2.re[id];                                         \
  (dst).c1.c2.im = (src).c1.c2.im[id];                                         \
  (dst).c1.c3.re = (src).c1.c3.re[id];                                         \
  (dst).c1.c3.im = (src).c1.c3.im[id];                                         \
  (dst).c2.c1.re = (src).c2.c1.re[id];                                         \
  (dst).c2.c1.im = (src).c2.c1.im[id];                                         \
  (dst).c2.c2.re = (src).c2.c2.re[id];                                         \
  (dst).c2.c2.im = (src).c2.c2.im[id];                                         \
  (dst).c2.c3.re = (src).c2.c3.re[id];                                         \
  (dst).c2.c3.im = (src).c2.c3.im[id];                                         \
  (dst).c3.c1.re = (src).c3.c1.re[id];                                         \
  (dst).c3.c1.im = (src).c3.c1.im[id];                                         \
  (dst).c3.c2.re = (src).c3.c2.re[id];                                         \
  (dst).c3.c2.im = (src).c3.c2.im[id];                                         \
  (dst).c3.c3.re = (src).c3.c3.re[id];                                         \
  (dst).c3.c3.im = (src).c3.c3.im[id];                                         \
  (dst).c4.c1.re = (src).c4.c1.re[id];                                         \
  (dst).c4.c1.im = (src).c4.c1.im[id];                                         \
  (dst).c4.c2.re = (src).c4.c2.re[id];                                         \
  (dst).c4.c2.im = (src).c4.c2.im[id];                                         \
  (dst).c4.c3.re = (src).c4.c3.re[id];                                         \
  (dst).c4.c3.im = (src).c4.c3.im[id]

#define _spinor_add2arrays(dst, src, id)                                       \
  (dst).c1.c1.re[id] += (src).c1.c1.re;                                        \
  (dst).c1.c1.im[id] += (src).c1.c1.im;                                        \
  (dst).c1.c2.re[id] += (src).c1.c2.re;                                        \
  (dst).c1.c2.im[id] += (src).c1.c2.im;                                        \
  (dst).c1.c3.re[id] += (src).c1.c3.re;                                        \
  (dst).c1.c3.im[id] += (src).c1.c3.im;                                        \
  (dst).c2.c1.re[id] += (src).c2.c1.re;                                        \
  (dst).c2.c1.im[id] += (src).c2.c1.im;                                        \
  (dst).c2.c2.re[id] += (src).c2.c2.re;                                        \
  (dst).c2.c2.im[id] += (src).c2.c2.im;                                        \
  (dst).c2.c3.re[id] += (src).c2.c3.re;                                        \
  (dst).c2.c3.im[id] += (src).c2.c3.im;                                        \
  (dst).c3.c1.re[id] += (src).c3.c1.re;                                        \
  (dst).c3.c1.im[id] += (src).c3.c1.im;                                        \
  (dst).c3.c2.re[id] += (src).c3.c2.re;                                        \
  (dst).c3.c2.im[id] += (src).c3.c2.im;                                        \
  (dst).c3.c3.re[id] += (src).c3.c3.re;                                        \
  (dst).c3.c3.im[id] += (src).c3.c3.im;                                        \
  (dst).c4.c1.re[id] += (src).c4.c1.re;                                        \
  (dst).c4.c1.im[id] += (src).c4.c1.im;                                        \
  (dst).c4.c2.re[id] += (src).c4.c2.re;                                        \
  (dst).c4.c2.im[id] += (src).c4.c2.im;                                        \
  (dst).c4.c3.re[id] += (src).c4.c3.re;                                        \
  (dst).c4.c3.im[id] += (src).c4.c3.im

#endif
