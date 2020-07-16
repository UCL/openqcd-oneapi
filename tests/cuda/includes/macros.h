#ifndef MACROS_H
#define MACROS_H

#define _vector_mul_assign(r, c)                                                 \
    (r).c1.re *= (c);                                                            \
    (r).c1.im *= (c);                                                            \
    (r).c2.re *= (c);                                                            \
    (r).c2.im *= (c);                                                            \
    (r).c3.re *= (c);                                                            \
    (r).c3.im *= (c)

#define _vector_add(r, s1, s2)                                                   \
    (r).c1.re = (s1).c1.re + (s2).c1.re;                                         \
    (r).c1.im = (s1).c1.im + (s2).c1.im;                                         \
    (r).c2.re = (s1).c2.re + (s2).c2.re;                                         \
    (r).c2.im = (s1).c2.im + (s2).c2.im;                                         \
    (r).c3.re = (s1).c3.re + (s2).c3.re;                                         \
    (r).c3.im = (s1).c3.im + (s2).c3.im

#define _vector_sub(r, s1, s2)                                                   \
    (r).c1.re = (s1).c1.re - (s2).c1.re;                                         \
    (r).c1.im = (s1).c1.im - (s2).c1.im;                                         \
    (r).c2.re = (s1).c2.re - (s2).c2.re;                                         \
    (r).c2.im = (s1).c2.im - (s2).c2.im;                                         \
    (r).c3.re = (s1).c3.re - (s2).c3.re;                                         \
    (r).c3.im = (s1).c3.im - (s2).c3.im

#define _vector_i_add(r, s1, s2)                                                 \
    (r).c1.re = (s1).c1.re - (s2).c1.im;                                         \
    (r).c1.im = (s1).c1.im + (s2).c1.re;                                         \
    (r).c2.re = (s1).c2.re - (s2).c2.im;                                         \
    (r).c2.im = (s1).c2.im + (s2).c2.re;                                         \
    (r).c3.re = (s1).c3.re - (s2).c3.im;                                         \
    (r).c3.im = (s1).c3.im + (s2).c3.re

#define _vector_i_sub(r, s1, s2)                                                 \
    (r).c1.re = (s1).c1.re + (s2).c1.im;                                         \
    (r).c1.im = (s1).c1.im - (s2).c1.re;                                         \
    (r).c2.re = (s1).c2.re + (s2).c2.im;                                         \
    (r).c2.im = (s1).c2.im - (s2).c2.re;                                         \
    (r).c3.re = (s1).c3.re + (s2).c3.im;                                         \
    (r).c3.im = (s1).c3.im - (s2).c3.re

#define _vector_add_assign(r, s)                                                 \
    (r).c1.re += (s).c1.re;                                                      \
    (r).c1.im += (s).c1.im;                                                      \
    (r).c2.re += (s).c2.re;                                                      \
    (r).c2.im += (s).c2.im;                                                      \
    (r).c3.re += (s).c3.re;                                                      \
    (r).c3.im += (s).c3.im

#define _vector_sub_assign(r, s)                                                 \
    (r).c1.re -= (s).c1.re;                                                      \
    (r).c1.im -= (s).c1.im;                                                      \
    (r).c2.re -= (s).c2.re;                                                      \
    (r).c2.im -= (s).c2.im;                                                      \
    (r).c3.re -= (s).c3.re;                                                      \
    (r).c3.im -= (s).c3.im

#define _vector_i_add_assign(r, s)                                               \
    (r).c1.re -= (s).c1.im;                                                      \
    (r).c1.im += (s).c1.re;                                                      \
    (r).c2.re -= (s).c2.im;                                                      \
    (r).c2.im += (s).c2.re;                                                      \
    (r).c3.re -= (s).c3.im;                                                      \
    (r).c3.im += (s).c3.re

#define _vector_i_sub_assign(r, s)                                               \
    (r).c1.re += (s).c1.im;                                                      \
    (r).c1.im -= (s).c1.re;                                                      \
    (r).c2.re += (s).c2.im;                                                      \
    (r).c2.im -= (s).c2.re;                                                      \
    (r).c3.re += (s).c3.im;                                                      \
    (r).c3.im -= (s).c3.re

#define _su3_multiply(r, u, s)                                                   \
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

#define _su3_inverse_multiply(r, u, s)                                           \
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

#endif
