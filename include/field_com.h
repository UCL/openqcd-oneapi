
/*******************************************************************************
 *
 * File field_com.h
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef FIELD_COM_H
#define FIELD_COM_H

#include "su3.h"
#include <stdlib.h>

/* LINK_COMMUNICATION_C */
extern void
openqcd_field_com__copy_boundary_su3_field(openqcd__su3_dble *su3_field);
extern void
openqcd_field_com__add_boundary_su3_field(openqcd__su3_dble *su3_field);
extern void openqcd_field_com__copy_boundary_su3_alg_field(
    openqcd__su3_alg_dble *su3_alg_field);
extern void openqcd_field_com__add_boundary_su3_alg_field(
    openqcd__su3_alg_dble *su3_alg_field);

/* LINK_PARTIAL_COMMUNICATION_C */
extern void
openqcd_field_com__copy_partial_boundary_su3_field(openqcd__su3_dble *su3_field,
                                                   int const *dirs);
extern void
openqcd_field_com__add_partial_boundary_su3_field(openqcd__su3_dble *su3_field,
                                                  int const *dirs);
extern void openqcd_field_com__copy_partial_boundary_su3_alg_field(
    openqcd__su3_alg_dble *su3_alg_field, int const *dirs);
extern void openqcd_field_com__add_partial_boundary_su3_alg_field(
    openqcd__su3_alg_dble *su3_alg_field, int const *dirs);

extern void openqcd_field_com__copy_spatial_boundary_su3_field(
    openqcd__su3_dble *su3_field);
extern void
openqcd_field_com__add_spatial_boundary_su3_field(openqcd__su3_dble *su3_field);
extern void openqcd_field_com__copy_spatial_boundary_su3_alg_field(
    openqcd__su3_alg_dble *su3_alg_field);
extern void openqcd_field_com__add_spatial_boundary_su3_alg_field(
    openqcd__su3_alg_dble *su3_alg_field);

/* COMMUNICATION_BUFFER_C */
extern double *openqcd_field_com__communication_buffer(void);

#if defined(OPENQCD_INTERNAL)
/* LINK_COMMUNICATION_C */
#define copy_boundary_su3_field(...)                                           \
  openqcd_field_com__copy_boundary_su3_field(__VA_ARGS__)
#define add_boundary_su3_field(...)                                            \
  openqcd_field_com__add_boundary_su3_field(__VA_ARGS__)
#define copy_boundary_su3_alg_field(...)                                       \
  openqcd_field_com__copy_boundary_su3_alg_field(__VA_ARGS__)
#define add_boundary_su3_alg_field(...)                                        \
  openqcd_field_com__add_boundary_su3_alg_field(__VA_ARGS__)

/* LINK_PARTIAL_COMMUNICATION_C */
#define copy_partial_boundary_su3_field(...)                                   \
  openqcd_field_com__copy_partial_boundary_su3_field(__VA_ARGS__)
#define add_partial_boundary_su3_field(...)                                    \
  openqcd_field_com__add_partial_boundary_su3_field(__VA_ARGS__)
#define copy_partial_boundary_su3_alg_field(...)                               \
  openqcd_field_com__copy_partial_boundary_su3_alg_field(__VA_ARGS__)
#define add_partial_boundary_su3_alg_field(...)                                \
  openqcd_field_com__add_partial_boundary_su3_alg_field(__VA_ARGS__)

#define copy_spatial_boundary_su3_field(...)                                   \
  openqcd_field_com__copy_spatial_boundary_su3_field(__VA_ARGS__)
#define add_spatial_boundary_su3_field(...)                                    \
  openqcd_field_com__add_spatial_boundary_su3_field(__VA_ARGS__)
#define copy_spatial_boundary_su3_alg_field(...)                               \
  openqcd_field_com__copy_spatial_boundary_su3_alg_field(__VA_ARGS__)
#define add_spatial_boundary_su3_alg_field(...)                                \
  openqcd_field_com__add_spatial_boundary_su3_alg_field(__VA_ARGS__)

/* COMMUNICATION_BUFFER_C */
#define communication_buffer(...)                                              \
  openqcd_field_com__communication_buffer(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif /* FIELD_COM_H */
