#ifndef FIELD_COM_H
#define FIELD_COM_H

#include "su3.h"
#include <stdlib.h>

/* LINK_COMMUNICATION_C */
extern void copy_boundary_su3_field(su3_dble *su3_field);
extern void add_boundary_su3_field(su3_dble *su3_field);
extern void copy_boundary_su3_alg_field(su3_alg_dble *su3_alg_field);
extern void add_boundary_su3_alg_field(su3_alg_dble *su3_alg_field);

/* LINK_PARTIAL_COMMUNICATION_C */
extern void copy_partial_boundary_su3_field(su3_dble *su3_field,
                                            int const *dirs);
extern void add_partial_boundary_su3_field(su3_dble *su3_field,
                                           int const *dirs);
extern void copy_partial_boundary_su3_alg_field(su3_alg_dble *su3_alg_field,
                                                int const *dirs);
extern void add_partial_boundary_su3_alg_field(su3_alg_dble *su3_alg_field,
                                               int const *dirs);

extern void copy_spatial_boundary_su3_field(su3_dble *su3_field);
extern void add_spatial_boundary_su3_field(su3_dble *su3_field);
extern void copy_spatial_boundary_su3_alg_field(su3_alg_dble *su3_alg_field);
extern void add_spatial_boundary_su3_alg_field(su3_alg_dble *su3_alg_field);

/* COMMUNICATION_BUFFER_C */
extern double *communication_buffer(void);

#endif /* FIELD_COM_H */
