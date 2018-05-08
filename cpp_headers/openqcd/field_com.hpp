
/*******************************************************************************
 *
 * File field_com.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_FIELD_COM_HPP
#define CPP_FIELD_COM_HPP

extern "C" {
#include "c_headers/field_com.h"
}

namespace openqcd {
namespace fieldcom {

// LINK_COMMUNICATION_C
const auto &copy_boundary_su3_field =
    openqcd_field_com__copy_boundary_su3_field;
const auto &add_boundary_su3_field = openqcd_field_com__add_boundary_su3_field;
const auto &copy_boundary_su3_alg_field =
    openqcd_field_com__copy_boundary_su3_alg_field;
const auto &add_boundary_su3_alg_field =
    openqcd_field_com__add_boundary_su3_alg_field;

// LINK_PARTIAL_COMMUNICATION_C
const auto &copy_partial_boundary_su3_field =
    openqcd_field_com__copy_partial_boundary_su3_field;
const auto &add_partial_boundary_su3_field =
    openqcd_field_com__add_partial_boundary_su3_field;
const auto &copy_partial_boundary_su3_alg_field =
    openqcd_field_com__copy_partial_boundary_su3_alg_field;
const auto &add_partial_boundary_su3_alg_field =
    openqcd_field_com__add_partial_boundary_su3_alg_field;

const auto &copy_spatial_boundary_su3_field =
    openqcd_field_com__copy_spatial_boundary_su3_field;
const auto &add_spatial_boundary_su3_field =
    openqcd_field_com__add_spatial_boundary_su3_field;
const auto &copy_spatial_boundary_su3_alg_field =
    openqcd_field_com__copy_spatial_boundary_su3_alg_field;
const auto &add_spatial_boundary_su3_alg_field =
    openqcd_field_com__add_spatial_boundary_su3_alg_field;

// COMMUNICATION_BUFFER_C
const auto &communication_buffer = openqcd_field_com__communication_buffer;

} // namespace fieldcom
} // namespace openqcd

#endif // CPP_FIELD_COM_HPP
