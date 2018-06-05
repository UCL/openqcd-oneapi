
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

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/field_com.h"
}

namespace openqcd {
namespace fieldcom {

// LINK_COMMUNICATION_C
OPENQCD_MODULE_FUNCTION_ALIAS(copy_boundary_su3_field, field_com)
OPENQCD_MODULE_FUNCTION_ALIAS(add_boundary_su3_field, field_com)
OPENQCD_MODULE_FUNCTION_ALIAS(copy_boundary_su3_alg_field, field_com)
OPENQCD_MODULE_FUNCTION_ALIAS(add_boundary_su3_alg_field, field_com)

// LINK_PARTIAL_COMMUNICATION_C
OPENQCD_MODULE_FUNCTION_ALIAS(copy_partial_boundary_su3_field, field_com)
OPENQCD_MODULE_FUNCTION_ALIAS(add_partial_boundary_su3_field, field_com)
OPENQCD_MODULE_FUNCTION_ALIAS(copy_partial_boundary_su3_alg_field, field_com)
OPENQCD_MODULE_FUNCTION_ALIAS(add_partial_boundary_su3_alg_field, field_com)

OPENQCD_MODULE_FUNCTION_ALIAS(copy_spatial_boundary_su3_field, field_com)
OPENQCD_MODULE_FUNCTION_ALIAS(add_spatial_boundary_su3_field, field_com)
OPENQCD_MODULE_FUNCTION_ALIAS(copy_spatial_boundary_su3_alg_field, field_com)
OPENQCD_MODULE_FUNCTION_ALIAS(add_spatial_boundary_su3_alg_field, field_com)

// COMMUNICATION_BUFFER_C
OPENQCD_MODULE_FUNCTION_ALIAS(communication_buffer, field_com)

} // namespace fieldcom
} // namespace openqcd

#endif // CPP_FIELD_COM_HPP
