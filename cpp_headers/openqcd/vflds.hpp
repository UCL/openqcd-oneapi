
/*******************************************************************************
 *
 * File vflds.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_VFLDS_HPP
#define CPP_VFLDS_HPP

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/vflds.h"
}

namespace openqcd {
namespace vflds {

// VCOM_C
OPENQCD_MODULE_FUNCTION_ALIAS(cpv_int_bnd, vflds)
OPENQCD_MODULE_FUNCTION_ALIAS(cpv_ext_bnd, vflds)

// VDCOM_C
OPENQCD_MODULE_FUNCTION_ALIAS(cpvd_int_bnd, vflds)
OPENQCD_MODULE_FUNCTION_ALIAS(cpvd_ext_bnd, vflds)

// VFLDS_C
OPENQCD_MODULE_FUNCTION_ALIAS(vflds, vflds)
OPENQCD_MODULE_FUNCTION_ALIAS(vdflds, vflds)

// VINIT_C
OPENQCD_MODULE_FUNCTION_ALIAS(set_v2zero, vflds)
OPENQCD_MODULE_FUNCTION_ALIAS(set_vd2zero, vflds)
OPENQCD_MODULE_FUNCTION_ALIAS(random_v, vflds)
OPENQCD_MODULE_FUNCTION_ALIAS(random_vd, vflds)
OPENQCD_MODULE_FUNCTION_ALIAS(assign_v2v, vflds)
OPENQCD_MODULE_FUNCTION_ALIAS(assign_v2vd, vflds)
OPENQCD_MODULE_FUNCTION_ALIAS(assign_vd2v, vflds)
OPENQCD_MODULE_FUNCTION_ALIAS(assign_vd2vd, vflds)
OPENQCD_MODULE_FUNCTION_ALIAS(add_v2vd, vflds)
OPENQCD_MODULE_FUNCTION_ALIAS(diff_vd2v, vflds)

} // namespace vflds
} // namespace openqcd

#endif // CPP_VFLDS_HPP
