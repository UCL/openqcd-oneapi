
/*******************************************************************************
 *
 * File tcharge.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_TCHARGE_HPP
#define CPP_TCHARGE_HPP

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/tcharge.h"
}

namespace openqcd {
namespace tcharge {

// FTCOM_C
OPENQCD_MODULE_FUNCTION_ALIAS(copy_bnd_ft, tcharge)
OPENQCD_MODULE_FUNCTION_ALIAS(add_bnd_ft, tcharge)

// FTENSOR_C
OPENQCD_MODULE_FUNCTION_ALIAS(ftensor, tcharge)

// TCHARGE_C
OPENQCD_MODULE_FUNCTION_ALIAS(tcharge, tcharge)
OPENQCD_MODULE_FUNCTION_ALIAS(tcharge_slices, tcharge)

// YM_ACTION_C
OPENQCD_MODULE_FUNCTION_ALIAS(ym_action, tcharge)
OPENQCD_MODULE_FUNCTION_ALIAS(ym_action_slices, tcharge)

} // namespace tcharge
} // namespace openqcd

#endif // CPP_TCHARGE_HPP
