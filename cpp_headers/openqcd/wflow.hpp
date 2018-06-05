
/*******************************************************************************
 *
 * File wflow.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_WFLOW_HPP
#define CPP_WFLOW_HPP

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/wflow.h"
}

namespace openqcd {
namespace wflow {

// WFLOW_C
OPENQCD_MODULE_FUNCTION_ALIAS(fwd_euler, wflow)
OPENQCD_MODULE_FUNCTION_ALIAS(fwd_rk2, wflow)
OPENQCD_MODULE_FUNCTION_ALIAS(fwd_rk3, wflow)

} // namespace wflow
} // namespace openqcd

#endif
