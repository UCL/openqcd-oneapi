
/*******************************************************************************
 *
 * File mdflds.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_MDFLDS_HPP
#define CPP_MDFLDS_HPP

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/mdflds.h"
}

namespace openqcd {
namespace mdflds {

using mdflds_t = openqcd_mdflds__mdflds_t;

// MDFLDS_C
OPENQCD_MODULE_FUNCTION_ALIAS(mdflds, mdflds)
OPENQCD_MODULE_FUNCTION_ALIAS(set_frc2zero, mdflds)
OPENQCD_MODULE_FUNCTION_ALIAS(bnd_mom2zero, mdflds)
OPENQCD_MODULE_FUNCTION_ALIAS(random_mom, mdflds)
OPENQCD_MODULE_FUNCTION_ALIAS(momentum_action, mdflds)
OPENQCD_MODULE_FUNCTION_ALIAS(copy_bnd_frc, mdflds)
OPENQCD_MODULE_FUNCTION_ALIAS(add_bnd_frc, mdflds)

} // namespace mdflds
} // namespace openqcd

#endif
