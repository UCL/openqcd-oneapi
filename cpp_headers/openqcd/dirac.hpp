
/*******************************************************************************
 *
 * File dirac.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_DIRAC_HPP
#define CPP_DIRAC_HPP

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/dirac.h"
}

namespace openqcd {
namespace dirac {

// DW_BND_C
OPENQCD_MODULE_FUNCTION_ALIAS(Dw_bnd, dirac)

// DW_C
OPENQCD_MODULE_FUNCTION_ALIAS(Dw, dirac)
OPENQCD_MODULE_FUNCTION_ALIAS(Dwee, dirac)
OPENQCD_MODULE_FUNCTION_ALIAS(Dwoo, dirac)
OPENQCD_MODULE_FUNCTION_ALIAS(Dweo, dirac)
OPENQCD_MODULE_FUNCTION_ALIAS(Dwoe, dirac)
OPENQCD_MODULE_FUNCTION_ALIAS(Dwhat, dirac)
OPENQCD_MODULE_FUNCTION_ALIAS(Dw_blk, dirac)
OPENQCD_MODULE_FUNCTION_ALIAS(Dwee_blk, dirac)
OPENQCD_MODULE_FUNCTION_ALIAS(Dwoo_blk, dirac)
OPENQCD_MODULE_FUNCTION_ALIAS(Dwoe_blk, dirac)
OPENQCD_MODULE_FUNCTION_ALIAS(Dweo_blk, dirac)
OPENQCD_MODULE_FUNCTION_ALIAS(Dwhat_blk, dirac)

// DW_DBLE_C
OPENQCD_MODULE_FUNCTION_ALIAS(Dw_dble, dirac)
OPENQCD_MODULE_FUNCTION_ALIAS(Dwee_dble, dirac)
OPENQCD_MODULE_FUNCTION_ALIAS(Dwoo_dble, dirac)
OPENQCD_MODULE_FUNCTION_ALIAS(Dweo_dble, dirac)
OPENQCD_MODULE_FUNCTION_ALIAS(Dwoe_dble, dirac)
OPENQCD_MODULE_FUNCTION_ALIAS(Dwhat_dble, dirac)
OPENQCD_MODULE_FUNCTION_ALIAS(Dw_blk_dble, dirac)
OPENQCD_MODULE_FUNCTION_ALIAS(Dwee_blk_dble, dirac)
OPENQCD_MODULE_FUNCTION_ALIAS(Dwoo_blk_dble, dirac)
OPENQCD_MODULE_FUNCTION_ALIAS(Dwoe_blk_dble, dirac)
OPENQCD_MODULE_FUNCTION_ALIAS(Dweo_blk_dble, dirac)
OPENQCD_MODULE_FUNCTION_ALIAS(Dwhat_blk_dble, dirac)

} // namespace dirac
} // namespace openqcd

#endif // ifndef CPP_DIRAC_HPP
