
/*******************************************************************************
 *
 * File forces.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_FORCES_HPP
#define CPP_FORCES_HPP

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/forces.h"
}

namespace openqcd {
namespace forces {

// FORCE0_C
OPENQCD_MODULE_FUNCTION_ALIAS(plaq_frc, forces)
OPENQCD_MODULE_FUNCTION_ALIAS(force0, forces)
OPENQCD_MODULE_FUNCTION_ALIAS(action0, forces)

// FORCE1_C
OPENQCD_MODULE_FUNCTION_ALIAS(setpf1, forces)
OPENQCD_MODULE_FUNCTION_ALIAS(force1, forces)
OPENQCD_MODULE_FUNCTION_ALIAS(action1, forces)

// FORCE2_C
OPENQCD_MODULE_FUNCTION_ALIAS(setpf2, forces)
OPENQCD_MODULE_FUNCTION_ALIAS(force2, forces)
OPENQCD_MODULE_FUNCTION_ALIAS(action2, forces)

// FORCE3_C
OPENQCD_MODULE_FUNCTION_ALIAS(setpf3, forces)
OPENQCD_MODULE_FUNCTION_ALIAS(force3, forces)
OPENQCD_MODULE_FUNCTION_ALIAS(action3, forces)

// FORCE4_C
OPENQCD_MODULE_FUNCTION_ALIAS(setpf4, forces)
OPENQCD_MODULE_FUNCTION_ALIAS(force4, forces)
OPENQCD_MODULE_FUNCTION_ALIAS(action4, forces)

// FORCE5_C
OPENQCD_MODULE_FUNCTION_ALIAS(setpf5, forces)
OPENQCD_MODULE_FUNCTION_ALIAS(force5, forces)
OPENQCD_MODULE_FUNCTION_ALIAS(action5, forces)

// FRCFCTS_C
OPENQCD_MODULE_FUNCTION_ALIAS(det2xt, forces)
OPENQCD_MODULE_FUNCTION_ALIAS(prod2xt, forces)
OPENQCD_MODULE_FUNCTION_ALIAS(prod2xv, forces)

// GENFRC_C
OPENQCD_MODULE_FUNCTION_ALIAS(sw_frc, forces)
OPENQCD_MODULE_FUNCTION_ALIAS(hop_frc, forces)

// TMCG_C
OPENQCD_MODULE_FUNCTION_ALIAS(tmcg, forces)
OPENQCD_MODULE_FUNCTION_ALIAS(tmcgeo, forces)

// TMCGM_C
OPENQCD_MODULE_FUNCTION_ALIAS(tmcgm, forces)

// XTENSOR_C
OPENQCD_MODULE_FUNCTION_ALIAS(xtensor, forces)
OPENQCD_MODULE_FUNCTION_ALIAS(set_xt2zero, forces)
OPENQCD_MODULE_FUNCTION_ALIAS(add_det2xt, forces)
OPENQCD_MODULE_FUNCTION_ALIAS(add_prod2xt, forces)
OPENQCD_MODULE_FUNCTION_ALIAS(xvector, forces)
OPENQCD_MODULE_FUNCTION_ALIAS(set_xv2zero, forces)
OPENQCD_MODULE_FUNCTION_ALIAS(add_prod2xv, forces)

} // namespace forces
} // namespace openqcd

#endif
