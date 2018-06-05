
/*******************************************************************************
 *
 * File sw_term.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_SW_TERM_HPP
#define CPP_SW_TERM_HPP

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/sw_term.h"
}

namespace openqcd {
namespace sw_term {

// PAULI_C
OPENQCD_MODULE_FUNCTION_ALIAS(mul_pauli, sw_term)
OPENQCD_MODULE_FUNCTION_ALIAS(mul_pauli2, sw_term)
OPENQCD_MODULE_FUNCTION_ALIAS(assign_pauli, sw_term)
OPENQCD_MODULE_FUNCTION_ALIAS(apply_sw, sw_term)

// PAULI_DBLE_C
OPENQCD_MODULE_FUNCTION_ALIAS(mul_pauli_dble, sw_term)
OPENQCD_MODULE_FUNCTION_ALIAS(mul_pauli2_dble, sw_term)
OPENQCD_MODULE_FUNCTION_ALIAS(inv_pauli_dble, sw_term)
OPENQCD_MODULE_FUNCTION_ALIAS(det_pauli_dble, sw_term)
OPENQCD_MODULE_FUNCTION_ALIAS(apply_sw_dble, sw_term)
OPENQCD_MODULE_FUNCTION_ALIAS(apply_swinv_dble, sw_term)

// SWFLDS_C
OPENQCD_MODULE_FUNCTION_ALIAS(swfld, sw_term)
OPENQCD_MODULE_FUNCTION_ALIAS(swdfld, sw_term)
OPENQCD_MODULE_FUNCTION_ALIAS(assign_swd2sw, sw_term)

// SW_TERM_C
OPENQCD_MODULE_FUNCTION_ALIAS(sw_term, sw_term)

} // namespace sw_term
} // namespace openqcd

#endif // CPP_SW_TERM_HPP
