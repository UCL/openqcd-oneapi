
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

extern "C" {
#include "c_headers/sw_term.h"
}

namespace openqcd {
namespace sw_term {

// PAULI_C
const auto &mul_pauli = openqcd_sw_term__mul_pauli;
const auto &mul_pauli2 = openqcd_sw_term__mul_pauli2;
const auto &assign_pauli = openqcd_sw_term__assign_pauli;
const auto &apply_sw = openqcd_sw_term__apply_sw;

// PAULI_DBLE_C
const auto &mul_pauli_dble = openqcd_sw_term__mul_pauli_dble;
const auto &mul_pauli2_dble = openqcd_sw_term__mul_pauli2_dble;
const auto &inv_pauli_dble = openqcd_sw_term__inv_pauli_dble;
const auto &det_pauli_dble = openqcd_sw_term__det_pauli_dble;
const auto &apply_sw_dble = openqcd_sw_term__apply_sw_dble;
const auto &apply_swinv_dble = openqcd_sw_term__apply_swinv_dble;

// SWFLDS_C
const auto &swfld = openqcd_sw_term__swfld;
const auto &swdfld = openqcd_sw_term__swdfld;
const auto &assign_swd2sw = openqcd_sw_term__assign_swd2sw;

// SW_TERM_C
const auto &sw_term = openqcd_sw_term__sw_term;

} // namespace sw_term
} // namespace openqcd

#endif // CPP_SW_TERM_HPP
