
/*******************************************************************************
 *
 * File stout_smearing.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_STOUT_SMEARING_HPP
#define CPP_STOUT_SMEARING_HPP

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/stout_smearing.h"
}

namespace openqcd {
namespace stout_smearing {

using ch_mat_coeff_pair_t = openqcd_stout_smearing__ch_mat_coeff_pair_t;

// FORCE_UNSMEARING_C
OPENQCD_MODULE_FUNCTION_ALIAS(unsmear_force, stout_smearing)
OPENQCD_MODULE_FUNCTION_ALIAS(unsmear_mdforce, stout_smearing)

// SMEARED_FIELDS_C
OPENQCD_MODULE_FUNCTION_ALIAS(smeared_fields, stout_smearing)
OPENQCD_MODULE_FUNCTION_ALIAS(free_smearing_ch_coeff_fields, stout_smearing)
OPENQCD_MODULE_FUNCTION_ALIAS(smearing_ch_coeff_fields, stout_smearing)

// STOUT_SMEARING_C
OPENQCD_MODULE_FUNCTION_ALIAS(smear_fields, stout_smearing)
OPENQCD_MODULE_FUNCTION_ALIAS(unsmear_fields, stout_smearing)

} // namespace stout_smearing
} // namespace openqcd

#endif // CPP_STOUT_SMEARING_HPP
