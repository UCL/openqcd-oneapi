
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

extern "C" {
#include "c_headers/stout_smearing.h"
}

namespace openqcd {
namespace stout_smearing {

using ch_mat_coeff_pair_t = openqcd_stout_smearing__ch_mat_coeff_pair_t;

// FORCE_UNSMEARING_C
const auto &unsmear_force = openqcd_stout_smearing__unsmear_force;
const auto &unsmear_mdforce = openqcd_stout_smearing__unsmear_mdforce;

// SMEARED_FIELDS_C
const auto &smeared_fields = openqcd_stout_smearing__smeared_fields;
const auto &free_smearing_ch_coeff_fields =
    openqcd_stout_smearing__free_smearing_ch_coeff_fields;
const auto &smearing_ch_coeff_fields =
    openqcd_stout_smearing__smearing_ch_coeff_fields;

// STOUT_SMEARING_C
const auto &smear_fields = openqcd_stout_smearing__smear_fields;
const auto &unsmear_fields = openqcd_stout_smearing__unsmear_fields;

} // namespace stout_smearing
} // namespace openqcd

#endif // CPP_STOUT_SMEARING_HPP
