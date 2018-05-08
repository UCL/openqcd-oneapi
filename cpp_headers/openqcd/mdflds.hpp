
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

extern "C" {
#include "c_headers/mdflds.h"
}

namespace openqcd {
namespace mdflds {

using mdflds_t = openqcd_mdflds__mdflds_t;

// MDFLDS_C
const auto &mdflds = openqcd_mdflds__mdflds;
const auto &set_frc2zero = openqcd_mdflds__set_frc2zero;
const auto &bnd_mom2zero = openqcd_mdflds__bnd_mom2zero;
const auto &random_mom = openqcd_mdflds__random_mom;
const auto &momentum_action = openqcd_mdflds__momentum_action;
const auto &copy_bnd_frc = openqcd_mdflds__copy_bnd_frc;
const auto &add_bnd_frc = openqcd_mdflds__add_bnd_frc;

} // namespace mdflds
} // namespace openqcd

#endif
