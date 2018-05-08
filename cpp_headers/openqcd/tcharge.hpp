
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

extern "C" {
#include "c_headers/tcharge.h"
}

namespace openqcd {
namespace tcharge {

// FTCOM_C
const auto &copy_bnd_ft = openqcd_tcharge__copy_bnd_ft;
const auto &add_bnd_ft = openqcd_tcharge__add_bnd_ft;
const auto &free_ftcom_bufs = openqcd_tcharge__free_ftcom_bufs;

// FTENSOR_C
const auto &ftensor = openqcd_tcharge__ftensor;

// TCHARGE_C
const auto &tcharge = openqcd_tcharge__tcharge;
const auto &tcharge_slices = openqcd_tcharge__tcharge_slices;

// YM_ACTION_C
const auto &ym_action = openqcd_tcharge__ym_action;
const auto &ym_action_slices = openqcd_tcharge__ym_action_slices;

} // namespace tcharge
} // namespace openqcd

#endif // CPP_TCHARGE_HPP
