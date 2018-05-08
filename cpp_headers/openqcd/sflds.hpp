
/*******************************************************************************
 *
 * File sflds.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_SFLDS_HPP
#define CPP_SFLDS_HPP

extern "C" {
#include "c_headers/sflds.h"
}

namespace openqcd {
namespace sflds {

// PBND_C
const auto &assign_s2w = openqcd_sflds__assign_s2w;
const auto &add_assign_w2s = openqcd_sflds__add_assign_w2s;
const auto &sub_assign_w2s = openqcd_sflds__sub_assign_w2s;
const auto &mulg5_sub_assign_w2s = openqcd_sflds__mulg5_sub_assign_w2s;

// PBND_DBLE_C
const auto &assign_sd2wd = openqcd_sflds__assign_sd2wd;
const auto &add_assign_wd2sd = openqcd_sflds__add_assign_wd2sd;
const auto &sub_assign_wd2sd = openqcd_sflds__sub_assign_wd2sd;
const auto &mulg5_sub_assign_wd2sd = openqcd_sflds__mulg5_sub_assign_wd2sd;

// SFLDS_C
const auto &set_s2zero = openqcd_sflds__set_s2zero;
const auto &set_sd2zero = openqcd_sflds__set_sd2zero;
const auto &random_s = openqcd_sflds__random_s;
const auto &random_sd = openqcd_sflds__random_sd;
const auto &assign_s2s = openqcd_sflds__assign_s2s;
const auto &assign_s2sd = openqcd_sflds__assign_s2sd;
const auto &assign_sd2s = openqcd_sflds__assign_sd2s;
const auto &assign_sd2sd = openqcd_sflds__assign_sd2sd;
const auto &diff_s2s = openqcd_sflds__diff_s2s;
const auto &add_s2sd = openqcd_sflds__add_s2sd;
const auto &diff_sd2s = openqcd_sflds__diff_sd2s;

// SCOM_C
const auto &cps_int_bnd = openqcd_sflds__cps_int_bnd;
const auto &cps_ext_bnd = openqcd_sflds__cps_ext_bnd;
const auto &free_sbufs = openqcd_sflds__free_sbufs;

// SDCOM_C
const auto &cpsd_int_bnd = openqcd_sflds__cpsd_int_bnd;
const auto &cpsd_ext_bnd = openqcd_sflds__cpsd_ext_bnd;

} // namespace sflds
} // namespace openqcd

#endif // CPP_SFLDS_HPP
