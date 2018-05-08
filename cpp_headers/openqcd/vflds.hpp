
/*******************************************************************************
 *
 * File vflds.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_VFLDS_HPP
#define CPP_VFLDS_HPP

extern "C" {
#include "c_headers/vflds.h"
}

namespace openqcd {
namespace vflds {

// VCOM_C
const auto &cpv_int_bnd = openqcd_vflds__cpv_int_bnd;
const auto &cpv_ext_bnd = openqcd_vflds__cpv_ext_bnd;

// VDCOM_C
const auto &cpvd_int_bnd = openqcd_vflds__cpvd_int_bnd;
const auto &cpvd_ext_bnd = openqcd_vflds__cpvd_ext_bnd;

// VFLDS_C
const auto &vflds = openqcd_vflds__vflds;
const auto &vdflds = openqcd_vflds__vdflds;

// VINIT_C
const auto &set_v2zero = openqcd_vflds__set_v2zero;
const auto &set_vd2zero = openqcd_vflds__set_vd2zero;
const auto &random_v = openqcd_vflds__random_v;
const auto &random_vd = openqcd_vflds__random_vd;
const auto &assign_v2v = openqcd_vflds__assign_v2v;
const auto &assign_v2vd = openqcd_vflds__assign_v2vd;
const auto &assign_vd2v = openqcd_vflds__assign_vd2v;
const auto &assign_vd2vd = openqcd_vflds__assign_vd2vd;
const auto &add_v2vd = openqcd_vflds__add_v2vd;
const auto &diff_vd2v = openqcd_vflds__diff_vd2v;

} // namespace vflds
} // namespace openqcd

#endif // CPP_VFLDS_HPP
