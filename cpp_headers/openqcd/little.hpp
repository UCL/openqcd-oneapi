
/*******************************************************************************
 *
 * File little.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_LITTLE_HPP
#define CPP_LITTLE_HPP

extern "C" {
#include "c_headers/little.h"
}

namespace openqcd {
namespace little {

using Aw_t = openqcd_little__Aw_t;
using Aw_dble_t = openqcd_little__Aw_dble_t;
using b2b_flds_t = openqcd_little__b2b_flds_t;

// AW_COM_C
const auto &b2b_flds = openqcd_little__b2b_flds;
const auto &cpAoe_ext_bnd = openqcd_little__cpAoe_ext_bnd;
const auto &cpAee_int_bnd = openqcd_little__cpAee_int_bnd;

// AW_C
const auto &Aw = openqcd_little__Aw;
const auto &Aweeinv = openqcd_little__Aweeinv;
const auto &Awooinv = openqcd_little__Awooinv;
const auto &Awoe = openqcd_little__Awoe;
const auto &Aweo = openqcd_little__Aweo;
const auto &Awhat = openqcd_little__Awhat;

// AW_DBLE_C
const auto &Aw_dble = openqcd_little__Aw_dble;
const auto &Aweeinv_dble = openqcd_little__Aweeinv_dble;
const auto &Awooinv_dble = openqcd_little__Awooinv_dble;
const auto &Awoe_dble = openqcd_little__Awoe_dble;
const auto &Aweo_dble = openqcd_little__Aweo_dble;
const auto &Awhat_dble = openqcd_little__Awhat_dble;

// AW_GEN_C
const auto &gather_ud = openqcd_little__gather_ud;
const auto &gather_sd = openqcd_little__gather_sd;
const auto &apply_u2sd = openqcd_little__apply_u2sd;
const auto &apply_udag2sd = openqcd_little__apply_udag2sd;
const auto &spinor_prod_gamma = openqcd_little__spinor_prod_gamma;

// AW_OPS_C
const auto &Awop = openqcd_little__Awop;
const auto &Awophat = openqcd_little__Awophat;
const auto &Awop_dble = openqcd_little__Awop_dble;
const auto &Awophat_dble = openqcd_little__Awophat_dble;
const auto &set_Aw = openqcd_little__set_Aw;
const auto &set_Awhat = openqcd_little__set_Awhat;

// LTL_MODES_C
const auto &set_ltl_modes = openqcd_little__set_ltl_modes;
const auto &ltl_matrix = openqcd_little__ltl_matrix;

} // namespace little
} // namespace openqcd

#endif // CPP_LITTLE_HPP
