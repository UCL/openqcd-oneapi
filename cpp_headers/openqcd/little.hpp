
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

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/little.h"
}

namespace openqcd {
namespace little {

using Aw_t = openqcd_little__Aw_t;
using Aw_dble_t = openqcd_little__Aw_dble_t;
using b2b_flds_t = openqcd_little__b2b_flds_t;

// AW_COM_C
OPENQCD_MODULE_FUNCTION_ALIAS(b2b_flds, little)
OPENQCD_MODULE_FUNCTION_ALIAS(cpAoe_ext_bnd, little)
OPENQCD_MODULE_FUNCTION_ALIAS(cpAee_int_bnd, little)

// AW_C
OPENQCD_MODULE_FUNCTION_ALIAS(Aw, little)
OPENQCD_MODULE_FUNCTION_ALIAS(Aweeinv, little)
OPENQCD_MODULE_FUNCTION_ALIAS(Awooinv, little)
OPENQCD_MODULE_FUNCTION_ALIAS(Awoe, little)
OPENQCD_MODULE_FUNCTION_ALIAS(Aweo, little)
OPENQCD_MODULE_FUNCTION_ALIAS(Awhat, little)

// AW_DBLE_C
OPENQCD_MODULE_FUNCTION_ALIAS(Aw_dble, little)
OPENQCD_MODULE_FUNCTION_ALIAS(Aweeinv_dble, little)
OPENQCD_MODULE_FUNCTION_ALIAS(Awooinv_dble, little)
OPENQCD_MODULE_FUNCTION_ALIAS(Awoe_dble, little)
OPENQCD_MODULE_FUNCTION_ALIAS(Aweo_dble, little)
OPENQCD_MODULE_FUNCTION_ALIAS(Awhat_dble, little)

// AW_GEN_C
OPENQCD_MODULE_FUNCTION_ALIAS(gather_ud, little)
OPENQCD_MODULE_FUNCTION_ALIAS(gather_sd, little)
OPENQCD_MODULE_FUNCTION_ALIAS(apply_u2sd, little)
OPENQCD_MODULE_FUNCTION_ALIAS(apply_udag2sd, little)
OPENQCD_MODULE_FUNCTION_ALIAS(spinor_prod_gamma, little)

// AW_OPS_C
OPENQCD_MODULE_FUNCTION_ALIAS(Awop, little)
OPENQCD_MODULE_FUNCTION_ALIAS(Awophat, little)
OPENQCD_MODULE_FUNCTION_ALIAS(Awop_dble, little)
OPENQCD_MODULE_FUNCTION_ALIAS(Awophat_dble, little)
OPENQCD_MODULE_FUNCTION_ALIAS(set_Aw, little)
OPENQCD_MODULE_FUNCTION_ALIAS(set_Awhat, little)

// LTL_MODES_C
OPENQCD_MODULE_FUNCTION_ALIAS(set_ltl_modes, little)
OPENQCD_MODULE_FUNCTION_ALIAS(ltl_matrix, little)

} // namespace little
} // namespace openqcd

#endif // CPP_LITTLE_HPP
