
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

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/sflds.h"
}

namespace openqcd {
namespace sflds {

// PBND_C
OPENQCD_MODULE_FUNCTION_ALIAS(assign_s2w, sflds)
OPENQCD_MODULE_FUNCTION_ALIAS(add_assign_w2s, sflds)
OPENQCD_MODULE_FUNCTION_ALIAS(sub_assign_w2s, sflds)
OPENQCD_MODULE_FUNCTION_ALIAS(mulg5_sub_assign_w2s, sflds)

// PBND_DBLE_C
OPENQCD_MODULE_FUNCTION_ALIAS(assign_sd2wd, sflds)
OPENQCD_MODULE_FUNCTION_ALIAS(add_assign_wd2sd, sflds)
OPENQCD_MODULE_FUNCTION_ALIAS(sub_assign_wd2sd, sflds)
OPENQCD_MODULE_FUNCTION_ALIAS(mulg5_sub_assign_wd2sd, sflds)

// SFLDS_C
OPENQCD_MODULE_FUNCTION_ALIAS(set_s2zero, sflds)
OPENQCD_MODULE_FUNCTION_ALIAS(set_sd2zero, sflds)
OPENQCD_MODULE_FUNCTION_ALIAS(random_s, sflds)
OPENQCD_MODULE_FUNCTION_ALIAS(random_sd, sflds)
OPENQCD_MODULE_FUNCTION_ALIAS(assign_s2s, sflds)
OPENQCD_MODULE_FUNCTION_ALIAS(assign_s2sd, sflds)
OPENQCD_MODULE_FUNCTION_ALIAS(assign_sd2s, sflds)
OPENQCD_MODULE_FUNCTION_ALIAS(assign_sd2sd, sflds)
OPENQCD_MODULE_FUNCTION_ALIAS(diff_s2s, sflds)
OPENQCD_MODULE_FUNCTION_ALIAS(add_s2sd, sflds)
OPENQCD_MODULE_FUNCTION_ALIAS(diff_sd2s, sflds)

// SCOM_C
OPENQCD_MODULE_FUNCTION_ALIAS(cps_int_bnd, sflds)
OPENQCD_MODULE_FUNCTION_ALIAS(cps_ext_bnd, sflds)

// SDCOM_C
OPENQCD_MODULE_FUNCTION_ALIAS(cpsd_int_bnd, sflds)
OPENQCD_MODULE_FUNCTION_ALIAS(cpsd_ext_bnd, sflds)

} // namespace sflds
} // namespace openqcd

#endif // CPP_SFLDS_HPP
