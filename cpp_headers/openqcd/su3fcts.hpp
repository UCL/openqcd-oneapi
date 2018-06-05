
/*******************************************************************************
 *
 * File su3fcts.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef SU3CPP_FCTS_HPP
#define SU3CPP_FCTS_HPP

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/su3fcts.h"
}

namespace openqcd {
namespace su3fcts {

using ch_drv0_t = openqcd_su3fcts__ch_drv0_t;
using ch_drv1_t = openqcd_su3fcts__ch_drv1_t;
using ch_drv2_t = openqcd_su3fcts__ch_drv2_t;

// CHEXP_C
OPENQCD_MODULE_FUNCTION_ALIAS(ch2mat, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(chexp_drv0, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(chexp_drv1, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(chexp_drv2, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(expXsu3, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(expXsu3_w_factors, su3fcts)

// CM3X3_C
OPENQCD_MODULE_FUNCTION_ALIAS(cm3x3_zero, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(cm3x3_unity, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(cm3x3_assign, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(cm3x3_swap, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(cm3x3_dagger, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(cm3x3_tr, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(cm3x3_retr, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(cm3x3_imtr, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(cm3x3_add, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(cm3x3_mul_add, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(cm3x3_mulr, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(cm3x3_mulr_add, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(cm3x3_mulc, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(cm3x3_mulc_add, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(cm3x3_lc1, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(cm3x3_lc2, su3fcts)

// RANDOM_SU3_C
OPENQCD_MODULE_FUNCTION_ALIAS(random_su3, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(random_su3_dble, su3fcts)

// SU3REN_C
OPENQCD_MODULE_FUNCTION_ALIAS(project_to_su3, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(project_to_su3_dble, su3fcts)

// SU3PROD_C
OPENQCD_MODULE_FUNCTION_ALIAS(su3xsu3, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(su3dagxsu3, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(su3xsu3dag, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(su3dagxsu3dag, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(su3xu3alg, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(su3dagxu3alg, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(u3algxsu3, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(u3algxsu3dag, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(prod2su3alg, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(prod2u3alg, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(rotate_su3alg, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(su3xsu3alg, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(su3algxsu3, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(su3dagxsu3alg, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(su3algxsu3dag, su3fcts)
OPENQCD_MODULE_FUNCTION_ALIAS(su3algxsu3_tr, su3fcts)

} // namespace su3fcts
} // namespace openqcd

#endif // SU3CPP_FCTS_HPP
