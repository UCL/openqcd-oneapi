
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

extern "C" {
#include "c_headers/su3fcts.h"
}

namespace openqcd {
namespace su3fcts {

using ch_drv0_t = openqcd_su3fcts__ch_drv0_t;
using ch_drv1_t = openqcd_su3fcts__ch_drv1_t;
using ch_drv2_t = openqcd_su3fcts__ch_drv2_t;

// CHEXP_C
const auto &ch2mat = openqcd_su3fcts__ch2mat;
const auto &chexp_drv0 = openqcd_su3fcts__chexp_drv0;
const auto &chexp_drv1 = openqcd_su3fcts__chexp_drv1;
const auto &chexp_drv2 = openqcd_su3fcts__chexp_drv2;
const auto &expXsu3 = openqcd_su3fcts__expXsu3;
const auto &expXsu3_w_factors = openqcd_su3fcts__expXsu3_w_factors;

// CM3X3_C
const auto &cm3x3_zero = openqcd_su3fcts__cm3x3_zero;
const auto &cm3x3_unity = openqcd_su3fcts__cm3x3_unity;
const auto &cm3x3_assign = openqcd_su3fcts__cm3x3_assign;
const auto &cm3x3_swap = openqcd_su3fcts__cm3x3_swap;
const auto &cm3x3_dagger = openqcd_su3fcts__cm3x3_dagger;
const auto &cm3x3_tr = openqcd_su3fcts__cm3x3_tr;
const auto &cm3x3_retr = openqcd_su3fcts__cm3x3_retr;
const auto &cm3x3_imtr = openqcd_su3fcts__cm3x3_imtr;
const auto &cm3x3_add = openqcd_su3fcts__cm3x3_add;
const auto &cm3x3_mul_add = openqcd_su3fcts__cm3x3_mul_add;
const auto &cm3x3_mulr = openqcd_su3fcts__cm3x3_mulr;
const auto &cm3x3_mulr_add = openqcd_su3fcts__cm3x3_mulr_add;
const auto &cm3x3_mulc = openqcd_su3fcts__cm3x3_mulc;
const auto &cm3x3_mulc_add = openqcd_su3fcts__cm3x3_mulc_add;
const auto &cm3x3_lc1 = openqcd_su3fcts__cm3x3_lc1;
const auto &cm3x3_lc2 = openqcd_su3fcts__cm3x3_lc2;

// RANDOM_SU3_C
const auto &random_su3 = openqcd_su3fcts__random_su3;
const auto &random_su3_dble = openqcd_su3fcts__random_su3_dble;

// SU3REN_C
const auto &project_to_su3 = openqcd_su3fcts__project_to_su3;
const auto &project_to_su3_dble = openqcd_su3fcts__project_to_su3_dble;

// SU3PROD_C
const auto &su3xsu3 = openqcd_su3fcts__su3xsu3;
const auto &su3dagxsu3 = openqcd_su3fcts__su3dagxsu3;
const auto &su3xsu3dag = openqcd_su3fcts__su3xsu3dag;
const auto &su3dagxsu3dag = openqcd_su3fcts__su3dagxsu3dag;
const auto &su3xu3alg = openqcd_su3fcts__su3xu3alg;
const auto &su3dagxu3alg = openqcd_su3fcts__su3dagxu3alg;
const auto &u3algxsu3 = openqcd_su3fcts__u3algxsu3;
const auto &u3algxsu3dag = openqcd_su3fcts__u3algxsu3dag;
const auto &prod2su3alg = openqcd_su3fcts__prod2su3alg;
const auto &prod2u3alg = openqcd_su3fcts__prod2u3alg;
const auto &rotate_su3alg = openqcd_su3fcts__rotate_su3alg;
const auto &su3xsu3alg = openqcd_su3fcts__su3xsu3alg;
const auto &su3algxsu3 = openqcd_su3fcts__su3algxsu3;
const auto &su3dagxsu3alg = openqcd_su3fcts__su3dagxsu3alg;
const auto &su3algxsu3dag = openqcd_su3fcts__su3algxsu3dag;
const auto &su3algxsu3_tr = openqcd_su3fcts__su3algxsu3_tr;

} // namespace su3fcts
} // namespace openqcd

#endif // SU3CPP_FCTS_HPP
