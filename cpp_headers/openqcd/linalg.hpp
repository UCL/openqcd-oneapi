
/*******************************************************************************
 *
 * File linalg.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_LINALG_HPP
#define CPP_LINALG_HPP

extern "C" {
#include "c_headers/linalg.h"
}

namespace openqcd {
namespace linalg {

// CMATRIX_C
const auto &cmat_vec = openqcd_linalg__cmat_vec;
const auto &cmat_vec_assign = openqcd_linalg__cmat_vec_assign;
const auto &cmat_add = openqcd_linalg__cmat_add;
const auto &cmat_sub = openqcd_linalg__cmat_sub;
const auto &cmat_mul = openqcd_linalg__cmat_mul;
const auto &cmat_dag = openqcd_linalg__cmat_dag;

// CMATRIX_DBLE_C
const auto &cmat_vec_dble = openqcd_linalg__cmat_vec_dble;
const auto &cmat_vec_assign_dble = openqcd_linalg__cmat_vec_assign_dble;
const auto &cmat_add_dble = openqcd_linalg__cmat_add_dble;
const auto &cmat_sub_dble = openqcd_linalg__cmat_sub_dble;
const auto &cmat_mul_dble = openqcd_linalg__cmat_mul_dble;
const auto &cmat_dag_dble = openqcd_linalg__cmat_dag_dble;
const auto &cmat_inv_dble = openqcd_linalg__cmat_inv_dble;

// LIEALG_C
const auto &random_alg = openqcd_linalg__random_alg;
const auto &norm_square_alg = openqcd_linalg__norm_square_alg;
const auto &scalar_prod_alg = openqcd_linalg__scalar_prod_alg;
const auto &set_alg2zero = openqcd_linalg__set_alg2zero;
const auto &set_ualg2zero = openqcd_linalg__set_ualg2zero;
const auto &assign_alg2alg = openqcd_linalg__assign_alg2alg;
const auto &swap_alg = openqcd_linalg__swap_alg;
const auto &add_alg = openqcd_linalg__add_alg;
const auto &muladd_assign_alg = openqcd_linalg__muladd_assign_alg;
const auto &project_to_su3alg = openqcd_linalg__project_to_su3alg;
const auto &su3alg_to_cm3x3 = openqcd_linalg__su3alg_to_cm3x3;

// SALG_C
const auto &spinor_prod = openqcd_linalg__spinor_prod;
const auto &spinor_prod_re = openqcd_linalg__spinor_prod_re;
const auto &norm_square = openqcd_linalg__norm_square;
const auto &mulc_spinor_add = openqcd_linalg__mulc_spinor_add;
const auto &mulr_spinor_add = openqcd_linalg__mulr_spinor_add;
const auto &project = openqcd_linalg__project;
const auto &scale = openqcd_linalg__scale;
const auto &normalize = openqcd_linalg__normalize;
const auto &rotate = openqcd_linalg__rotate;
const auto &mulg5 = openqcd_linalg__mulg5;
const auto &mulmg5 = openqcd_linalg__mulmg5;

// SALG_DBLE_C
const auto &spinor_prod_dble = openqcd_linalg__spinor_prod_dble;
const auto &spinor_prod_re_dble = openqcd_linalg__spinor_prod_re_dble;
const auto &spinor_prod5_dble = openqcd_linalg__spinor_prod5_dble;
const auto &norm_square_dble = openqcd_linalg__norm_square_dble;
const auto& mulr_spinor_assign_dble = openqcd_linalg__mulr_spinor_assign_dble;
const auto &mulc_spinor_add_dble = openqcd_linalg__mulc_spinor_add_dble;
const auto &mulr_spinor_add_dble = openqcd_linalg__mulr_spinor_add_dble;
const auto &combine_spinor_dble = openqcd_linalg__combine_spinor_dble;
const auto &project_dble = openqcd_linalg__project_dble;
const auto &scale_dble = openqcd_linalg__scale_dble;
const auto &normalize_dble = openqcd_linalg__normalize_dble;
const auto &rotate_dble = openqcd_linalg__rotate_dble;
const auto &mulg5_dble = openqcd_linalg__mulg5_dble;
const auto &mulmg5_dble = openqcd_linalg__mulmg5_dble;

// VALG_C
const auto &vprod = openqcd_linalg__vprod;
const auto &vnorm_square = openqcd_linalg__vnorm_square;
const auto &mulc_vadd = openqcd_linalg__mulc_vadd;
const auto &vproject = openqcd_linalg__vproject;
const auto &vscale = openqcd_linalg__vscale;
const auto &vnormalize = openqcd_linalg__vnormalize;
const auto &vrotate = openqcd_linalg__vrotate;

// VALG_DBLE_C
const auto &vprod_dble = openqcd_linalg__vprod_dble;
const auto &vnorm_square_dble = openqcd_linalg__vnorm_square_dble;
const auto &mulc_vadd_dble = openqcd_linalg__mulc_vadd_dble;
const auto &vproject_dble = openqcd_linalg__vproject_dble;
const auto &vscale_dble = openqcd_linalg__vscale_dble;
const auto &vnormalize_dble = openqcd_linalg__vnormalize_dble;
const auto &vrotate_dble = openqcd_linalg__vrotate_dble;

} // namespace linalg
} // namespace openqcd

#endif
