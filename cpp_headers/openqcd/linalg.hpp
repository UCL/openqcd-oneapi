
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
OPENQCD_MODULE_FUNCTION_ALIAS(cmat_vec, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(cmat_vec_assign, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(cmat_add, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(cmat_sub, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(cmat_mul, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(cmat_dag, linalg)

// CMATRIX_DBLE_C
OPENQCD_MODULE_FUNCTION_ALIAS(cmat_vec_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(cmat_vec_assign_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(cmat_add_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(cmat_sub_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(cmat_mul_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(cmat_dag_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(cmat_inv_dble, linalg)

// LIEALG_C
OPENQCD_MODULE_FUNCTION_ALIAS(random_alg, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(norm_square_alg, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(scalar_prod_alg, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(set_alg2zero, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(set_ualg2zero, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(assign_alg2alg, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(swap_alg, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(add_alg, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(muladd_assign_alg, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(project_to_su3alg, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(su3alg_to_cm3x3, linalg)

// SALG_C
OPENQCD_MODULE_FUNCTION_ALIAS(spinor_prod, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(spinor_prod_re, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(norm_square, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(mulc_spinor_add, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(mulr_spinor_add, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(project, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(scale, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(normalize, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(rotate, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(mulg5, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(mulmg5, linalg)

// SALG_DBLE_C
OPENQCD_MODULE_FUNCTION_ALIAS(spinor_prod_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(spinor_prod_re_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(spinor_prod5_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(norm_square_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(mulr_spinor_assign_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(mulc_spinor_add_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(mulr_spinor_add_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(combine_spinor_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(project_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(scale_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(normalize_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(rotate_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(mulg5_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(mulmg5_dble, linalg)

// VALG_C
OPENQCD_MODULE_FUNCTION_ALIAS(vprod, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(vnorm_square, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(mulc_vadd, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(vproject, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(vscale, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(vnormalize, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(vrotate, linalg)

// VALG_DBLE_C
OPENQCD_MODULE_FUNCTION_ALIAS(vprod_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(vnorm_square_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(mulc_vadd_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(vproject_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(vscale_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(vnormalize_dble, linalg)
OPENQCD_MODULE_FUNCTION_ALIAS(vrotate_dble, linalg)

} // namespace linalg
} // namespace openqcd

#endif
