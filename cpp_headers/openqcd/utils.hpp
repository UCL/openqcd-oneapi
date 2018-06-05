
/*******************************************************************************
 *
 * File utils.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_UTILS_HPP
#define CPP_UTILS_HPP

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/utils.h"
}

namespace openqcd {
namespace utils {

using stdint_t = openqcd_utils__stdint_t;
using stduint_t = openqcd_utils__stduint_t;
using ptset_t = openqcd_utils__ptset_t;

// ENDIAN_C
OPENQCD_MODULE_FUNCTION_ALIAS(endianness, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(bswap_int, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(bswap_double, utils)

// ERROR_C
OPENQCD_MODULE_FUNCTION_ALIAS(set_error_file, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(error, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(error_root, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(error_loc, utils)

// HSUM_C
OPENQCD_MODULE_FUNCTION_ALIAS(init_hsum, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(reset_hsum, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(add_to_hsum, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(local_hsum, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(global_hsum, utils)

// MUTILS_C
OPENQCD_MODULE_FUNCTION_ALIAS(find_opt, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(fdigits, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(check_dir, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(check_dir_root, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(name_size, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(find_section, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(find_optional_section, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(read_line, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(read_optional_line, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(count_tokens, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(read_iprms, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(read_optional_iprms, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(read_dprms, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(read_optional_dprms, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(copy_file, utils)

// UTILS_C
OPENQCD_MODULE_FUNCTION_ALIAS(safe_mod, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(amalloc, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(afree, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(amem_use_mb, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(amem_max_mb, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(mpi_permanent_tag, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(mpi_tag, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(message, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(mpc_bcast_c, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(mpc_bcast_d, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(mpc_bcast_i, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(mpc_gsum_d, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(mpc_print_info, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(square_dble, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(sinc_dble, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(smear_xi0_dble, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(smear_xi1_dble, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(mul_icomplex, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(mul_assign_scalar_complex, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(is_equal_f, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(not_equal_f, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(is_equal_d, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(not_equal_d, utils)

// WSPACE_C
OPENQCD_MODULE_FUNCTION_ALIAS(alloc_wud, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(reserve_wud, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(release_wud, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(wud_size, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(alloc_wfd, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(reserve_wfd, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(release_wfd, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(wfd_size, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(alloc_ws, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(reserve_ws, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(release_ws, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(ws_size, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(alloc_wsd, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(reserve_wsd, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(release_wsd, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(wsd_size, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(alloc_wv, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(reserve_wv, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(release_wv, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(wv_size, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(alloc_wvd, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(reserve_wvd, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(release_wvd, utils)
OPENQCD_MODULE_FUNCTION_ALIAS(wvd_size, utils)

} // namespace utils
} // namespace openqcd

#endif
