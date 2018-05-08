
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

extern "C" {
#include "c_headers/utils.h"
}

namespace openqcd {
namespace utils {

using stdint_t = openqcd_utils__stdint_t;
using stduint_t = openqcd_utils__stduint_t;
using ptset_t = openqcd_utils__ptset_t;

// ENDIAN_C
const auto &endianness = openqcd_utils__endianness;
const auto &bswap_int = openqcd_utils__bswap_int;
const auto &bswap_double = openqcd_utils__bswap_double;

// ERROR_C
const auto &set_error_file = openqcd_utils__set_error_file;
const auto &error = openqcd_utils__error;
const auto &error_root = openqcd_utils__error_root;
const auto &error_loc = openqcd_utils__error_loc;

// HSUM_C
const auto &init_hsum = openqcd_utils__init_hsum;
const auto &reset_hsum = openqcd_utils__reset_hsum;
const auto &add_to_hsum = openqcd_utils__add_to_hsum;
const auto &local_hsum = openqcd_utils__local_hsum;
const auto &global_hsum = openqcd_utils__global_hsum;

// MUTILS_C
const auto &find_opt = openqcd_utils__find_opt;
const auto &fdigits = openqcd_utils__fdigits;
const auto &check_dir = openqcd_utils__check_dir;
const auto &check_dir_root = openqcd_utils__check_dir_root;
const auto &name_size = openqcd_utils__name_size;
const auto &find_section = openqcd_utils__find_section;
const auto &find_optional_section = openqcd_utils__find_optional_section;
const auto &read_line = openqcd_utils__read_line;
const auto &read_optional_line = openqcd_utils__read_optional_line;
const auto &count_tokens = openqcd_utils__count_tokens;
const auto &read_iprms = openqcd_utils__read_iprms;
const auto &read_optional_iprms = openqcd_utils__read_optional_iprms;
const auto &read_dprms = openqcd_utils__read_dprms;
const auto &read_optional_dprms = openqcd_utils__read_optional_dprms;
const auto &copy_file = openqcd_utils__copy_file;

const auto &No_Section_Found = openqcd_utils__No_Section_Found;

// UTILS_C
const auto &safe_mod = openqcd_utils__safe_mod;
const auto &amalloc = openqcd_utils__amalloc;
const auto &afree = openqcd_utils__afree;
const auto &amem_use_mb = openqcd_utils__amem_use_mb;
const auto &amem_max_mb = openqcd_utils__amem_max_mb;
const auto &mpi_permanent_tag = openqcd_utils__mpi_permanent_tag;
const auto &mpi_tag = openqcd_utils__mpi_tag;
const auto &message = openqcd_utils__message;
const auto &mpc_bcast_c = openqcd_utils__mpc_bcast_c;
const auto &mpc_bcast_d = openqcd_utils__mpc_bcast_d;
const auto &mpc_bcast_i = openqcd_utils__mpc_bcast_i;
const auto &mpc_gsum_d = openqcd_utils__mpc_gsum_d;
const auto &mpc_print_info = openqcd_utils__mpc_print_info;
const auto &square_dble = openqcd_utils__square_dble;
const auto &sinc_dble = openqcd_utils__sinc_dble;
const auto &smear_xi0_dble = openqcd_utils__smear_xi0_dble;
const auto &smear_xi1_dble = openqcd_utils__smear_xi1_dble;
const auto &mul_icomplex = openqcd_utils__mul_icomplex;
const auto &mul_assign_scalar_complex =
    openqcd_utils__mul_assign_scalar_complex;
const auto &is_equal_f = openqcd_utils__is_equal_f;
const auto &not_equal_f = openqcd_utils__not_equal_f;
const auto &is_equal_d = openqcd_utils__is_equal_d;
const auto &not_equal_d = openqcd_utils__not_equal_d;

// WSPACE_C
const auto &alloc_wud = openqcd_utils__alloc_wud;
const auto &reserve_wud = openqcd_utils__reserve_wud;
const auto &release_wud = openqcd_utils__release_wud;
const auto &wud_size = openqcd_utils__wud_size;
const auto &alloc_wfd = openqcd_utils__alloc_wfd;
const auto &reserve_wfd = openqcd_utils__reserve_wfd;
const auto &release_wfd = openqcd_utils__release_wfd;
const auto &wfd_size = openqcd_utils__wfd_size;
const auto &alloc_ws = openqcd_utils__alloc_ws;
const auto &reserve_ws = openqcd_utils__reserve_ws;
const auto &release_ws = openqcd_utils__release_ws;
const auto &ws_size = openqcd_utils__ws_size;
const auto &alloc_wsd = openqcd_utils__alloc_wsd;
const auto &reserve_wsd = openqcd_utils__reserve_wsd;
const auto &release_wsd = openqcd_utils__release_wsd;
const auto &wsd_size = openqcd_utils__wsd_size;
const auto &alloc_wv = openqcd_utils__alloc_wv;
const auto &reserve_wv = openqcd_utils__reserve_wv;
const auto &release_wv = openqcd_utils__release_wv;
const auto &wv_size = openqcd_utils__wv_size;
const auto &alloc_wvd = openqcd_utils__alloc_wvd;
const auto &reserve_wvd = openqcd_utils__reserve_wvd;
const auto &release_wvd = openqcd_utils__release_wvd;
const auto &wvd_size = openqcd_utils__wvd_size;

} // namespace utils
} // namespace openqcd

#endif
