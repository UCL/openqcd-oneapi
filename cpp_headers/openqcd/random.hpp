
/*******************************************************************************
 *
 * File random.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_RANDOM_HPP
#define CPP_RANDOM_HPP

extern "C" {
#include "c_headers/random.h"
}

namespace openqcd {
namespace random {

// GAUSS_C
const auto &gauss = openqcd_random__gauss;
const auto &gauss_dble = openqcd_random__gauss_dble;

// RANLUX_C
const auto &start_ranlux = openqcd_random__start_ranlux;
const auto &export_ranlux = openqcd_random__export_ranlux;
const auto &import_ranlux = openqcd_random__import_ranlux;

// RANLXS_C
const auto &ranlxs = openqcd_random__ranlxs;
const auto &rlxs_init = openqcd_random__rlxs_init;
const auto &rlxs_size = openqcd_random__rlxs_size;
const auto &rlxs_get = openqcd_random__rlxs_get;
const auto &rlxs_reset = openqcd_random__rlxs_reset;

// RANLXD_C
const auto &ranlxd = openqcd_random__ranlxd;
const auto &rlxd_init = openqcd_random__rlxd_init;
const auto &rlxd_size = openqcd_random__rlxd_size;
const auto &rlxd_get = openqcd_random__rlxd_get;
const auto &rlxd_reset = openqcd_random__rlxd_reset;

// RANLUX_SITE_C
const auto &ranlxs_site = openqcd_random__ranlxs_site;
const auto &ranlxd_site = openqcd_random__ranlxd_site;
const auto &start_ranlux_site = openqcd_random__start_ranlux_site;

} // namespace random
} // namespace openqcd

#endif // CPP_RANDOM_HPP
