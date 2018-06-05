
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

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/random.h"
}

namespace openqcd {
namespace random {

// GAUSS_C
OPENQCD_MODULE_FUNCTION_ALIAS(gauss, random)
OPENQCD_MODULE_FUNCTION_ALIAS(gauss_dble, random)

// RANLUX_C
OPENQCD_MODULE_FUNCTION_ALIAS(start_ranlux, random)
OPENQCD_MODULE_FUNCTION_ALIAS(export_ranlux, random)
OPENQCD_MODULE_FUNCTION_ALIAS(import_ranlux, random)

// RANLXS_C
OPENQCD_MODULE_FUNCTION_ALIAS(ranlxs, random)
OPENQCD_MODULE_FUNCTION_ALIAS(rlxs_init, random)
OPENQCD_MODULE_FUNCTION_ALIAS(rlxs_size, random)
OPENQCD_MODULE_FUNCTION_ALIAS(rlxs_get, random)
OPENQCD_MODULE_FUNCTION_ALIAS(rlxs_reset, random)

// RANLXD_C
OPENQCD_MODULE_FUNCTION_ALIAS(ranlxd, random)
OPENQCD_MODULE_FUNCTION_ALIAS(rlxd_init, random)
OPENQCD_MODULE_FUNCTION_ALIAS(rlxd_size, random)
OPENQCD_MODULE_FUNCTION_ALIAS(rlxd_get, random)
OPENQCD_MODULE_FUNCTION_ALIAS(rlxd_reset, random)

// RANLUX_SITE_C
OPENQCD_MODULE_FUNCTION_ALIAS(ranlxs_site, random)
OPENQCD_MODULE_FUNCTION_ALIAS(ranlxd_site, random)
OPENQCD_MODULE_FUNCTION_ALIAS(start_ranlux_site, random)

} // namespace random
} // namespace openqcd

#endif // CPP_RANDOM_HPP
