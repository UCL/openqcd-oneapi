
/*******************************************************************************
 *
 * File lattice.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_LATTICE_HPP
#define CPP_LATTICE_HPP

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/lattice.h"
}

namespace openqcd {
namespace lattice {

using uidx_t = openqcd_lattice__uidx_t;
using ftidx_t = openqcd_lattice__ftidx_t;

// BCNDS_C
OPENQCD_MODULE_FUNCTION_ALIAS(bnd_lks, lattice)
OPENQCD_MODULE_FUNCTION_ALIAS(bnd_bnd_lks, lattice)
OPENQCD_MODULE_FUNCTION_ALIAS(bnd_pts, lattice)
OPENQCD_MODULE_FUNCTION_ALIAS(set_bc, lattice)
OPENQCD_MODULE_FUNCTION_ALIAS(check_bc, lattice)
OPENQCD_MODULE_FUNCTION_ALIAS(bnd_s2zero, lattice)
OPENQCD_MODULE_FUNCTION_ALIAS(bnd_sd2zero, lattice)

// FTIDX_C
OPENQCD_MODULE_FUNCTION_ALIAS(ftidx, lattice)
OPENQCD_MODULE_FUNCTION_ALIAS(plaq_ftidx, lattice)

// GEOMETRY_C
OPENQCD_MODULE_FUNCTION_ALIAS(ipr_global, lattice)
OPENQCD_MODULE_FUNCTION_ALIAS(ipt_global, lattice)
OPENQCD_MODULE_FUNCTION_ALIAS(global_time, lattice)
OPENQCD_MODULE_FUNCTION_ALIAS(geometry, lattice)

// UIDX_C
OPENQCD_MODULE_FUNCTION_ALIAS(uidx, lattice)
OPENQCD_MODULE_FUNCTION_ALIAS(alloc_uidx, lattice)
OPENQCD_MODULE_FUNCTION_ALIAS(plaq_uidx, lattice)

} // namespace lattice
} // namespace openqcd

#endif
