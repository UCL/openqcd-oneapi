
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

extern "C" {
#include "c_headers/lattice.h"
}

namespace openqcd {
namespace lattice {

using uidx_t = openqcd_lattice__uidx_t;
using ftidx_t = openqcd_lattice__ftidx_t;

// BCNDS_C
const auto &bnd_lks = openqcd_lattice__bnd_lks;
const auto &bnd_bnd_lks = openqcd_lattice__bnd_bnd_lks;
const auto &bnd_pts = openqcd_lattice__bnd_pts;
const auto &set_bc = openqcd_lattice__set_bc;
const auto &check_bc = openqcd_lattice__check_bc;
const auto &bnd_s2zero = openqcd_lattice__bnd_s2zero;
const auto &bnd_sd2zero = openqcd_lattice__bnd_sd2zero;

// FTIDX_C
const auto &ftidx = openqcd_lattice__ftidx;
const auto &plaq_ftidx = openqcd_lattice__plaq_ftidx;

// GEOMETRY_C
const auto &ipr_global = openqcd_lattice__ipr_global;
const auto &ipt_global = openqcd_lattice__ipt_global;
const auto &global_time = openqcd_lattice__global_time;
const auto &geometry = openqcd_lattice__geometry;

// UIDX_C
const auto &uidx = openqcd_lattice__uidx;
const auto &alloc_uidx = openqcd_lattice__alloc_uidx;
const auto &plaq_uidx = openqcd_lattice__plaq_uidx;

} // namespace lattice
} // namespace openqcd

#endif
