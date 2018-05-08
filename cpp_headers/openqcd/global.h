
/*******************************************************************************
 *
 * File global.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Global parameters and arrays
 *
 *******************************************************************************/

#ifndef CPP_GLOBAL_HPP
#define CPP_GLOBAL_HPP

extern "C" {
#include "c_headers/global.h"
}

namespace openqcd {

const auto& NPROC0 = openqcd__NPROC0;
const auto& NPROC1 = openqcd__NPROC1;
const auto& NPROC2 = openqcd__NPROC2;
const auto& NPROC3 = openqcd__NPROC3;

const auto& L0 = openqcd__L0;
const auto& L1 = openqcd__L1;
const auto& L2 = openqcd__L2;
const auto& L3 = openqcd__L3;

const auto& NPROC0_BLK = openqcd__NPROC0_BLK;
const auto& NPROC1_BLK = openqcd__NPROC1_BLK;
const auto& NPROC2_BLK = openqcd__NPROC2_BLK;
const auto& NPROC3_BLK = openqcd__NPROC3_BLK;

const auto& NPROC = openqcd__NPROC;
const auto& VOLUME = openqcd__VOLUME;
const auto& FACE0 = openqcd__FACE0;
const auto& FACE1 = openqcd__FACE1;
const auto& FACE2 = openqcd__FACE2;
const auto& FACE3 = openqcd__FACE3;
const auto& BNDRY = openqcd__BNDRY;
const auto& NSPIN = openqcd__NSPIN;

const auto& cpr = openqcd__cpr;
const auto& npr = openqcd__npr;

const auto& ipt = openqcd__ipt;
const auto& iup = openqcd__iup;
const auto& idn = openqcd__idn;
const auto& map = openqcd__map;

const auto& set_lattice_sizes = openqcd__set_lattice_sizes;

} // namespace openqcd 

#ifndef CPP_SU3_HPP
#include "su3.hpp"
#endif

#endif /* CPP_GLOBAL_HPP */
