
/*******************************************************************************
 *
 * File dirac.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_DIRAC_HPP
#define CPP_DIRAC_HPP

extern "C" {
#include "c_headers/dirac.h"
}

namespace openqcd {
namespace dirac {

// DW_BND_C
const auto &Dw_bnd = openqcd_dirac__Dw_bnd;

// DW_C
const auto &Dw = openqcd_dirac__Dw;
const auto &Dwee = openqcd_dirac__Dwee;
const auto &Dwoo = openqcd_dirac__Dwoo;
const auto &Dweo = openqcd_dirac__Dweo;
const auto &Dwoe = openqcd_dirac__Dwoe;
const auto &Dwhat = openqcd_dirac__Dwhat;
const auto &Dw_blk = openqcd_dirac__Dw_blk;
const auto &Dwee_blk = openqcd_dirac__Dwee_blk;
const auto &Dwoo_blk = openqcd_dirac__Dwoo_blk;
const auto &Dwoe_blk = openqcd_dirac__Dwoe_blk;
const auto &Dweo_blk = openqcd_dirac__Dweo_blk;
const auto &Dwhat_blk = openqcd_dirac__Dwhat_blk;

// DW_DBLE_C
const auto &Dw_dble = openqcd_dirac__Dw_dble;
const auto &Dwee_dble = openqcd_dirac__Dwee_dble;
const auto &Dwoo_dble = openqcd_dirac__Dwoo_dble;
const auto &Dweo_dble = openqcd_dirac__Dweo_dble;
const auto &Dwoe_dble = openqcd_dirac__Dwoe_dble;
const auto &Dwhat_dble = openqcd_dirac__Dwhat_dble;
const auto &Dw_blk_dble = openqcd_dirac__Dw_blk_dble;
const auto &Dwee_blk_dble = openqcd_dirac__Dwee_blk_dble;
const auto &Dwoo_blk_dble = openqcd_dirac__Dwoo_blk_dble;
const auto &Dwoe_blk_dble = openqcd_dirac__Dwoe_blk_dble;
const auto &Dweo_blk_dble = openqcd_dirac__Dweo_blk_dble;
const auto &Dwhat_blk_dble = openqcd_dirac__Dwhat_blk_dble;

} // namespace dirac
} // namespace openqcd

#endif // ifndef CPP_DIRAC_HPP
