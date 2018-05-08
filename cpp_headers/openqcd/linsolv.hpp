
/*******************************************************************************
 *
 * File linsolv.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_LINSOLV_HPP
#define CPP_LINSOLV_HPP

extern "C" {
#include "c_headers/linsolv.h"
}

namespace openqcd {
namespace linsolv {

/* CGNE_C */
const auto &cgne = openqcd_linsolv__cgne;

/* FGCR4VD_C */
const auto &fgcr4vd = openqcd_linsolv__fgcr4vd;

/* FGCR_C */
const auto &fgcr = openqcd_linsolv__fgcr;

/* MSCG_C */
const auto &mscg = openqcd_linsolv__mscg;

} // namespace linsolv
} // namespace openqcd

#endif
