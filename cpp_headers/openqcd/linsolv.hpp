
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

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/linsolv.h"
}

namespace openqcd {
namespace linsolv {

/* CGNE_C */
OPENQCD_MODULE_FUNCTION_ALIAS(cgne, linsolv)

/* FGCR4VD_C */
OPENQCD_MODULE_FUNCTION_ALIAS(fgcr4vd, linsolv)

/* FGCR_C */
OPENQCD_MODULE_FUNCTION_ALIAS(fgcr, linsolv)

/* MSCG_C */
OPENQCD_MODULE_FUNCTION_ALIAS(mscg, linsolv)

} // namespace linsolv
} // namespace openqcd

#endif
