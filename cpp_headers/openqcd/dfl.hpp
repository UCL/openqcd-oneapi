
/*******************************************************************************
 *
 * File dfl.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_DFL_HPP
#define CPP_DFL_HPP

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/dfl.h"
}

namespace openqcd {
namespace dfl {

using dfl_grid_t = openqcd_dfl__dfl_grid_t;

// DFL_GEOMETRY_C
OPENQCD_MODULE_FUNCTION_ALIAS(dfl_geometry, dfl)

// DFL_MODES_C
OPENQCD_MODULE_FUNCTION_ALIAS(dfl_modes, dfl)
OPENQCD_MODULE_FUNCTION_ALIAS(dfl_update, dfl)
OPENQCD_MODULE_FUNCTION_ALIAS(dfl_modes2, dfl)
OPENQCD_MODULE_FUNCTION_ALIAS(dfl_update2, dfl)

// DFL_SAP_GCR_C
OPENQCD_MODULE_FUNCTION_ALIAS(dfl_sap_gcr, dfl)
OPENQCD_MODULE_FUNCTION_ALIAS(dfl_sap_gcr2, dfl)

// DFL_SUBSPACE_C
OPENQCD_MODULE_FUNCTION_ALIAS(dfl_sd2vd, dfl)
OPENQCD_MODULE_FUNCTION_ALIAS(dfl_vd2sd, dfl)
OPENQCD_MODULE_FUNCTION_ALIAS(dfl_sub_vd2sd, dfl)
OPENQCD_MODULE_FUNCTION_ALIAS(dfl_s2v, dfl)
OPENQCD_MODULE_FUNCTION_ALIAS(dfl_v2s, dfl)
OPENQCD_MODULE_FUNCTION_ALIAS(dfl_sub_v2s, dfl)
OPENQCD_MODULE_FUNCTION_ALIAS(dfl_subspace, dfl)

/* LTL_GCR */
OPENQCD_MODULE_FUNCTION_ALIAS(ltl_gcr, dfl)

} // namespace dfl
} // namespace openqcd

#endif // ifndef CPP_DFL_HPP
