
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

extern "C" {
#include "c_headers/dfl.h"
}

namespace openqcd {
namespace dfl {

using dfl_grid_t = openqcd_dfl__dfl_grid_t;

// DFL_GEOMETRY_C
const auto& dfl_geometry = openqcd_dfl__dfl_geometry;

// DFL_MODES_C
const auto& dfl_modes = openqcd_dfl__dfl_modes;
const auto& dfl_update = openqcd_dfl__dfl_update;
const auto& dfl_modes2 = openqcd_dfl__dfl_modes2;
const auto& dfl_update2 = openqcd_dfl__dfl_update2;

// DFL_SAP_GCR_C
const auto& dfl_sap_gcr = openqcd_dfl__dfl_sap_gcr;
const auto& dfl_sap_gcr2 = openqcd_dfl__dfl_sap_gcr2;

// DFL_SUBSPACE_C
const auto& dfl_sd2vd = openqcd_dfl__dfl_sd2vd;
const auto& dfl_vd2sd = openqcd_dfl__dfl_vd2sd;
const auto& dfl_sub_vd2sd = openqcd_dfl__dfl_sub_vd2sd;
const auto& dfl_s2v = openqcd_dfl__dfl_s2v;
const auto& dfl_v2s = openqcd_dfl__dfl_v2s;
const auto& dfl_sub_v2s = openqcd_dfl__dfl_sub_v2s;
const auto& dfl_subspace = openqcd_dfl__dfl_subspace;

/* LTL_GCR */
const auto& ltl_gcr = openqcd_dfl__ltl_gcr;

} // namespace dfl 
} // namespace openqcd 

#endif // ifndef CPP_DFL_HPP
