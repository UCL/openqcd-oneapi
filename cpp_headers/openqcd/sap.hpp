
/*******************************************************************************
 *
 * File sap.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_SAP_HPP
#define CPP_SAP_HPP

extern "C" {
#include "c_headers/sap.h"
}

namespace openqcd {
namespace sap {

// BLK_SOLV_C
const auto &blk_mres = openqcd_sap__blk_mres;
const auto &blk_eo_mres = openqcd_sap__blk_eo_mres;

// SAP_COM_C
const auto &sap_com = openqcd_sap__sap_com;

/* SAP */
const auto &sap = openqcd_sap__sap;

/* SAP_GCR */
const auto &sap_gcr = openqcd_sap__sap_gcr;

} // namespace sap
} // namespace openqcd

#endif // CPP_SAP_HPP
