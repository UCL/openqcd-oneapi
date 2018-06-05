
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

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/sap.h"
}

namespace openqcd {
namespace sap {

// BLK_SOLV_C
OPENQCD_MODULE_FUNCTION_ALIAS(blk_mres, sap)
OPENQCD_MODULE_FUNCTION_ALIAS(blk_eo_mres, sap)

// SAP_COM_C
OPENQCD_MODULE_FUNCTION_ALIAS(sap_com, sap)

/* SAP */
OPENQCD_MODULE_FUNCTION_ALIAS(sap, sap)

/* SAP_GCR */
OPENQCD_MODULE_FUNCTION_ALIAS(sap_gcr, sap)

} // namespace sap
} // namespace openqcd

#endif // CPP_SAP_HPP
