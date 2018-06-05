
/*******************************************************************************
 *
 * File ratfcts.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_RATFCTS_HPP
#define CPP_RATFCTS_HPP

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/ratfcts.h"
}

namespace openqcd {
namespace ratfct {

using ratfct_t = openqcd_ratfcts__ratfct_t;

// ELLIPTIC_C
OPENQCD_MODULE_FUNCTION_ALIAS(ellipticK, ratfcts)
OPENQCD_MODULE_FUNCTION_ALIAS(sncndn, ratfcts)

// RATFCTS_C
OPENQCD_MODULE_FUNCTION_ALIAS(ratfct, ratfcts)

// ZOLOTAREV_C
OPENQCD_MODULE_FUNCTION_ALIAS(zolotarev, ratfcts)

} // namespace ratfct
} // namespace openqcd

#endif // CPP_RATFCTS_HPP
