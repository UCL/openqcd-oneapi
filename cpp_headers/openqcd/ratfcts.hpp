
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

extern "C" {
#include "c_headers/ratfcts.h"
}

namespace openqcd {
namespace ratfct {

using ratfct_t = openqcd_ratfcts__ratfct_t;

// ELLIPTIC_C
const auto &ellipticK = openqcd_ratfcts__ellipticK;
const auto &sncndn = openqcd_ratfcts__sncndn;

// RATFCTS_C
const auto &ratfct = openqcd_ratfcts__ratfct;

// ZOLOTAREV_C
const auto &zolotarev = openqcd_ratfcts__zolotarev;

} // namespace ratfct
} // namespace openqcd

#endif // CPP_RATFCTS_HPP
