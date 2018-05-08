
/*******************************************************************************
 *
 * File wflow.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_WFLOW_HPP
#define CPP_WFLOW_HPP

extern "C" {
#include "c_headers/wflow.h"
}

namespace openqcd {
namespace wflow {

// WFLOW_C
const auto &fwd_euler = openqcd_wflow__fwd_euler;
const auto &fwd_rk2 = openqcd_wflow__fwd_rk2;
const auto &fwd_rk3 = openqcd_wflow__fwd_rk3;

} // namespace wflow
} // namespace openqcd

#endif
