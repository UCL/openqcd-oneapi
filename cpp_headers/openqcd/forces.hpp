
/*******************************************************************************
 *
 * File forces.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_FORCES_HPP
#define CPP_FORCES_HPP

extern "C" {
#include "c_headers/forces.h"
}

namespace openqcd {
namespace forces {

// FORCE0_C
const auto &plaq_frc = openqcd_forces__plaq_frc;
const auto &force0 = openqcd_forces__force0;
const auto &action0 = openqcd_forces__action0;

// FORCE1_C
const auto &setpf1 = openqcd_forces__setpf1;
const auto &force1 = openqcd_forces__force1;
const auto &action1 = openqcd_forces__action1;

// FORCE2_C
const auto &setpf2 = openqcd_forces__setpf2;
const auto &force2 = openqcd_forces__force2;
const auto &action2 = openqcd_forces__action2;

// FORCE3_C
const auto &setpf3 = openqcd_forces__setpf3;
const auto &force3 = openqcd_forces__force3;
const auto &action3 = openqcd_forces__action3;

// FORCE4_C
const auto &setpf4 = openqcd_forces__setpf4;
const auto &force4 = openqcd_forces__force4;
const auto &action4 = openqcd_forces__action4;

// FORCE5_C
const auto &setpf5 = openqcd_forces__setpf5;
const auto &force5 = openqcd_forces__force5;
const auto &action5 = openqcd_forces__action5;

// FRCFCTS_C
const auto &det2xt = openqcd_forces__det2xt;
const auto &prod2xt = openqcd_forces__prod2xt;
const auto &prod2xv = openqcd_forces__prod2xv;

// GENFRC_C
const auto &sw_frc = openqcd_forces__sw_frc;
const auto &hop_frc = openqcd_forces__hop_frc;

// TMCG_C
const auto &tmcg = openqcd_forces__tmcg;
const auto &tmcgeo = openqcd_forces__tmcgeo;

// TMCGM_C
const auto &tmcgm = openqcd_forces__tmcgm;

// XTENSOR_C
const auto &xtensor = openqcd_forces__xtensor;
const auto &set_xt2zero = openqcd_forces__set_xt2zero;
const auto &add_det2xt = openqcd_forces__add_det2xt;
const auto &add_prod2xt = openqcd_forces__add_prod2xt;
const auto &xvector = openqcd_forces__xvector;
const auto &set_xv2zero = openqcd_forces__set_xv2zero;
const auto &add_prod2xv = openqcd_forces__add_prod2xv;

} // namespace forces
} // namespace openqcd

#endif
