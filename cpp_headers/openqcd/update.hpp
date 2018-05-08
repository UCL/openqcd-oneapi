
/*******************************************************************************
 *
 * File update.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_UPDATE_HPP
#define CPP_UPDATE_HPP

extern "C" {
#include "c_headers/update.h"
}

namespace openqcd {
namespace update {

using mdstep_t = openqcd_update__mdstep_t;

/* CHRONO */
const auto &setup_chrono = openqcd_update__setup_chrono;
const auto &mdtime = openqcd_update__mdtime;
const auto &step_mdtime = openqcd_update__step_mdtime;
const auto &add_chrono = openqcd_update__add_chrono;
const auto &get_chrono = openqcd_update__get_chrono;
const auto &reset_chrono = openqcd_update__reset_chrono;

/* COUNTERS */
const auto &setup_counters = openqcd_update__setup_counters;
const auto &clear_counters = openqcd_update__clear_counters;
const auto &add2counter = openqcd_update__add2counter;
const auto &get_count = openqcd_update__get_count;
const auto &print_avgstat = openqcd_update__print_avgstat;
const auto &print_all_avgstat = openqcd_update__print_all_avgstat;

// MDSTEPS_C
const auto &set_mdsteps = openqcd_update__set_mdsteps;
const auto &mdsteps = openqcd_update__mdsteps;
const auto &print_mdsteps = openqcd_update__print_mdsteps;

// MDINT_C
const auto &run_mdint = openqcd_update__run_mdint;

// HMC_C
const auto &hmc_sanity_check = openqcd_update__hmc_sanity_check;
const auto &hmc_wsize = openqcd_update__hmc_wsize;
const auto &run_hmc = openqcd_update__run_hmc;

// RWRAT_C
const auto &rwrat = openqcd_update__rwrat;

// RWTM_C
const auto &rwtm1 = openqcd_update__rwtm1;
const auto &rwtm2 = openqcd_update__rwtm2;

// RWTMEO_C
const auto &rwtm1eo = openqcd_update__rwtm1eo;
const auto &rwtm2eo = openqcd_update__rwtm2eo;

} // namespace update
} // namespace openqcd

#endif // CPP_UPDATE_HPP
