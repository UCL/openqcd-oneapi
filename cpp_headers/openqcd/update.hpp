
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

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/update.h"
}

namespace openqcd {
namespace update {

using mdstep_t = openqcd_update__mdstep_t;

/* CHRONO */
OPENQCD_MODULE_FUNCTION_ALIAS(setup_chrono, update)
OPENQCD_MODULE_FUNCTION_ALIAS(mdtime, update)
OPENQCD_MODULE_FUNCTION_ALIAS(step_mdtime, update)
OPENQCD_MODULE_FUNCTION_ALIAS(add_chrono, update)
OPENQCD_MODULE_FUNCTION_ALIAS(get_chrono, update)
OPENQCD_MODULE_FUNCTION_ALIAS(reset_chrono, update)

/* COUNTERS */
OPENQCD_MODULE_FUNCTION_ALIAS(setup_counters, update)
OPENQCD_MODULE_FUNCTION_ALIAS(clear_counters, update)
OPENQCD_MODULE_FUNCTION_ALIAS(add2counter, update)
OPENQCD_MODULE_FUNCTION_ALIAS(get_count, update)
OPENQCD_MODULE_FUNCTION_ALIAS(print_avgstat, update)
OPENQCD_MODULE_FUNCTION_ALIAS(print_all_avgstat, update)

// MDSTEPS_C
OPENQCD_MODULE_FUNCTION_ALIAS(set_mdsteps, update)
OPENQCD_MODULE_FUNCTION_ALIAS(mdsteps, update)
OPENQCD_MODULE_FUNCTION_ALIAS(print_mdsteps, update)

// MDINT_C
OPENQCD_MODULE_FUNCTION_ALIAS(run_mdint, update)

// HMC_C
OPENQCD_MODULE_FUNCTION_ALIAS(hmc_sanity_check, update)
OPENQCD_MODULE_FUNCTION_ALIAS(hmc_wsize, update)
OPENQCD_MODULE_FUNCTION_ALIAS(run_hmc, update)

// RWRAT_C
OPENQCD_MODULE_FUNCTION_ALIAS(rwrat, update)

// RWTM_C
OPENQCD_MODULE_FUNCTION_ALIAS(rwtm1, update)
OPENQCD_MODULE_FUNCTION_ALIAS(rwtm2, update)

// RWTMEO_C
OPENQCD_MODULE_FUNCTION_ALIAS(rwtm1eo, update)
OPENQCD_MODULE_FUNCTION_ALIAS(rwtm2eo, update)

} // namespace update
} // namespace openqcd

#endif // CPP_UPDATE_HPP
