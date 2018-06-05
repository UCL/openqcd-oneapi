
/*******************************************************************************
 *
 * File flags.h
 *
 * Copyright (C) 2009-2014, 2016 Martin Luescher, Isabel Campos
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_FLAGS_HPP
#define CPP_FLAGS_HPP

extern "C" {
#include "c_headers/flags.h"
}

namespace openqcd {
namespace flags {

using event_t = openqcd_flags__event_t;
using query_t = openqcd_flags__query_t;
using action_t = openqcd_flags__action_t;
using integrator_t = openqcd_flags__integrator_t;
using force_t = openqcd_flags__force_t;
using rwfact_t = openqcd_flags__rwfact_t;
using solver_t = openqcd_flags__solver_t;
using action_parms_t = openqcd_flags__action_parms_t;
using bc_parms_t = openqcd_flags__bc_parms_t;
using dfl_parms_t = openqcd_flags__dfl_parms_t;
using dfl_pro_parms_t = openqcd_flags__dfl_pro_parms_t;
using dfl_gen_parms_t = openqcd_flags__dfl_gen_parms_t;
using dfl_upd_parms_t = openqcd_flags__dfl_upd_parms_t;
using force_parms_t = openqcd_flags__force_parms_t;
using hmc_parms_t = openqcd_flags__hmc_parms_t;
using lat_parms_t = openqcd_flags__lat_parms_t;
using mdint_parms_t = openqcd_flags__mdint_parms_t;
using rat_parms_t = openqcd_flags__rat_parms_t;
using rw_parms_t = openqcd_flags__rw_parms_t;
using sw_parms_t = openqcd_flags__sw_parms_t;
using sap_parms_t = openqcd_flags__sap_parms_t;
using solver_parms_t = openqcd_flags__solver_parms_t;
using tm_parms_t = openqcd_flags__tm_parms_t;
using ani_params_t = openqcd_flags__ani_params_t;
using stout_smearing_params_t = openqcd_flags__stout_smearing_params_t;

// FLAGS_C
OPENQCD_MODULE_FUNCTION_ALIAS(set_flags, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(set_grid_flags, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(query_flags, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(query_grid_flags, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(print_flags, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(print_grid_flags, flags)

// ACTION_PARMS_C
OPENQCD_MODULE_FUNCTION_ALIAS(set_action_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(action_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(read_action_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(print_action_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(write_action_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(check_action_parms, flags)

// DFL_PARMS_C
OPENQCD_MODULE_FUNCTION_ALIAS(set_dfl_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(dfl_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(set_dfl_pro_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(dfl_pro_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(set_dfl_gen_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(dfl_gen_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(set_dfl_upd_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(dfl_upd_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(print_dfl_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(write_dfl_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(check_dfl_parms, flags)

// FORCE_PARMS_C
OPENQCD_MODULE_FUNCTION_ALIAS(set_force_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(force_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(read_force_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(read_force_parms2, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(print_force_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(print_force_parms2, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(write_force_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(check_force_parms, flags)

// HMC_PARMS_C
OPENQCD_MODULE_FUNCTION_ALIAS(set_hmc_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(hmc_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(print_hmc_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(write_hmc_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(check_hmc_parms, flags)

// LAT_PARMS_C
OPENQCD_MODULE_FUNCTION_ALIAS(set_lat_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(lat_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(print_lat_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(write_lat_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(check_lat_parms, flags)

OPENQCD_MODULE_FUNCTION_ALIAS(set_bc_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(bc_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(print_bc_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(write_bc_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(check_bc_parms, flags)

OPENQCD_MODULE_FUNCTION_ALIAS(sea_quark_mass, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(bc_type, flags)

OPENQCD_MODULE_FUNCTION_ALIAS(set_sw_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(sw_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(set_tm_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(tm_parms, flags)

// MDINT_PARMS_C
OPENQCD_MODULE_FUNCTION_ALIAS(set_mdint_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(mdint_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(read_mdint_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(print_mdint_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(write_mdint_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(check_mdint_parms, flags)

// RAT_PARMS_C
OPENQCD_MODULE_FUNCTION_ALIAS(set_rat_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(rat_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(read_rat_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(print_rat_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(write_rat_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(check_rat_parms, flags)

// RW_PARMS_C
OPENQCD_MODULE_FUNCTION_ALIAS(set_rw_parms, flags)

OPENQCD_MODULE_FUNCTION_ALIAS(rw_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(read_rw_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(print_rw_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(write_rw_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(check_rw_parms, flags)

// SAP_PARMS_C
OPENQCD_MODULE_FUNCTION_ALIAS(set_sap_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(sap_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(print_sap_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(write_sap_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(check_sap_parms, flags)

// SOLVER_PARMS_C
OPENQCD_MODULE_FUNCTION_ALIAS(set_solver_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(solver_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(read_solver_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(print_solver_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(write_solver_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(check_solver_parms, flags)

// ANISOTROPY_PARMS_C
OPENQCD_MODULE_FUNCTION_ALIAS(set_ani_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(set_no_ani_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(ani_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(print_ani_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(ani_params_initialised, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(write_ani_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(check_ani_parms, flags)

// SMEARING_PARMS_C
OPENQCD_MODULE_FUNCTION_ALIAS(set_stout_smearing_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(set_no_stout_smearing_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(reset_stout_smearing, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(stout_smearing_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(print_stout_smearing_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(write_stout_smearing_parms, flags)
OPENQCD_MODULE_FUNCTION_ALIAS(check_stout_smearing_parms, flags)

} // namespace flags
} // namespace openqcd

#endif
