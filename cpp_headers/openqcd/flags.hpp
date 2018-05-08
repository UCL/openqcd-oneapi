
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
const auto &set_flags = openqcd_flags__set_flags;
const auto &set_grid_flags = openqcd_flags__set_grid_flags;
const auto &query_flags = openqcd_flags__query_flags;
const auto &query_grid_flags = openqcd_flags__query_grid_flags;
const auto &print_flags = openqcd_flags__print_flags;
const auto &print_grid_flags = openqcd_flags__print_grid_flags;

// ACTION_PARMS_C
const auto &set_action_parms = openqcd_flags__set_action_parms;
const auto &action_parms = openqcd_flags__action_parms;
const auto &read_action_parms = openqcd_flags__read_action_parms;
const auto &print_action_parms = openqcd_flags__print_action_parms;
const auto &write_action_parms = openqcd_flags__write_action_parms;
const auto &check_action_parms = openqcd_flags__check_action_parms;

// DFL_PARMS_C
const auto &set_dfl_parms = openqcd_flags__set_dfl_parms;
const auto &dfl_parms = openqcd_flags__dfl_parms;
const auto &set_dfl_pro_parms = openqcd_flags__set_dfl_pro_parms;
const auto &dfl_pro_parms = openqcd_flags__dfl_pro_parms;
const auto &set_dfl_gen_parms = openqcd_flags__set_dfl_gen_parms;
const auto &dfl_gen_parms = openqcd_flags__dfl_gen_parms;
const auto &set_dfl_upd_parms = openqcd_flags__set_dfl_upd_parms;
const auto &dfl_upd_parms = openqcd_flags__dfl_upd_parms;
const auto &print_dfl_parms = openqcd_flags__print_dfl_parms;
const auto &write_dfl_parms = openqcd_flags__write_dfl_parms;
const auto &check_dfl_parms = openqcd_flags__check_dfl_parms;

// FORCE_PARMS_C
const auto &set_force_parms = openqcd_flags__set_force_parms;
const auto &force_parms = openqcd_flags__force_parms;
const auto &read_force_parms = openqcd_flags__read_force_parms;
const auto &read_force_parms2 = openqcd_flags__read_force_parms2;
const auto &print_force_parms = openqcd_flags__print_force_parms;
const auto &print_force_parms2 = openqcd_flags__print_force_parms2;
const auto &write_force_parms = openqcd_flags__write_force_parms;
const auto &check_force_parms = openqcd_flags__check_force_parms;

// HMC_PARMS_C
const auto &set_hmc_parms = openqcd_flags__set_hmc_parms;
const auto &hmc_parms = openqcd_flags__hmc_parms;
const auto &print_hmc_parms = openqcd_flags__print_hmc_parms;
const auto &write_hmc_parms = openqcd_flags__write_hmc_parms;
const auto &check_hmc_parms = openqcd_flags__check_hmc_parms;

// LAT_PARMS_C
const auto &set_lat_parms = openqcd_flags__set_lat_parms;
const auto &lat_parms = openqcd_flags__lat_parms;
const auto &print_lat_parms = openqcd_flags__print_lat_parms;
const auto &write_lat_parms = openqcd_flags__write_lat_parms;
const auto &check_lat_parms = openqcd_flags__check_lat_parms;

const auto &set_bc_parms = openqcd_flags__set_bc_parms;
const auto &bc_parms = openqcd_flags__bc_parms;
const auto &print_bc_parms = openqcd_flags__print_bc_parms;
const auto &write_bc_parms = openqcd_flags__write_bc_parms;
const auto &check_bc_parms = openqcd_flags__check_bc_parms;

const auto &sea_quark_mass = openqcd_flags__sea_quark_mass;
const auto &bc_type = openqcd_flags__bc_type;

const auto &set_sw_parms = openqcd_flags__set_sw_parms;
const auto &sw_parms = openqcd_flags__sw_parms;
const auto &set_tm_parms = openqcd_flags__set_tm_parms;
const auto &tm_parms = openqcd_flags__tm_parms;

// MDINT_PARMS_C
const auto &set_mdint_parms = openqcd_flags__set_mdint_parms;
const auto &mdint_parms = openqcd_flags__mdint_parms;
const auto &read_mdint_parms = openqcd_flags__read_mdint_parms;
const auto &print_mdint_parms = openqcd_flags__print_mdint_parms;
const auto &write_mdint_parms = openqcd_flags__write_mdint_parms;
const auto &check_mdint_parms = openqcd_flags__check_mdint_parms;

// RAT_PARMS_C
const auto &set_rat_parms = openqcd_flags__set_rat_parms;
const auto &rat_parms = openqcd_flags__rat_parms;
const auto &read_rat_parms = openqcd_flags__read_rat_parms;
const auto &print_rat_parms = openqcd_flags__print_rat_parms;
const auto &write_rat_parms = openqcd_flags__write_rat_parms;
const auto &check_rat_parms = openqcd_flags__check_rat_parms;

// RW_PARMS_C
const auto &set_rw_parms = openqcd_flags__set_rw_parms;

const auto &rw_parms = openqcd_flags__rw_parms;
const auto &read_rw_parms = openqcd_flags__read_rw_parms;
const auto &print_rw_parms = openqcd_flags__print_rw_parms;
const auto &write_rw_parms = openqcd_flags__write_rw_parms;
const auto &check_rw_parms = openqcd_flags__check_rw_parms;

// SAP_PARMS_C
const auto &set_sap_parms = openqcd_flags__set_sap_parms;
const auto &sap_parms = openqcd_flags__sap_parms;
const auto &print_sap_parms = openqcd_flags__print_sap_parms;
const auto &write_sap_parms = openqcd_flags__write_sap_parms;
const auto &check_sap_parms = openqcd_flags__check_sap_parms;

// SOLVER_PARMS_C
const auto &set_solver_parms = openqcd_flags__set_solver_parms;
const auto &solver_parms = openqcd_flags__solver_parms;
const auto &read_solver_parms = openqcd_flags__read_solver_parms;
const auto &print_solver_parms = openqcd_flags__print_solver_parms;
const auto &write_solver_parms = openqcd_flags__write_solver_parms;
const auto &check_solver_parms = openqcd_flags__check_solver_parms;

// ANISOTROPY_PARMS_C
const auto &set_ani_parms = openqcd_flags__set_ani_parms;
const auto &set_no_ani_parms = openqcd_flags__set_no_ani_parms;
const auto &ani_parms = openqcd_flags__ani_parms;
const auto &print_ani_parms = openqcd_flags__print_ani_parms;
const auto &ani_params_initialised = openqcd_flags__ani_params_initialised;
const auto &write_ani_parms = openqcd_flags__write_ani_parms;
const auto &check_ani_parms = openqcd_flags__check_ani_parms;

// SMEARING_PARMS_C
const auto &set_stout_smearing_parms = openqcd_flags__set_stout_smearing_parms;
const auto &set_no_stout_smearing_parms =
    openqcd_flags__set_no_stout_smearing_parms;
const auto &reset_stout_smearing = openqcd_flags__reset_stout_smearing;
const auto &stout_smearing_parms = openqcd_flags__stout_smearing_parms;
const auto &print_stout_smearing_parms =
    openqcd_flags__print_stout_smearing_parms;
const auto &write_stout_smearing_parms =
    openqcd_flags__write_stout_smearing_parms;
const auto &check_stout_smearing_parms =
    openqcd_flags__check_stout_smearing_parms;

} // namespace flags
} // namespace openqcd

#endif
