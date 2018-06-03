
/*******************************************************************************
 *
 * File uflds.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_UFLDS_HPP
#define CPP_UFLDS_HPP

extern "C" {
#include "c_headers/uflds.h"
}

namespace openqcd {
namespace uflds {

// BSTAP_C
const auto &bstap = openqcd_uflds__bstap;
const auto &set_bstap = openqcd_uflds__set_bstap;

// PLAQ_SUM_C
const auto &plaq_sum_split_dble = openqcd_uflds__plaq_sum_split_dble;
const auto &plaq_sum_dble = openqcd_uflds__plaq_sum_dble;
const auto &plaq_wsum_split_dble = openqcd_uflds__plaq_wsum_split_dble;
const auto &plaq_wsum_dble = openqcd_uflds__plaq_wsum_dble;
const auto &plaq_action_slices = openqcd_uflds__plaq_action_slices;
const auto &spatial_link_sum = openqcd_uflds__spatial_link_sum;
const auto &temporal_link_sum = openqcd_uflds__temporal_link_sum;

// POLYAKOV_LOOP_C
const auto &polyakov_loop = openqcd_uflds__polyakov_loop;

// SHIFT_C
const auto &shift_ud = openqcd_uflds__shift_ud;

// UFLDS_C
const auto &ufld = openqcd_uflds__ufld;
const auto &udfld = openqcd_uflds__udfld;
const auto &apply_ani_ud = openqcd_uflds__apply_ani_ud;
const auto &remove_ani_ud = openqcd_uflds__remove_ani_ud;
const auto &random_ud = openqcd_uflds__random_ud;
const auto &set_ud_phase = openqcd_uflds__set_ud_phase;
const auto &unset_ud_phase = openqcd_uflds__unset_ud_phase;
const auto &renormalize_ud = openqcd_uflds__renormalize_ud;
const auto &assign_ud2u = openqcd_uflds__assign_ud2u;
const auto &swap_udfld = openqcd_uflds__swap_udfld;
const auto &copy_bnd_ud = openqcd_uflds__copy_bnd_ud;

} // namespace uflds
} // namespace openqcd

#endif // CPP_UFLDS_HPP
