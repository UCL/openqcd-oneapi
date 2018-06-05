
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

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/uflds.h"
}

namespace openqcd {
namespace uflds {

// BSTAP_C
OPENQCD_MODULE_FUNCTION_ALIAS(bstap, uflds)
OPENQCD_MODULE_FUNCTION_ALIAS(set_bstap, uflds)

// PLAQ_SUM_C
OPENQCD_MODULE_FUNCTION_ALIAS(plaq_sum_split_dble, uflds)
OPENQCD_MODULE_FUNCTION_ALIAS(plaq_sum_dble, uflds)
OPENQCD_MODULE_FUNCTION_ALIAS(plaq_wsum_split_dble, uflds)
OPENQCD_MODULE_FUNCTION_ALIAS(plaq_wsum_dble, uflds)
OPENQCD_MODULE_FUNCTION_ALIAS(plaq_action_slices, uflds)
OPENQCD_MODULE_FUNCTION_ALIAS(spatial_link_sum, uflds)
OPENQCD_MODULE_FUNCTION_ALIAS(temporal_link_sum, uflds)

// POLYAKOV_LOOP_C
OPENQCD_MODULE_FUNCTION_ALIAS(polyakov_loop, uflds)

// SHIFT_C
OPENQCD_MODULE_FUNCTION_ALIAS(shift_ud, uflds)

// UFLDS_C
OPENQCD_MODULE_FUNCTION_ALIAS(ufld, uflds)
OPENQCD_MODULE_FUNCTION_ALIAS(udfld, uflds)
OPENQCD_MODULE_FUNCTION_ALIAS(apply_ani_ud, uflds)
OPENQCD_MODULE_FUNCTION_ALIAS(remove_ani_ud, uflds)
OPENQCD_MODULE_FUNCTION_ALIAS(random_ud, uflds)
OPENQCD_MODULE_FUNCTION_ALIAS(set_ud_phase, uflds)
OPENQCD_MODULE_FUNCTION_ALIAS(unset_ud_phase, uflds)
OPENQCD_MODULE_FUNCTION_ALIAS(renormalize_ud, uflds)
OPENQCD_MODULE_FUNCTION_ALIAS(assign_ud2u, uflds)
OPENQCD_MODULE_FUNCTION_ALIAS(swap_udfld, uflds)
OPENQCD_MODULE_FUNCTION_ALIAS(copy_bnd_ud, uflds)

} // namespace uflds
} // namespace openqcd

#endif // CPP_UFLDS_HPP
