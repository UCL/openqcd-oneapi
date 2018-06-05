
/*******************************************************************************
 *
 * File block.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_BLOCK_HPP
#define CPP_BLOCK_HPP

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/block.h"
}

namespace openqcd {
namespace block {

using bndry_t = openqcd_block__bndry_t;
using block_t = openqcd_block__block_t;
using blk_grid_t = openqcd_block__blk_grid_t;

// BLOCK_C
OPENQCD_MODULE_FUNCTION_ALIAS(alloc_blk, block)
OPENQCD_MODULE_FUNCTION_ALIAS(alloc_bnd, block)
OPENQCD_MODULE_FUNCTION_ALIAS(clone_blk, block)
OPENQCD_MODULE_FUNCTION_ALIAS(free_blk, block)
OPENQCD_MODULE_FUNCTION_ALIAS(ipt_blk, block)

// BLK_GRID_C
OPENQCD_MODULE_FUNCTION_ALIAS(alloc_bgr, block)
OPENQCD_MODULE_FUNCTION_ALIAS(blk_list, block)

// MAP_U2BLK_C
OPENQCD_MODULE_FUNCTION_ALIAS(assign_ud2ubgr, block)
OPENQCD_MODULE_FUNCTION_ALIAS(assign_ud2udblk, block)

// MAP_SW2BLK_C
OPENQCD_MODULE_FUNCTION_ALIAS(assign_swd2swbgr, block)
OPENQCD_MODULE_FUNCTION_ALIAS(assign_swd2swdblk, block)

// MAP_S2BLK_C
OPENQCD_MODULE_FUNCTION_ALIAS(assign_s2sblk, block)
OPENQCD_MODULE_FUNCTION_ALIAS(assign_sblk2s, block)
OPENQCD_MODULE_FUNCTION_ALIAS(assign_s2sdblk, block)
OPENQCD_MODULE_FUNCTION_ALIAS(assign_sd2sdblk, block)
OPENQCD_MODULE_FUNCTION_ALIAS(assign_sdblk2sd, block)

} // namespace block
} // namespace openqcd

#endif // ifndef CPP_BLOCK_HPP
