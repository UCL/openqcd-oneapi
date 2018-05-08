
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

extern "C" {
#include "c_headers/block.h"
}

namespace openqcd {
namespace block {

using bndry_t = openqcd_block__bndry_t;
using block_t = openqcd_block__block_t;
using blk_grid_t = openqcd_block__blk_grid_t;

// BLOCK_C
const auto &alloc_blk = openqcd_block__alloc_blk;
const auto &alloc_bnd = openqcd_block__alloc_bnd;
const auto &clone_blk = openqcd_block__clone_blk;
const auto &free_blk = openqcd_block__free_blk;
const auto &ipt_blk = openqcd_block__ipt_blk;

// BLK_GRID_C
const auto &alloc_bgr = openqcd_block__alloc_bgr;
const auto &blk_list = openqcd_block__blk_list;

// MAP_U2BLK_C
const auto &assign_ud2ubgr = openqcd_block__assign_ud2ubgr;
const auto &assign_ud2udblk = openqcd_block__assign_ud2udblk;

// MAP_SW2BLK_C
const auto &assign_swd2swbgr = openqcd_block__assign_swd2swbgr;
const auto &assign_swd2swdblk = openqcd_block__assign_swd2swdblk;

// MAP_S2BLK_C
const auto &assign_s2sblk = openqcd_block__assign_s2sblk;
const auto &assign_sblk2s = openqcd_block__assign_sblk2s;
const auto &assign_s2sdblk = openqcd_block__assign_s2sdblk;
const auto &assign_sd2sdblk = openqcd_block__assign_sd2sdblk;
const auto &assign_sdblk2sd = openqcd_block__assign_sdblk2sd;

} // namespace block
} // namespace openqcd

#endif // ifndef CPP_BLOCK_HPP
