
/*******************************************************************************
 *
 * File archive.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_ARCHIVE_HPP
#define CPP_ARCHIVE_HPP

extern "C" {
#include "c_headers/archive.h"
}

namespace openqcd {
namespace archive {

// ARCHIVE_H
const auto& write_cnfg = openqcd_archive__write_cnfg;
const auto& read_cnfg = openqcd_archive__read_cnfg;
const auto& export_cnfg = openqcd_archive__export_cnfg;
const auto& import_cnfg = openqcd_archive__import_cnfg;

// SARCHIVE_H
const auto& write_sfld = openqcd_archive__write_sfld;
const auto& read_sfld = openqcd_archive__read_sfld;
const auto& export_sfld = openqcd_archive__export_sfld;
const auto& import_sfld = openqcd_archive__import_sfld;

} // namespace archive 
} // namespace openqcd 

#endif // define CPP_ARCHIVE_HPP
