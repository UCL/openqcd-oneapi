
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

#include "internal/function_alias.hpp"

extern "C" {
#include "c_headers/archive.h"
}

namespace openqcd {
namespace archive {

// ARCHIVE_H
OPENQCD_MODULE_FUNCTION_ALIAS(write_cnfg, archive)
OPENQCD_MODULE_FUNCTION_ALIAS(read_cnfg, archive)
OPENQCD_MODULE_FUNCTION_ALIAS(export_cnfg, archive)
OPENQCD_MODULE_FUNCTION_ALIAS(import_cnfg, archive)

// SARCHIVE_H
OPENQCD_MODULE_FUNCTION_ALIAS(write_sfld, archive)
OPENQCD_MODULE_FUNCTION_ALIAS(read_sfld, archive)
OPENQCD_MODULE_FUNCTION_ALIAS(export_sfld, archive)
OPENQCD_MODULE_FUNCTION_ALIAS(import_sfld, archive)

} // namespace archive
} // namespace openqcd

#endif // define CPP_ARCHIVE_HPP
