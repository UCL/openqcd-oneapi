
/*******************************************************************************
 *
 * File archive.h
 *
 * Copyright (C) 2011 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef ARCHIVE_H
#define ARCHIVE_H

#include "su3.h"
#include <stdio.h>

/* ARCHIVE_C */
extern void openqcd_archive__write_cnfg(char const *out);
extern void openqcd_archive__read_cnfg(char const *in);
extern void openqcd_archive__export_cnfg(char const *out);
extern void openqcd_archive__import_cnfg(char const *in);

/* MOD_ARCHIVE_C */
extern void openqcd_archive__mod_export_cnfg(char const *out);

/* SARCHIVE_C */
extern void openqcd_archive__write_sfld(char const *out,
                                        openqcd__spinor_dble const *sd);
extern void openqcd_archive__read_sfld(char const *in,
                                       openqcd__spinor_dble *sd);
extern void openqcd_archive__export_sfld(char const *out,
                                         openqcd__spinor_dble const *sd);
extern void openqcd_archive__import_sfld(char const *in,
                                         openqcd__spinor_dble *sd);

#if defined(OPENQCD_INTERNAL)
#define write_cnfg(...) openqcd_archive__write_cnfg(__VA_ARGS__)
#define read_cnfg(...) openqcd_archive__read_cnfg(__VA_ARGS__)
#define export_cnfg(...) openqcd_archive__export_cnfg(__VA_ARGS__)
#define import_cnfg(...) openqcd_archive__import_cnfg(__VA_ARGS__)

#define mod_export_cnfg(...) openqcd_archive__mod_export_cnfg(__VA_ARGS__)

#define write_sfld(...) openqcd_archive__write_sfld(__VA_ARGS__)
#define read_sfld(...) openqcd_archive__read_sfld(__VA_ARGS__)
#define export_sfld(...) openqcd_archive__export_sfld(__VA_ARGS__)
#define import_sfld(...) openqcd_archive__import_sfld(__VA_ARGS__)
#endif

#endif
