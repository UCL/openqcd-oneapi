
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
extern void write_cnfg(char const *out);
extern void read_cnfg(char const *in);
extern void export_cnfg(char const *out);
extern void import_cnfg(char const *in);

/* SARCHIVE_C */
extern void write_sfld(char const *out, spinor_dble const *sd);
extern void read_sfld(char const *in, spinor_dble *sd);
extern void export_sfld(char const *out, spinor_dble const *sd);
extern void import_sfld(char const *in, spinor_dble *sd);

#endif
