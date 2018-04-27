
/*******************************************************************************
 *
 * File version.h
 *
 * Copyright (C) 2009 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef VERSION_H
#define VERSION_H

#define openQCD_RELEASE "openQCD-FASTSUM v1.0 (based on openQCD-1.6)"

extern const char *openqcd__build_date;
extern const char *openqcd__build_git_sha;
extern const char *openqcd__build_user_cflags;

#if defined(OPENQCD_INTERNAL)
#define build_date openqcd__build_date
#define build_git_sha openqcd__build_git_sha
#define build_user_cflags openqcd__build_user_cflags
#endif /* defined OPENQCD_INTERNAL */

#endif
