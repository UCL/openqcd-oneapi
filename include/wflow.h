/*******************************************************************************
 *
 * File wflow.h
 *
 * Copyright (C) 2009, 2010, 2011, 2012 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef WFLOW_H
#define WFLOW_H

/* WFLOW_C */
extern void openqcd_wflow__fwd_euler(int n, double eps);
extern void openqcd_wflow__fwd_rk2(int n, double eps);
extern void openqcd_wflow__fwd_rk3(int n, double eps);

#if defined(OPENQCD_INTERNAL)
#define fwd_euler(...) openqcd_wflow__fwd_euler(__VA_ARGS__)
#define fwd_rk2(...) openqcd_wflow__fwd_rk2(__VA_ARGS__)
#define fwd_rk3(...) openqcd_wflow__fwd_rk3(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif
