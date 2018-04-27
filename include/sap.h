
/*******************************************************************************
 *
 * File sap.h
 *
 * Copyright (C) 2011 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef SAP_H
#define SAP_H

#include "su3.h"

/* BLK_SOLV_C */
extern void openqcd_sap__blk_mres(int n, float mu, int nmr);
extern void openqcd_sap__blk_eo_mres(int n, float mu, int nmr);

/* SAP_COM_C */
#if ((defined SAP_COM_C) || (defined BLK_GRID_C))
extern void alloc_sap_bufs(void);
#endif
extern void openqcd_sap__sap_com(int ic, openqcd__spinor *r);

/* SAP */
extern void openqcd_sap__sap(float mu, int isolv, int nmr, openqcd__spinor *psi,
                             openqcd__spinor *eta);

/* SAP_GCR */
extern double openqcd_sap__sap_gcr(int nkv, int nmx, double res, double mu,
                                   openqcd__spinor_dble *eta,
                                   openqcd__spinor_dble *psi, int *status);

#if defined(OPENQCD_INTERNAL)
/* BLK_SOLV_C */
#define blk_mres(...) openqcd_sap__blk_mres(__VA_ARGS__)
#define blk_eo_mres(...) openqcd_sap__blk_eo_mres(__VA_ARGS__)

/* SAP_COM_C */
#define sap_com(...) openqcd_sap__sap_com(__VA_ARGS__)

/* SAP */
#define sap(...) openqcd_sap__sap(__VA_ARGS__)

/* SAP_GCR */
#define sap_gcr(...) openqcd_sap__sap_gcr(__VA_ARGS__)
#endif /* defined OPENQCD_INTERNAL */

#endif
