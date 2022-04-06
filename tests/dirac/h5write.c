
/*******************************************************************************
 *
 * File time1.c
 *
 * Copyright (C) 2022 Tuomas Koskela, UCL ARC
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Write spinor data into a hdf5 file
 *
 *******************************************************************************/


#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>

#include "su3.h"


void write_spinor(char* filename, spinor** ps)
{

  // Open new HDF5 file, overwrite existing, default file creation and file access property lists
  file = H5Fcreate (filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  char* dsetname = "complex";

  hid_t file, filetype, memtype, strtype, space, dset;
  int status;
  complex num;
  num.re = 5.3;
  num.im = 1.0;

  // Create compound datatype for memory
  memtype = H5Tcreate (H5T_COMPOUND, sizeof (complex));

  /*
   * Add fields to compound datatype. The C library provides a macro for calculating the offset
   * http://davis.lbl.gov/Manuals/HDF5-1.8.7/UG/11_Datatypes.html
   * HOFFSET(s,m)
   * This macro computes the offset of member m within a struct s
   * offsetof(s,m)
   * This macro defined in stddef.h does exactly the same thing as the HOFFSET() macro.
   */
  status = H5Tinsert (memtype, "re", HOFFSET (complex, re), H5T_NATIVE_FLOAT);
  status = H5Tinsert (memtype, "im", HOFFSET (complex, im), H5T_NATIVE_FLOAT);

  // Create compound datatype for the file.
  filetype = H5Tcreate(H5T_COMPOUND, 2 * sizeof(float));
  status = H5Tinsert (filetype, "re", 0, H5T_NATIVE_FLOAT);
  status = H5Tinsert (filetype, "im", sizeof(float), H5T_NATIVE_FLOAT);

  space = H5Screate_simple (1, dims, NULL);
  // Create dataset on file
  dset = H5Dcreate (file, dsetname, filetype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  // Write data to dataset
  status = H5Dwrite (dset, memtype, H5S_ALL, H5P_DEFAULT, num);
  
  h5Fclose (file);
  
}
