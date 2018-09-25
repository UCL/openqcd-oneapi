# Stout smearing tests

These programs tests that the stout smearing procedures are correctly
implemented. There are currently 8 tests available:

 * `test01_omega_matrices`: computes the omega matrices from a known gauge
   configuration
 * `test0[2-4]_smear_nn`: reads a reference config, smears it, and compares with
   a second smeared reference config
 * `test05_unsmearing_intrinsics`: checks the intermediate computations for the
     unsmearing against known results
 * `test06_unsmearing_1`: carries out an unsmearing and checks that the expected
     links change
 * `test07_smeared_field_cycling`: checks that the smearing and unsmearing
     routines correctly keeps track of the locations of all the gauge fields
 * `test08_xi_matrix_test`: computes the xi matrix for a known setup and looks
     for discrepancies

Tests 2 to 4 requires reference configurations stored in a `configuration`
subfolder named in the following way:

 * `orig_config.conf`: the unsmeared configuration
 * `smeared_conf_n1.conf`: original config smeared once
 * `smeared_conf_n2.conf`: original config smeared twice
 * `smeared_conf_n3.conf`: original config smeared thrice

One needs to check that the config files have the same geometry as the one
specified in [`global.h`](include/global.h). The tests assume that the
configurations are smeared with `rho_t = 0.00` and `rho_s = 0.25`. This can
easily be changed by modifying the test source files.

A set of configs generated and smeared by chroma is available with through git
LFS (Large File Storage). One can access these by simply calling:

```bash
git lfs fetch --all
```

provided that [git-lfs](https://git-lfs.github.com/) is installed.
