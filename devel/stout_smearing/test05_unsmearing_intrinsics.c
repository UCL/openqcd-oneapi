
/*
 * Created: 14-07-2017
 * Modified:
 * Author: Jonas R. Glesaaen (jonas@glesaaen.com)
 */

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "su3.h"
#include "lattice.h"
#include "global.h"
#include "stout_smearing.h"

#include <devel/testing_utilities/data_type_diffs.c>
#include <devel/testing_utilities/diff_printing.c>
#include <devel/testing_utilities/test_counter.c>
#include <modules/stout_smearing/force_unsmearing.c>

void get_inputs(su3_dble *exp_mat, su3_dble *link, su3_alg_dble *force)
{
  (*exp_mat).c11.re = -0.25732715021457997;
  (*exp_mat).c11.im = +0.2156848637799973;
  (*exp_mat).c12.re = -0.48500084047815184;
  (*exp_mat).c12.im = +0.7363152620501596;
  (*exp_mat).c13.re = 0.7812119236879149;
  (*exp_mat).c13.im = -0.5862079641571598;
  (*exp_mat).c21.re = 0.5991385855970677;
  (*exp_mat).c21.im = +0.18824442033354538;
  (*exp_mat).c22.re = 0.6771528973038392;
  (*exp_mat).c22.im = +0.18064473593968522;
  (*exp_mat).c23.re = -0.8430967671888907;
  (*exp_mat).c23.im = -0.8040413582189334;
  (*exp_mat).c31.re = -0.6320128744566076;
  (*exp_mat).c31.im = -0.8202327750659357;
  (*exp_mat).c32.re = 0.33007334587956816;
  (*exp_mat).c32.im = -0.5049787338511353;
  (*exp_mat).c33.re = 0.4489537207004859;
  (*exp_mat).c33.im = +0.17468942248307018;

  (*link).c11.re = -0.04523973000965238;
  (*link).c11.im = -0.408599969028007;
  (*link).c12.re = 0.5958346134785913;
  (*link).c12.im = +0.29070979628949095;
  (*link).c13.re = 0.048604694242178496;
  (*link).c13.im = -0.6237835710875447;
  (*link).c21.re = 0.7764677719341057;
  (*link).c21.im = +0.2654232745053738;
  (*link).c22.re = 0.23101794725586697;
  (*link).c22.im = +0.12954289281669631;
  (*link).c23.re = -0.5063273535774693;
  (*link).c23.im = +0.01141238292972568;
  (*link).c31.re = -0.39704033951018475;
  (*link).c31.im = +0.0025856216145244645;
  (*link).c32.re = 0.17573744458050627;
  (*link).c32.im = +0.677816070376633;
  (*link).c33.re = -0.343005196275533;
  (*link).c33.im = +0.4841295735668078;

  (*force).c1 = 0.2621841203510975;
  (*force).c2 = 0.624057296419485;
  (*force).c3 = 0.08224787691325641;
  (*force).c4 = 0.7378000709654313;
  (*force).c5 = 0.8777438407601124;
  (*force).c6 = 0.47206813871453646;
  (*force).c7 = 0.531778347595155;
  (*force).c8 = -0.8333242209822327;
}

void get_check_B_matrices(su3_dble *b1, su3_dble *b2)
{
  (*b1).c11.re = -0.053635798837777804;
  (*b1).c11.im = -0.0035156923612036477;
  (*b1).c12.re = -0.07248005854564346;
  (*b1).c12.im = 0.03066278894225473;
  (*b1).c13.re = 0.11379914418801403;
  (*b1).c13.im = -0.09132685783305354;
  (*b1).c21.re = 0.070938414467771;
  (*b1).c21.im = 0.09114548451165246;
  (*b1).c22.re = -0.04579441303757871;
  (*b1).c22.im = -0.008209865115226945;
  (*b1).c23.re = -0.05415904161261219;
  (*b1).c23.im = -0.08644239988774735;
  (*b1).c31.re = -0.07239619404094232;
  (*b1).c31.im = -0.09435803509188516;
  (*b1).c32.re = 0.10038330730915285;
  (*b1).c32.im = -0.08645735585821507;
  (*b1).c33.re = -0.06318436730761105;
  (*b1).c33.im = -0.008820687099832657;

  (*b2).c11.re = 0.1374864737631121;
  (*b2).c11.im = -0.0015585340231506727;
  (*b2).c12.re = -0.019634383959787987;
  (*b2).c12.im = 0.010107086708235083;
  (*b2).c13.re = 0.029681631018554644;
  (*b2).c13.im = -0.024723948722255504;
  (*b2).c21.re = 0.01917197666241721;
  (*b2).c21.im = 0.022923494652829057;
  (*b2).c22.re = 0.13913211224159872;
  (*b2).c22.im = -0.002820273823820807;
  (*b2).c23.re = -0.0160210704399449;
  (*b2).c23.im = -0.02349808514832701;
  (*b2).c31.re = -0.020807733250518816;
  (*b2).c31.im = -0.025571899348370817;
  (*b2).c32.re = 0.025888794579341512;
  (*b2).c32.im = -0.023323680668855647;
  (*b2).c33.re = 0.13549521680298432;
  (*b2).c33.im = -0.0030108873532083236;
}

void get_correct_arrays(complex_dble *p, complex_dble *r1, complex_dble *r2,
                        complex_dble *b1, complex_dble *b2)
{
  p[0].re = 0.9987479324418084;
  p[0].im = -0.1470599851428233;
  p[1].re = 0.6616130264324228;
  p[1].im = -0.03532563497662331;
  p[2].re = 0.41211380280769516;
  p[2].im = -0.007389695200180007;

  r1[0].re = 15.13625272918758;
  r1[0].im = -5.777931453612267;
  r1[1].re = 5.8482536798073115;
  r1[1].im = -1.3670479610689474;
  r1[2].re = 5.128323696103253;
  r1[2].im = -0.2908098600753183;

  r2[0].re = -1.9429529877611387;
  r2[0].im = 3.557021739307065;
  r2[1].re = -2.9759634852218126;
  r2[1].im = 0.862696583818781;
  r2[2].re = -1.2735653006182304;
  r2[2].im = 0.17855929518769176;

  b1[0].re = -0.00004495756002511082;
  b1[0].im = -0.007389649492107446;
  b1[1].re = +0.13170642492391182;
  b1[1].im = -0.0024287676488612994;
  b1[2].re = +0.035720529588229875;
  b1[2].im = -0.0003567451217298016;

  b2[0].re = 0.1487009529598715;
  b2[0].im = -0.0025321599024573736;
  b2[1].re = 0.03572052958822984;
  b2[1].im = -0.0003567451217297726;
  b2[2].re = 0.0074723613749966295;
  b2[2].im = -0.00004546076717229438;
}

void get_check_Gamma_matrix(su3_dble *g)
{
  (*g).c11.re = -0.8309814526494195;
  (*g).c11.im = 0.13760238736631297;
  (*g).c12.re = -0.31827041214104684;
  (*g).c12.im = -0.7027178853061343;
  (*g).c13.re = 0.29782872383930614;
  (*g).c13.im = -0.29946592185124377;
  (*g).c21.re = -0.5739422640028671;
  (*g).c21.im = 0.644878164220348;
  (*g).c22.re = -0.018656696737085532;
  (*g).c22.im = 0.1305610136915265;
  (*g).c23.re = 1.1972089336891416;
  (*g).c23.im = 0.46942477403759714;
  (*g).c31.re = 0.1931948870645372;
  (*g).c31.im = -0.6503976530266228;
  (*g).c32.re = 1.16594782939195;
  (*g).c32.im = 0.24580903539871057;
  (*g).c33.re = -0.05852573737286909;
  (*g).c33.im = -0.3043809123404668;
}

int main(int argc, char *argv[])
{
  int i, my_rank;
  double diff;
  double u, w;
  ch_drv0_t ch_exp_coeffs;
  ch_mat_coeff_pair_t coeff_pair;
  su3_dble input_exp_matrix, input_link, exp_matrix;
  su3_dble b1_mat, b2_mat, b1_check, b2_check;
  su3_dble gamma_mat, gamma_check, g_force;
  su3_alg_dble X, input_force;
  complex_dble p_array_check[3], r1_array_check[3], r2_array_check[3];
  complex_dble b1_array_check[3], b2_array_check[3];
  complex_dble r1_array[3], r2_array[3];
  complex_dble b1_array[3], b2_array[3];

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  new_test_module();

  if (my_rank == 0) {
    printf("Checks of the programs in the module stout_smearing\n");
    printf("-------------------------------------------\n\n");

    printf("%dx%dx%dx%d lattice, ", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid, ", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n\n", L0, L1, L2, L3);
  }

  get_inputs(&input_exp_matrix, &input_link, &input_force);
  project_to_su3alg(&input_exp_matrix, &X);

  cm3x3_unity(1, &exp_matrix);
  expXsu3_w_factors(1., &X, &exp_matrix, &ch_exp_coeffs);

  get_correct_arrays(p_array_check, r1_array_check, r2_array_check,
                     b1_array_check, b2_array_check);

  if (my_rank == 0) {
    register_test(1, "Check of the CH coefficients from the exponential");
    diff = norm_diff_array_complex(ch_exp_coeffs.p, p_array_check, 3);

    print_test_header(1);
    printf("abs|p_array(X) - p_array(X)[mathematica]| = %.1e (should be 0.0)\n",
           diff);

    if (diff > 1e-10) {
      fail_test(1);
      report_complex_array_diff(ch_exp_coeffs.p, p_array_check, 3, "p");
    }

    printf("\n-----\n\n");
  }

  construct_uw(&ch_exp_coeffs, &u, &w);

  if (my_rank == 0) {
    register_test(2, "Check of the {u, w} coefficients");
    diff = norm_diff_array_complex(ch_exp_coeffs.p, p_array_check, 3);

    print_test_header(2);

    diff = abs_diff_double(u, 0.8454540576850309);
    printf("abs|u - u[mathematica]| = %.1e (should be 0.0)\n", diff);
    fail_test_if(2, diff > 1e-10);

    diff = abs_diff_double(w, 0.3604727236378043);
    printf("abs|w - w[mathematica]| = %.1e (should be 0.0)\n", diff);
    fail_test_if(2, diff > 1e-10);

    printf("\n-----\n\n");
  }

  construct_r_coeffs(u, w, r1_array, r2_array);

  if (my_rank == 0) {
    register_test(3, "Check of the elements of the r-arrays");

    print_test_header(3);

    diff = norm_diff_array_complex(r1_array, r1_array_check, 3);
    printf(
        "abs|r1_array(X) - r1_array(X)[mathematica]| = %.1e (should be 0.0)\n",
        diff);

    if (diff > 1e-10) {
      fail_test(3);
      report_complex_array_diff(r1_array, r1_array_check, 3, "r1");
    }

    diff = norm_diff_array_complex(r2_array, r2_array_check, 3);
    printf(
        "abs|r2_array(X) - r2_array(X)[mathematica]| = %.1e (should be 0.0)\n",
        diff);

    if (diff > 1e-10) {
      fail_test(3);
      report_complex_array_diff(r2_array, r2_array_check, 3, "r2");
    }

    printf("\n-----\n\n");
  }

  construct_b_arrays(&ch_exp_coeffs, b1_array, b2_array);

  if (my_rank == 0) {
    register_test(4, "Check of the elements of the b-arrays");

    print_test_header(4);

    diff = norm_diff_array_complex(b1_array, b1_array_check, 3);
    printf(
        "abs|b1_array(X) - b1_array(X)[mathematica]| = %.1e (should be 0.0)\n",
        diff);

    if (diff > 1e-10) {
      fail_test(4);
      report_complex_array_diff(b1_array, b1_array_check, 3, "b1");
    }

    diff = norm_diff_array_complex(b2_array, b2_array_check, 3);
    printf(
        "abs|b2_array(X) - b2_array(X)[mathematica]| = %.1e (should be 0.0)\n",
        diff);

    if (diff > 1e-10) {
      fail_test(4);
      report_complex_array_diff(b2_array, b2_array_check, 3, "b2");
    }

    printf("\n-----\n\n");
  }

  get_check_B_matrices(&b1_check, &b2_check);

  coeff_pair.X = X;
  coeff_pair.coeff.d = ch_exp_coeffs.d;
  coeff_pair.coeff.t = ch_exp_coeffs.t;

  for (i = 0; i < 3; ++i)
    coeff_pair.coeff.p[i] = ch_exp_coeffs.p[i];

  construct_b_matrices(&coeff_pair, &b1_mat, &b2_mat);

  if (my_rank == 0) {
    register_test(5, "Check of the elements of the B-matrices");

    print_test_header(5);

    diff = norm_diff_su3(&b1_mat, &b1_check);
    printf("abs|B1(X) - B1(X)[mathematica]| = %.1e (should be 0.0)\n", diff);

    if (diff > 1e-10) {
      fail_test(5);
      report_su3_diff(&b1_mat, &b1_check, "b1");
    }

    diff = norm_diff_su3(&b2_mat, &b2_check);
    printf("abs|B2(X) - B2(X)[mathematica]| = %.1e (should be 0.0)\n", diff);

    if (diff > 1e-10) {
      fail_test(5);
      report_su3_diff(&b2_mat, &b2_check, "b2");
    }

    printf("\n-----\n\n");
  }

  get_check_Gamma_matrix(&gamma_check);

  su3alg_to_cm3x3(&input_force, &g_force);

  compute_unsmearing_gamma_matrix(&g_force, &input_link, &coeff_pair,
                                  &gamma_mat);

  if (my_rank == 0) {
    register_test(6, "Check of the elements of the Gamma-matrices");
    print_test_header(6);

    diff = norm_diff_su3(&gamma_mat, &gamma_check);
    printf("abs|Gamma(X) - Gamma(X)[mathematica]| = %.1e (should be 0.0)\n",
           diff);

    if (diff > 1e-10) {
      fail_test(6);
      report_su3_diff(&gamma_mat, &gamma_check, "gamma");
    }

    printf("\n-----\n\n");
  }

  if (my_rank == 0)
    report_test_results();

  MPI_Finalize();
  return 0;
}
