
/*******************************************************************************
 *
 * File test01_projection_impl.c
 *
 * Author (2017, 2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Tests of the expXsu3 and expXsu3_w_factors functions
 *
 *******************************************************************************/

#define MAIN_PROGRAM

#include "global.h"
#include "linalg.h"
#include "mpi.h"
#include "su3fcts.h"

#include <devel/testing_utilities/data_type_diffs.c>
#include <devel/testing_utilities/diff_printing.c>
#include <devel/testing_utilities/test_counter.c>

void get_input_test1to3(su3_dble *u)
{
  (*u).c11.re = 0.17768103740057795;
  (*u).c11.im = 0.3925176720290544;
  (*u).c12.re = 0.2155407593189902;
  (*u).c12.im = -0.45484851438397955;
  (*u).c13.re = 0.3058823882466797;
  (*u).c13.im = -0.3511604287572472;
  (*u).c21.re = 0.1709159521922632;
  (*u).c21.im = 0.020851688815168412;
  (*u).c22.re = 0.4044586969877253;
  (*u).c22.im = 0.75088098310197;
  (*u).c23.re = 0.16478642900001317;
  (*u).c23.im = -0.12107992761308939;
  (*u).c31.re = -0.11674410269022939;
  (*u).c31.im = 0.4487211540607676;
  (*u).c32.re = 0.6895597606261576;
  (*u).c32.im = -0.38430832869281684;
  (*u).c33.re = -0.914257942795587;
  (*u).c33.im = 0.24086469783393571;
}

void get_input_test4to7(su3_dble *u)
{
  (*u).c11.re = -4.371247980099447;
  (*u).c11.im = -6.323045681986876;
  (*u).c12.re = 7.19091279847877;
  (*u).c12.im = 1.6334626242066932;
  (*u).c13.re = 3.8905101698288362;
  (*u).c13.im = 4.040319342448292;
  (*u).c21.re = -7.82700047313088;
  (*u).c21.im = 6.958375399368599;
  (*u).c22.re = -5.9007839120868795;
  (*u).c22.im = 6.041752017639212;
  (*u).c23.re = -7.246609145455501;
  (*u).c23.im = 8.269029639314823;
  (*u).c31.re = -0.14330937510348463;
  (*u).c31.im = 2.488481658525238;
  (*u).c32.re = -1.2357222526768723;
  (*u).c32.im = -5.638270044238396;
  (*u).c33.re = -3.2078793549962406;
  (*u).c33.im = -8.698577114293997;
}

void get_input_test8(su3_alg_dble *X)
{
  (*X).c1 = -0.26094122516234863;
  (*X).c2 = 0.7931803438936993;
  (*X).c3 = 0.8397374068639287;
  (*X).c4 = -0.5457791239450867;
  (*X).c5 = -0.17381832949749043;
  (*X).c6 = 0.7499041221199203;
  (*X).c7 = -0.6512063609528425;
  (*X).c8 = 0.3063410640165425;
}

void get_solution_test1(su3_alg_dble *X)
{
  (*X).c1 = -0.11945443702430521;
  (*X).c2 = 0.050550991398372894;
  (*X).c3 = 0.022312403563363503;
  (*X).c4 = -0.21699841278440557;
  (*X).c5 = 0.21131324546845454;
  (*X).c6 = 0.04878036265176022;
  (*X).c7 = -0.26238666581307224;
  (*X).c8 = -0.2526941281529531;
}

void get_solution_test2and3(su3_dble *u)
{
  (*u).c11.re = 0.9514207869244644;
  (*u).c11.im = -0.07176851038113856;
  (*u).c12.re = 0.07728455084594618;
  (*u).c12.im = -0.22392737852333292;
  (*u).c13.re = 0.178401540594238;
  (*u).c13.im = 0.04128298423704425;
  (*u).c21.re = 0.034850537278757164;
  (*u).c21.im = -0.18905204323225108;
  (*u).c22.re = 0.8716345130523773;
  (*u).c22.im = 0.26923193986802724;
  (*u).c23.re = -0.23869239449924384;
  (*u).c23.im = -0.2717317501142523;
  (*u).c31.re = -0.2236816549978313;
  (*u).c31.im = 0.0515659707969263;
  (*u).c32.re = 0.2606207107981202;
  (*u).c32.im = -0.20911219634955547;
  (*u).c33.re = 0.8882540899153168;
  (*u).c33.im = -0.21601174519655503;
}

void get_solution_test4and5(su3_dble *u)
{
  (*u).c11.re = -3.7877257791944787;
  (*u).c11.im = -2.973671804522236;
  (*u).c12.re = 7.867999209186197;
  (*u).c12.im = 1.769419994721958;
  (*u).c13.re = 5.0699037480109865;
  (*u).c13.im = 4.1423397284238455;
  (*u).c21.re = -9.333019365379835;
  (*u).c21.im = 4.00887351655639;
  (*u).c22.re = -7.447682389858476;
  (*u).c22.im = 4.056587519373624;
  (*u).c23.re = -9.241247705953462;
  (*u).c23.im = 7.609818259859227;
  (*u).c31.re = 1.1292708350304033;
  (*u).c31.im = 6.880520198518216;
  (*u).c32.re = -4.282736721997826;
  (*u).c32.im = -1.927323540909157;
  (*u).c33.re = -5.966446999820008;
  (*u).c33.im = -4.066299727096797;
}

void get_solution_test6and7(su3_dble *u)
{
  (*u).c11.re = 1.354946000755315;
  (*u).c11.im = -1.3091025831526082;
  (*u).c12.re = 8.477533335299754;
  (*u).c12.im = -2.4000782948903647;
  (*u).c13.re = 11.310328920548754;
  (*u).c13.im = 0.9129123936575301;
  (*u).c21.re = -6.905759404103351;
  (*u).c21.im = 7.028514935332748;
  (*u).c22.re = 2.4258976036639526;
  (*u).c22.im = 3.338953109550388;
  (*u).c23.re = 4.859253692400438;
  (*u).c23.im = 5.3270772595646285;
  (*u).c31.re = 0.7393424815800449;
  (*u).c31.im = -8.590513841438627;
  (*u).c32.re = 7.997848657504482;
  (*u).c32.im = -0.6232385939640022;
  (*u).c33.re = 6.882928316960657;
  (*u).c33.im = -3.191305171948967;
}

void get_solution_test8(su3_dble *u)
{
  (*u).c11.re = 0.;
  (*u).c11.im = 0.5322391187313507;
  (*u).c12.re = 0.8397374068639287;
  (*u).c12.im = -0.5457791239450867;
  (*u).c13.re = -0.17381832949749043;
  (*u).c13.im = 0.7499041221199203;
  (*u).c21.re = -0.8397374068639287;
  (*u).c21.im = -0.5457791239450867;
  (*u).c22.re = 0.;
  (*u).c22.im = 1.3150627942183966;
  (*u).c23.re = -0.6512063609528425;
  (*u).c23.im = 0.3063410640165425;
  (*u).c31.re = 0.17381832949749043;
  (*u).c31.im = 0.7499041221199203;
  (*u).c32.re = 0.6512063609528425;
  (*u).c32.im = 0.3063410640165425;
  (*u).c33.re = 0.;
  (*u).c33.im = -1.8473019129497472;
}

int main(int argc, char *argv[])
{
  int my_rank;
  su3_dble input_matrix, exp_matrix;
  su3_dble solution_matrix;
  su3_alg_dble input_alg;
  su3_alg_dble output_alg, correct_alg;
  ch_drv0_t s;

  double diff;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  new_test_module();

  if (my_rank == 0) {
    printf("Tests of the expXsu3 and expXsu3_w_factors functions\n");
    printf("-------------------------------------------\n\n");

    printf("%dx%dx%dx%d lattice, ", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid, ", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n\n", L0, L1, L2, L3);

    printf("-------------------------------------------\n\n");
  }

  get_input_test1to3(&input_matrix);
  project_to_su3alg(&input_matrix, &output_alg);

  get_solution_test1(&correct_alg);

  if (my_rank == 0) {
    register_test(1, "Check of project_to_su3alg()");
    print_test_header(1);

    diff = abs_diff_su3_alg(&output_alg, &correct_alg);

    printf("diff|X - X(mathematica)| = %.1e (should be 0.0)\n", diff);

    if (diff > 1e-10) {
      fail_test(1);
    }

    printf("\n-----\n\n");
  }

  cm3x3_unity(1, &exp_matrix);
  expXsu3(1., &output_alg, &exp_matrix);

  get_solution_test2and3(&solution_matrix);

  if (my_rank == 0) {
    register_test(2,
                  "Check of project_to_su3alg() in combination with expXsu3()");
    print_test_header(2);

    diff = norm_diff_su3(&exp_matrix, &solution_matrix);
    printf("abs|exp(X) - exp(X)[mathematica]| = %.1e (should be 0.0)\n\n",
           diff);

    if (diff > 1e-10) {
      fail_test(2);
      report_su3_diff(&exp_matrix, &solution_matrix, "exp(X)");
    }

    printf("-----\n\n");
  }

  cm3x3_unity(1, &exp_matrix);
  expXsu3_w_factors(1., &output_alg, &exp_matrix, &s);

  get_solution_test2and3(&solution_matrix);

  if (my_rank == 0) {
    register_test(
        3,
        "Check of project_to_su3alg() in combination with expXsu3_w_factors()");
    print_test_header(3);

    diff = norm_diff_su3(&exp_matrix, &solution_matrix);
    printf("abs|exp(X) - exp(X)[mathematica]| = %.1e (should be 0.0)\n\n",
           diff);

    if (diff > 1e-10) {
      fail_test(3);
      report_su3_diff(&exp_matrix, &solution_matrix, "exp(X)");
    }

    printf("-----\n\n");
  }

  get_input_test4to7(&exp_matrix);

  expXsu3(1., &output_alg, &exp_matrix);

  if (my_rank == 0) {
    register_test(4,
                  "Check of project_to_su3alg() in combination with expXsu3()");
    print_test_header(4);

    get_solution_test4and5(&solution_matrix);
    diff = norm_diff_su3(&exp_matrix, &solution_matrix);

    printf("abs|exp(X)*Y - exp(X)*Y[mathematica]| = %.1e (should be 0.0)\n\n",
           diff);

    if (diff > 1e-10) {
      fail_test(4);
      report_su3_diff(&exp_matrix, &solution_matrix, "exp(X)");
    }

    printf("-----\n\n");
  }

  get_input_test4to7(&exp_matrix);

  expXsu3_w_factors(1., &output_alg, &exp_matrix, &s);

  if (my_rank == 0) {
    register_test(
        5,
        "Check of project_to_su3alg() in combination with expXsu3_w_factors()");
    print_test_header(5);

    get_solution_test4and5(&solution_matrix);
    diff = norm_diff_su3(&exp_matrix, &solution_matrix);

    printf("abs|exp(X)*Y - exp(X)*Y[mathematica]| = %.1e (should be 0.0)\n\n",
           diff);

    if (diff > 1e-10) {
      fail_test(5);
      report_su3_diff(&exp_matrix, &solution_matrix, "exp(X)");
    }

    printf("-----\n\n");
  }

  get_input_test4to7(&exp_matrix);

  expXsu3(10., &output_alg, &exp_matrix);

  if (my_rank == 0) {
    register_test(6,
                  "Check of project_to_su3alg() in combination with expXsu3()");
    print_test_header(6);

    get_solution_test6and7(&solution_matrix);
    diff = norm_diff_su3(&exp_matrix, &solution_matrix);

    printf("abs|exp(10*X)*Y - exp(10*X)*Y[mathematica]| = %.1e (should be "
           "0.0)\n\n",
           diff);

    if (diff > 1e-10) {
      fail_test(6);
      report_su3_diff(&exp_matrix, &solution_matrix, "exp(X)");
    }

    printf("-----\n\n");
  }

  get_input_test4to7(&exp_matrix);

  expXsu3(10., &output_alg, &exp_matrix);

  if (my_rank == 0) {
    register_test(
        7,
        "Check of project_to_su3alg() in combination with expXsu3_w_factors()");
    print_test_header(7);

    get_solution_test6and7(&solution_matrix);
    diff = norm_diff_su3(&exp_matrix, &solution_matrix);

    printf("abs|exp(10*X)*Y - exp(10*X)*Y[mathematica]| = %.1e (should be "
           "0.0)\n\n",
           diff);

    if (diff > 1e-10) {
      fail_test(7);
      report_su3_diff(&exp_matrix, &solution_matrix, "exp(X)");
    }

    printf("-----\n\n");
  }

  get_input_test8(&input_alg);
  su3alg_to_cm3x3(&input_alg, &exp_matrix);

  if (my_rank == 0) {
    register_test(8, "Check of su3alg_to_cm3x3");
    print_test_header(8);

    get_solution_test8(&solution_matrix);
    diff = norm_diff_su3(&exp_matrix, &solution_matrix);

    printf("abs|X_mat - X_mat[mathematica]| = %.1e (should be "
           "0.0)\n\n",
           diff);

    if (diff > 1e-10) {
      fail_test(8);
      report_su3_diff(&exp_matrix, &solution_matrix, "X");
    }

    printf("-----\n\n");
  }

  cm3x3_unity(1, &exp_matrix);
  expXsu3_w_factors(1., &output_alg, &exp_matrix, &s);
  ch2mat(s.p, &output_alg, &exp_matrix);

  get_solution_test2and3(&solution_matrix);

  if (my_rank == 0) {
    register_test(9,
                  "Check of expXsu3_w_factors() in combination with ch2mat()");
    print_test_header(9);

    diff = norm_diff_su3(&exp_matrix, &solution_matrix);

    printf("abs|exp(X) - exp(X)[mathematica]| = %.1e (should be 0.0)\n\n",
           diff);

    if (diff > 1e-10) {
      fail_test(9);
      report_su3_diff(&exp_matrix, &solution_matrix, "exp(X)");
    }

    printf("-----\n\n");
  }

  if (my_rank == 0) {
    report_test_results();
  }

  MPI_Finalize();
  return 0;
}
