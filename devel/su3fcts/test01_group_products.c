
/*
 * Created: 09-05-2017
 * Author: Jonas R. Glesaaen (jonas@glesaaen.com)
 */

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "su3.h"
#include "global.h"
#include "su3fcts.h"

#include <devel/testing_utilities/data_type_diffs.c>
#include <devel/testing_utilities/diff_printing.c>
#include <devel/testing_utilities/test_counter.c>

void get_input_su3xsu3alg(su3_dble *u, su3_alg_dble *X)
{
  (*u).c11.re = -0.8089904336390457;
  (*u).c11.im = -0.5780987843305794;
  (*u).c12.re = 0.07383233720600613;
  (*u).c12.im = -0.8453077688910371;
  (*u).c13.re = 0.9515543237001816;
  (*u).c13.im = 0.4304256738675072;
  (*u).c21.re = -0.7100553144931094;
  (*u).c21.im = 0.3339277362652724;
  (*u).c22.re = -0.06739928826893848;
  (*u).c22.im = -0.9749873562713023;
  (*u).c23.re = 0.986352660016832;
  (*u).c23.im = 0.4159055366284763;
  (*u).c31.re = 0.9246110088679984;
  (*u).c31.im = -0.734731628577566;
  (*u).c32.re = -0.17359388382576313;
  (*u).c32.im = 0.3492875557281043;
  (*u).c33.re = -0.70645178316208;
  (*u).c33.im = 0.7116984589704867;

  (*X).c1 = -0.1537159810143467;
  (*X).c2 = 0.9265856723996877;
  (*X).c3 = -0.6958520203166936;
  (*X).c4 = -0.26750585013911543;
  (*X).c5 = 0.9846089565879708;
  (*X).c6 = 0.6417210599281731;
  (*X).c7 = -0.9643922615171228;
  (*X).c8 = -0.21068668709280525;
}

void get_input_su3algxsu3(su3_dble *u, su3_alg_dble *X)
{
  (*u).c11.re = -0.3073324597525846;
  (*u).c11.im = 0.7389022727913868;
  (*u).c12.re = 0.6590976565600037;
  (*u).c12.im = 0.2284608038782161;
  (*u).c13.re = 0.8767010887405462;
  (*u).c13.im = 0.2921156366488642;
  (*u).c21.re = 0.006497027002526856;
  (*u).c21.im = -0.6615441510499909;
  (*u).c22.re = -0.4984866825216483;
  (*u).c22.im = -0.4815777882447261;
  (*u).c23.re = 0.5490611203436737;
  (*u).c23.im = 0.03391031885606122;
  (*u).c31.re = 0.20054239691533793;
  (*u).c31.im = 0.6515432545060991;
  (*u).c32.re = -0.7717468825194458;
  (*u).c32.im = 0.39781706964827945;
  (*u).c33.re = 0.29332746867409387;
  (*u).c33.im = 0.8429261117219027;

  (*X).c1 = -0.9615516540947029;
  (*X).c2 = -0.349051014726391;
  (*X).c3 = 0.7653533839438551;
  (*X).c4 = -0.7316324870309847;
  (*X).c5 = -0.6147458636951275;
  (*X).c6 = -0.727595470020189;
  (*X).c7 = 0.4497306467521951;
  (*X).c8 = 0.4822589471881096;
}

void get_input_su3algxsu3dag(su3_dble *u, su3_alg_dble *X)
{
  (*u).c11.re = -0.12625835327797397;
  (*u).c11.im = -0.4410475674075758;
  (*u).c12.re = -0.9329196854202171;
  (*u).c12.im = -0.9863422961481985;
  (*u).c13.re = -0.6408850534741375;
  (*u).c13.im = -0.22602954424692845;
  (*u).c21.re = 0.4761229606513204;
  (*u).c21.im = 0.8803999300896592;
  (*u).c22.re = 0.38112037502485263;
  (*u).c22.im = -0.9261229624484293;
  (*u).c23.re = 0.5266277258182996;
  (*u).c23.im = 0.4336919155416319;
  (*u).c31.re = 0.0472804459350189;
  (*u).c31.im = -0.14122218882131854;
  (*u).c32.re = -0.26772642690400517;
  (*u).c32.im = -0.7888757869643865;
  (*u).c33.re = 0.5158409994025233;
  (*u).c33.im = 0.025984325258415097;

  (*X).c1 = 0.7558696910859912;
  (*X).c2 = -0.13030962287393422;
  (*X).c3 = 0.03187534072426512;
  (*X).c4 = 0.40207174456677075;
  (*X).c5 = 0.19910159047835885;
  (*X).c6 = -0.9398128510157715;
  (*X).c7 = 0.32772633585871347;
  (*X).c8 = -0.727331197008227;
}

void get_input_su3dagxsu3alg(su3_dble *u, su3_alg_dble *X)
{
  (*u).c11.re = -0.40096347498066454;
  (*u).c11.im = -0.9124165201164551;
  (*u).c12.re = -0.5051253294838323;
  (*u).c12.im = -0.1473012396754756;
  (*u).c13.re = -0.6541788570481843;
  (*u).c13.im = -0.9971262978586273;
  (*u).c21.re = -0.5120051870163964;
  (*u).c21.im = 0.7120184295181202;
  (*u).c22.re = 0.8159139818718701;
  (*u).c22.im = -0.7699558170024701;
  (*u).c23.re = -0.3913042207195008;
  (*u).c23.im = 0.49929362961173673;
  (*u).c31.re = -0.4190469291386747;
  (*u).c31.im = -0.18739205628921862;
  (*u).c32.re = -0.9341635316717287;
  (*u).c32.im = -0.8624256212123353;
  (*u).c33.re = 0.6444273480123393;
  (*u).c33.im = 0.8093839512999259;

  (*X).c1 = 0.8050811431254399;
  (*X).c2 = 0.17217745324545453;
  (*X).c3 = -0.6034033354048396;
  (*X).c4 = 0.8332457523690033;
  (*X).c5 = -0.16940515134562562;
  (*X).c6 = 0.23085265047040693;
  (*X).c7 = 0.8969350493953434;
  (*X).c8 = -0.4696664564967401;
}

void get_input_tracesu3algxsu3(su3_dble *u, su3_alg_dble *X)
{
  (*u).c11.re = -0.3062533170270121;
  (*u).c11.im = +0.4230473349313848;
  (*u).c12.re = 0.800876666195145;
  (*u).c12.im = +0.4617608230969208;
  (*u).c13.re = 0.605177740809971;
  (*u).c13.im = +0.8175281576308668;
  (*u).c21.re = 0.5451801755742425;
  (*u).c21.im = -0.9850560903352266;
  (*u).c22.re = 0.030076194360162845;
  (*u).c22.im = +0.830371110862262;
  (*u).c23.re = -0.5169206463064557;
  (*u).c23.im = -0.6667216588243234;
  (*u).c31.re = 0.5510934290713743;
  (*u).c31.im = +0.026085981304202477;
  (*u).c32.re = -0.475340737798327;
  (*u).c32.im = -0.3341584615142059;
  (*u).c33.re = -0.5615455480974028;
  (*u).c33.im = -0.7099985087185705;

  (*X).c1 = 0.7268731775480664;
  (*X).c2 = 0.4222869870808954;
  (*X).c3 = -0.5431606334203591;
  (*X).c4 = 0.25496398266719655;
  (*X).c5 = -0.8887290534231984;
  (*X).c6 = -0.82113919816423;
  (*X).c7 = -0.8971988781576621;
  (*X).c8 = 0.4953818709328388;
}

void get_solution_su3xsu3alg(su3_dble *u)
{
  (*u).c11.re = -0.94107549275084;
  (*u).c11.im = -1.0463724121358218;
  (*u).c12.re = 2.4597740996851654;
  (*u).c12.im = 0.9244106478142476;
  (*u).c13.re = 0.18895633940568934;
  (*u).c13.im = -2.1983569710556807;
  (*u).c21.re = -1.803864384198602;
  (*u).c21.im = -0.9857384915262242;
  (*u).c22.re = 2.8254292711876725;
  (*u).c22.im = 0.06769245716002081;
  (*u).c23.re = -0.2191572369947733;
  (*u).c23.im = -1.1518974669917714;
  (*u).c31.re = 0.7793594804522128;
  (*u).c31.im = -0.14999600829917403;
  (*u).c32.re = -1.802315692996394;
  (*u).c32.im = 0.8849041924411736;
  (*u).c33.re = 3.051174493421213;
  (*u).c33.im = 0.9874118488669827;
}

void get_solution_su3algsu3(su3_dble *u)
{
  (*u).c11.re = 0.840149931308454;
  (*u).c11.im = -0.6547250089601295;
  (*u).c12.re = 0.32944301667377596;
  (*u).c12.im = -0.5507201486316742;
  (*u).c13.re = 1.26087058273911;
  (*u).c13.im = -2.2563734404050937;
  (*u).c21.re = 1.5931054316702642;
  (*u).c21.im = 0.05929205168248747;
  (*u).c22.re = -0.11819371938618811;
  (*u).c22.im = -1.6349859239335356;
  (*u).c23.re = -0.7852317773996235;
  (*u).c23.im = 0.5198057283913184;
  (*u).c31.re = 0.8364530875440991;
  (*u).c31.im = 0.9256679304963751;
  (*u).c32.re = 1.1326392970712602;
  (*u).c32.im = -0.15961408102645455;
  (*u).c33.re = 0.7102757822530695;
  (*u).c33.im = -0.28604474469003144;
}

void get_solution_su3algsu3dag(su3_dble *u)
{
  (*u).c11.re = -0.6173950100136967;
  (*u).c11.im = 0.22467201824756322;
  (*u).c12.re = -0.11221331087057923;
  (*u).c12.im = -0.10067852378803266;
  (*u).c13.re = -0.335777136449674;
  (*u).c13.im = -0.5428903088548696;
  (*u).c21.re = 1.4006775778850278;
  (*u).c21.im = 2.007287910796576;
  (*u).c22.re = 1.7166987581291782;
  (*u).c22.im = -0.9314847320188364;
  (*u).c23.re = 1.3872396391612176;
  (*u).c23.im = 0.07042553964864318;
  (*u).c31.re = 1.233023752653649;
  (*u).c31.im = -0.2653153709372616;
  (*u).c32.re = 0.06732996590818957;
  (*u).c32.im = -0.3175818172305147;
  (*u).c33.re = 0.8112365664003439;
  (*u).c33.im = 0.38798474740266803;
}

void get_solution_su3dagsu3alg(su3_dble *u)
{
  (*u).c11.re = -0.7215748546635955;
  (*u).c11.im = -1.313098359875364;
  (*u).c12.re = -1.0783265529279358;
  (*u).c12.im = -0.11966680749947886;
  (*u).c13.re = -1.0226915011674123;
  (*u).c13.im = -0.8383600776132951;
  (*u).c21.re = -0.6505339536479997;
  (*u).c21.im = 0.5812579129307154;
  (*u).c22.re = 2.532177362193695;
  (*u).c22.im = -2.0178420169578137;
  (*u).c23.re = 0.7476681083179003;
  (*u).c23.im = -0.26456382819660473;
  (*u).c31.re = -0.49851247930399556;
  (*u).c31.im = -1.2549759823095428;
  (*u).c32.re = -2.1122441821433604;
  (*u).c32.im = -0.16077262805868453;
  (*u).c33.re = -0.33193949886504337;
  (*u).c33.im = -0.28708415717766;
}

void get_solution_tracesu3algsu3(complex_dble *c)
{
  (*c).re = 1.7583865954637918;
  (*c).im = -0.2238176754231529;
}

int main(int argc, char *argv[])
{
  int my_rank;
  su3_dble input_matrix, output_matrix, correct_result;
  su3_alg_dble input_alg;
  complex_dble output_complex, correct_complex;
  double diff = 0.;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  new_test_module();

  if (my_rank == 0) {
    printf("Checks of the programs in the module su3prod\n");
    printf("-------------------------------------------\n\n");

    printf("%dx%dx%dx%d lattice, ", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid, ", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n\n", L0, L1, L2, L3);
    printf("-------------------------------------------\n\n");
  }

  get_input_su3xsu3alg(&input_matrix, &input_alg);

  su3xsu3alg(&input_matrix, &input_alg, &output_matrix);

  get_solution_su3xsu3alg(&correct_result);

  if (my_rank == 0) {
    register_test(1, "Check of the matrix product su3xsu3alg");
    print_test_header(1);

    diff = norm_diff_su3(&output_matrix, &correct_result);

    printf("abs|result - result[mathematica]| = %.1e (should be 0.0)\n", diff);

    if (diff > 1e-10) {
      fail_test(1);
      report_su3_diff(&output_matrix, &correct_result, "su3xsu3alg");
    }

    printf("\n-----\n\n");
  }

  get_input_su3algxsu3(&input_matrix, &input_alg);

  su3algxsu3(&input_alg, &input_matrix, &output_matrix);

  get_solution_su3algsu3(&correct_result);

  if (my_rank == 0) {
    register_test(2, "Check of the matrix product su3algxsu3");
    print_test_header(2);

    diff = norm_diff_su3(&output_matrix, &correct_result);

    printf("abs|result - result[mathematica]| = %.1e (should be 0.0)\n", diff);

    if (diff > 1e-10) {
      fail_test(2);
      report_su3_diff(&output_matrix, &correct_result, "su3algxsu3");
    }

    printf("\n-----\n\n");
  }

  get_input_su3algxsu3dag(&input_matrix, &input_alg);

  su3algxsu3dag(&input_alg, &input_matrix, &output_matrix);

  get_solution_su3algsu3dag(&correct_result);

  if (my_rank == 0) {
    register_test(3, "Check of the matrix product su3algxsu3dag");
    print_test_header(3);

    diff = norm_diff_su3(&output_matrix, &correct_result);

    printf("abs|result - result[mathematica]| = %.1e (should be 0.0)\n", diff);

    if (diff > 1e-10) {
      fail_test(3);
      report_su3_diff(&output_matrix, &correct_result, "su3algxsu3");
    }

    printf("\n-----\n\n");
  }

  get_input_su3dagxsu3alg(&input_matrix, &input_alg);

  su3dagxsu3alg(&input_matrix, &input_alg, &output_matrix);

  get_solution_su3dagsu3alg(&correct_result);

  if (my_rank == 0) {
    register_test(4, "Check of the matrix product su3dagxsu3alg");
    print_test_header(4);

    diff = norm_diff_su3(&output_matrix, &correct_result);

    printf("abs|result - result[mathematica]| = %.1e (should be 0.0)\n", diff);

    if (diff > 1e-10) {
      fail_test(4);
      report_su3_diff(&output_matrix, &correct_result, "su3algxsu3");
    }

    printf("\n-----\n\n");
  }

  get_input_tracesu3algxsu3(&input_matrix, &input_alg);

  su3algxsu3_tr(&input_alg, &input_matrix, &output_complex);

  get_solution_tracesu3algsu3(&correct_complex);

  if (my_rank == 0) {
    register_test(5, "Check of the composite expression trsu3algxsu3");
    print_test_header(5);

    diff = norm_diff_complex(output_complex, correct_complex);

    printf("abs|result - result[mathematica]| = %.1e (should be 0.0)\n", diff);

    if (diff > 1e-10) {
      fail_test(5);
      report_complex_diff(output_complex, correct_complex, "result");
    }

    printf("\n-----\n\n");
  }

  if (my_rank == 0)
    report_test_results();

  MPI_Finalize();
  return 0;
}
