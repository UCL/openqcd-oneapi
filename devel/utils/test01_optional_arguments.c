
/*
 * Created: 14-08-2017
 * Modified: Mon 14 Aug 2017 14:09:28 BST
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
#include "utils.h"

#include <devel/testing_utilities/data_type_diffs.c>
#include <devel/testing_utilities/test_counter.c>

int main(int argc, char *argv[])
{
  int my_rank, check;
  FILE *fin = NULL;
  int opt_int, int_exp;
  long section_pos;
  double opt_double, double_exp;
  char opt_string[256], string_exp[256];
  char const true_str[] = KGRN "true" KNRM;
  char const false_str[] = KRED "false" KNRM;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  new_test_module();

  if (my_rank == 0) {
    fin = freopen("test01.in", "r", stdin);
    error_root(!fin, 1, "main", "Unable to open input file \"test01.in\"");

    printf("Checks of read_line and read_optional_line\n");
    printf("-------------------------------------------\n\n");

    printf("%dx%dx%dx%d lattice, ", NPROC0 * L0, NPROC1 * L1, NPROC2 * L2,
           NPROC3 * L3);
    printf("%dx%dx%dx%d process grid, ", NPROC0, NPROC1, NPROC2, NPROC3);
    printf("%dx%dx%dx%d local lattice\n", L0, L1, L2, L3);
    printf("\n-------------------------------------------\n\n");
  }

  if (my_rank == 0) {
    find_section("Test 01");
    read_line("par1_d", "%lf", &opt_double);
    read_line("par2_i", "%d", &opt_int);
    read_line("par3_s", "%s", opt_string);
  }

  { /* Test 1 */
    if (my_rank == 0) {
      register_test(1, "Checks of mandatory parameters");
      print_test_header(1);

      double_exp = -9.24;
      check = (abs_diff_double(opt_double, double_exp) < 1e-12);

      printf("(%s)\tpar1_d == %.2f\n", check ? true_str : false_str, double_exp);
      fail_test_if(1, !check);

      int_exp = 175154;
      check = (opt_int == int_exp);

      printf("(%s)\tpar2_i == %d\n", check ? true_str : false_str, int_exp);
      fail_test_if(1, !check);

      strcpy(string_exp, "Hello");
      check = (strcmp(string_exp, opt_string) == 0);

      printf("(%s)\tpar3_s == \"%s\"\n", check ? true_str : false_str, string_exp);
      fail_test_if(1, !check);

      printf("\n-------------------------------------------\n\n");
    }
  }

  if (my_rank == 0) {
    find_section("Test 02");
    read_optional_line("opt1_d", "%lf", &opt_double, 55.21);
    read_optional_line("opt2_i", "%d", &opt_int, -123);
    read_optional_line("opt3_s", "%s", opt_string, "Opt used");
  }

  { /* Test 2 */
    if (my_rank == 0) {
      register_test(2, "Check given optional parameters");
      print_test_header(2);

      double_exp = 24.12;
      check = (abs_diff_double(opt_double, double_exp) < 1e-12);

      printf("(%s)\topt1_d == %.2f\n", check ? true_str : false_str, double_exp);
      fail_test_if(2, !check);

      int_exp = 114;
      check = (opt_int == int_exp);

      printf("(%s)\topt2_i == %d\n", check ? true_str : false_str, int_exp);
      fail_test_if(2, !check);

      strcpy(string_exp, "World");
      check = (strcmp(string_exp, opt_string) == 0);

      printf("(%s)\topt3_s == \"%s\"\n", check ? true_str : false_str, string_exp);
      fail_test_if(2, !check);

      printf("\n-------------------------------------------\n\n");
    }
  }

  if (my_rank == 0) {
    find_section("Test 03");
    read_optional_line("opt1_d", "%lf", &opt_double, 55.21);
    read_optional_line("opt2_i", "%d", &opt_int, -123);
    read_optional_line("opt3_s", "%s", opt_string, "Opt used");
  }

  { /* Test 3 */
    if (my_rank == 0) {
      register_test(3, "Check missing optional parameters");
      print_test_header(3);

      double_exp = 55.21;
      check = (abs_diff_double(opt_double, double_exp) < 1e-12);

      printf("(%s)\topt1_d == %.2f\n", check ? true_str : false_str, double_exp);
      fail_test_if(3, !check);

      int_exp = -123;
      check = (opt_int == int_exp);

      printf("(%s)\topt2_i == %d\n", check ? true_str : false_str, int_exp);
      fail_test_if(3, !check);

      strcpy(string_exp, "Opt used");
      check = (strcmp(string_exp, opt_string) == 0);

      printf("(%s)\topt3_s == \"%s\"\n", check ? true_str : false_str, string_exp);
      fail_test_if(3, !check);

      printf("\n-------------------------------------------\n\n");
    }
  }

  if (my_rank == 0) {
    find_section("Test Optional 04");
    read_line("par1_d", "%lf", &opt_double);
    read_line("par2_i", "%d", &opt_int);
    read_line("par3_s", "%s", opt_string);
  }

  { /* Test 4 */
    if (my_rank == 0) {
      register_test(4, "Check parameters in optional section");
      print_test_header(4);

      double_exp = 5812.123;
      check = (abs_diff_double(opt_double, double_exp) < 1e-12);

      printf("(%s)\tpar1_d == %.2f\n", check ? true_str : false_str, double_exp);
      fail_test_if(4, !check);

      int_exp = -61234;
      check = (opt_int == int_exp);

      printf("(%s)\tpar2_i == %d\n", check ? true_str : false_str, int_exp);
      fail_test_if(4, !check);

      strcpy(string_exp, "Optional");
      check = (strcmp(string_exp, opt_string) == 0);

      printf("(%s)\tpar3_s == \"%s\"\n", check ? true_str : false_str, string_exp);
      fail_test_if(4, !check);

      printf("\n-------------------------------------------\n\n");
    }
  }

  section_pos = 0;

  if (my_rank == 0) {
    section_pos = find_optional_section("No Such Section");
  }

  { /* Test 5 */
    if (my_rank == 0) {
      register_test(5, "Find an optional (missing) section");
      print_test_header(5);

      check = (section_pos == No_Section_Found);
      printf("(%s)\t No_Section_Found\n", check ? true_str : false_str);

      fail_test_if(5, !check);

      printf("\n-------------------------------------------\n\n");
    }
  }


  if (my_rank == 0)
    report_test_results();

  MPI_Finalize();
  return 0;
}
