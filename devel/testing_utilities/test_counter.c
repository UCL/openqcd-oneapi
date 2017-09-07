#define TEST_COUNTER_C

#include "test_counter.h"
#include <string.h>
#include <stdio.h>

#define MAX_NUM_TESTS 64
#define MAX_NAME_LENGTH 256

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

static int num_failed_tests, num_registered_tests;
static int failed_tests[MAX_NUM_TESTS+1], registered_tests[MAX_NUM_TESTS+1];
static char test_names[MAX_NUM_TESTS][MAX_NAME_LENGTH];

static int get_test_array_id(int id)
{
  int i = 0;

  registered_tests[num_registered_tests] = id;

  while(1) {
    if (registered_tests[i] == id)
      return i;
    ++i;
  }
}

static void register_failed_test(int id)
{
  int i;

  for (i = 0; i < num_failed_tests; ++i) {
    if (failed_tests[i] == id)
      return;
  }

  failed_tests[num_failed_tests++] = id;
}

void new_test_module(void) { num_failed_tests = 0; num_registered_tests = 0; }

void register_test(int id, char const *name)
{
  int test_idx;

  error_root(strlen(name) >= MAX_NAME_LENGTH, 1, "register_test [test_counter.c]",
        "test name must be shorter than %d chars", MAX_NAME_LENGTH);

  test_idx = get_test_array_id(id);

  error_root((test_idx == num_registered_tests) &&
            (num_registered_tests == MAX_NUM_TESTS),
        1, "register_test [test_counter.c]",
        "too many registered tests, max number of tests %d", MAX_NUM_TESTS);

  strcpy(test_names[test_idx], name);

  if (test_idx == num_registered_tests)
    ++num_registered_tests;
}

void fail_test(int id)
{
  int test_idx;

  test_idx = get_test_array_id(id);

  error_root(test_idx == num_registered_tests, 1, "fail_test [test_counter.c]",
      "test #%d hasn't been registered", id);

  register_failed_test(id);
}

void fail_test_if(int id, int check)
{
  if (check)
    fail_test(id);
}

void print_test_header(int id)
{
  int test_idx;

  test_idx = get_test_array_id(id);

  error_root(test_idx == num_registered_tests, 1, "print_test_header [test_counter.c]",
      "test #%d hasn't been registered", id);

  printf("Test #%d: %s\n\n", id, test_names[test_idx]);
}

void report_test_results(void)
{
  int i, test_idx;

  printf("Ran %d tests\n\n", num_registered_tests);

  if (num_failed_tests == 0) {
    printf(KGRN "ALL TESTS SUCCEEDED\n" KNRM);
    return;
  }

  printf("%d out of %d tests failed. Failed tests were:\n", num_failed_tests, num_registered_tests);

  for (i = 0; i < num_failed_tests; ++i) {
    test_idx = get_test_array_id(failed_tests[i]);
    printf("[" KRED "failed" KNRM "] #%d: %s\n", failed_tests[i], test_names[test_idx]);
  }
}
