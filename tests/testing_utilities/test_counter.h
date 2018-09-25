#ifndef TEST_COUNTER_H
#define TEST_COUNTER_H

#include "utils.h"

extern void new_test_module(void);
extern void register_test(int id, char const *name);
extern void fail_test(int id);
extern void fail_test_if(int id, int check);
extern void print_test_header(int id);
extern void report_test_results(void);

#endif /* TEST_COUNTER_H */
