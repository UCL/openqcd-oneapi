#ifndef VERBOSE_QUERIES_H
#define VERBOSE_QUERIES_H

#include "flags.h"

char const *query_t_to_char(query_t qflag);
char const *bool_to_char(int bool_like);
char const *bool_to_char_colour(int bool_like, int expected);
void test_flag_verbose(int test_id, query_t qflag, int expected,
                       int is_verbose);

#endif /* VERBOSE_QUERIES_H */
