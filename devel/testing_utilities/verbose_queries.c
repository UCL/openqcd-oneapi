
/*
 * Created: 16-02-2018
 * Modified:
 * Author: Jonas R. Glesaaen (jonas@glesaaen.com)
 */

#include "verbose_queries.h"
#include "printing_macros.h"
#include "test_counter.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

char const *query_t_to_char(query_t qflag)
{
  static char text_buffer[256];

  switch (qflag) {
  case (U_MATCH_UD):
    sprintf(text_buffer, "%-20s", "(U_MATCH_UD)");
    break;
  case (UDBUF_UP2DATE):
    sprintf(text_buffer, "%-20s", "(UDBUF_UP2DATE)");
    break;
  case (BSTAP_UP2DATE):
    sprintf(text_buffer, "%-20s", "(BSTAP_UP2DATE)");
    break;
  case (FTS_UP2DATE):
    sprintf(text_buffer, "%-20s", "(FTS_UP2DATE)");
    break;
  case (UBGR_MATCH_UD):
    sprintf(text_buffer, "%-20s", "(UBGR_MATCH_UD)");
    break;
  case (UDBGR_MATCH_UD):
    sprintf(text_buffer, "%-20s", "(UDBGR_MATCH_UD)");
    break;
  case (SW_UP2DATE):
    sprintf(text_buffer, "%-20s", "(SW_UP2DATE)");
    break;
  case (SW_E_INVERTED):
    sprintf(text_buffer, "%-20s", "(SW_E_INVERTED)");
    break;
  case (SW_O_INVERTED):
    sprintf(text_buffer, "%-20s", "(SW_O_INVERTED)");
    break;
  case (SWD_UP2DATE):
    sprintf(text_buffer, "%-20s", "(SWD_UP2DATE)");
    break;
  case (SWD_E_INVERTED):
    sprintf(text_buffer, "%-20s", "(SWD_E_INVERTED)");
    break;
  case (SWD_O_INVERTED):
    sprintf(text_buffer, "%-20s", "(SWD_O_INVERTED)");
    break;
  case (AW_UP2DATE):
    sprintf(text_buffer, "%-20s", "(AW_UP2DATE)");
    break;
  case (AWHAT_UP2DATE):
    sprintf(text_buffer, "%-20s", "(AWHAT_UP2DATE)");
    break;
  case (UD_IS_CLEAN):
    sprintf(text_buffer, "%-20s", "(UD_IS_CLEAN)");
    break;
  case (UD_IS_SMEARED):
    sprintf(text_buffer, "%-20s", "(UD_IS_SMEARED)");
    break;
  case (SMEARED_UD_UP2DATE):
    sprintf(text_buffer, "%-20s", "(SMEARED_UD_UP2DATE)");
    break;
  case (UD_PHASE_SET):
    sprintf(text_buffer, "%-20s", "(UD_PHASE_SET)");
    break;
  case (QUERIES):
    sprintf(text_buffer, "%-20s", "(QUERIES)");
  }

  return text_buffer;
}

char const *bool_to_char(int bool_like)
{
  static char text_buffer[256];

  if (bool_like)
    strcpy(text_buffer, "true ");
  else
    strcpy(text_buffer, "false");

  return text_buffer;
}

char const *bool_to_char_colour(int bool_like, int expected)
{
  static char text_buffer[256];

  if (bool_like == expected) {
    if (bool_like)
      strcpy(text_buffer, KGRN "true " KNRM);
    else
      strcpy(text_buffer, KGRN "false" KNRM);
  } else {
    if (bool_like)
      strcpy(text_buffer, KRED "true " KNRM);
    else
      strcpy(text_buffer, KRED "false" KNRM);
  }

  return text_buffer;
}

void test_flag_verbose(int test_id, query_t qflag, int expected, int is_verbose)
{
  int query_result = query_flags(qflag);

  if (is_verbose || (query_result != expected)) {
    printf("query_flags%s  = %s    [expected %s]\n", query_t_to_char(qflag),
           bool_to_char(query_result),
           bool_to_char_colour(expected, query_result));
  }

  fail_test_if(test_id, query_result != expected);
}
