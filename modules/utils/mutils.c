
/*******************************************************************************
 *
 * File mutils.c
 *
 * Copyright (C) 2005, 2007, 2008, 2011, 2013, 2016 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 * Utility functions used in main programs
 *
 * The externally accessible functions are
 *
 *   int find_opt(int argc,char *argv[],char const *opt)
 *     On process 0, this program compares the string opt with the arguments
 *     argv[1],..,argv[argc-1] and returns the position of the first argument
 *     that matches the string. If there is no matching argument, or if the
 *     program is called from another process, the return value is 0.
 *
 *   int fdigits(double x)
 *     Returns the smallest integer n such that the value of x printed with
 *     print format %.nf coincides with x up to a relative error at most a
 *     few times the machine precision DBL_EPSILON.
 *
 *   void check_dir(char const *dir)
 *     This program checks whether the directory dir is locally accessible,
 *     from each process, and aborts the main program with an informative
 *     error message if this is not the case. The program must be called
 *     simultaneously on all processes, but the argument may depend on the
 *     process.
 *
 *   void check_dir_root(char const *dir)
 *     On process 0, this program checks whether the directory dir is
 *     accessible and aborts the main program with an informative error
 *     message if this is not the case. When called on other processes,
 *     the program does nothing.
 *
 *   int name_size(char const *format, ...)
 *     On process 0, this program returns the length of the string that
 *     would be printed by calling sprintf(*,format,...). The format
 *     string can be any combination of literal text and the conversion
 *     specifiers %s, %d and %.nf (where n is a positive integer). When
 *     called on other processes, the program does nothing and returns
 *     the value of NAME_SIZE.
 *
 *   long find_section(char const *title)
 *     On process 0, this program scans stdin for a line starting with
 *     the string "[title]" (after any number of blanks). It terminates
 *     with an error message if no such line is found or if there are
 *     several of them. The program returns the offset of the line from
 *     the beginning of the file and positions the file pointer to the
 *     next line. On processes other than 0, the program does nothing
 *     and returns No_Section_Found.
 *
 *   long find_optional_section(char const *title)
 *     Same behaviour as find_section, however it does not throw an error if no
 *     section is found, rather, it returns No_Section_Found as the current
 *     location.
 *
 *   long read_line(char const *tag, char const *format,...)
 *     On process 0, this program reads a line of text and data from stdin
 *     in a controlled manner, as described in the notes below. The tag can
 *     be the empty string "" and must otherwise be an alpha-numeric word
 *     that starts with a letter. If it is not empty, the program searches
 *     for the tag in the current section. An error occurs if the tag is not
 *     found. The program returns the offset of the line from the beginning
 *     of the file and positions the file pointer to the next line. On
 *     processes other than 0, the program does nothing and returns
 *     No_Section_Found.
 *
 *   long read_optional_line(char const *tag, char const *format, ...)
 *     Same behaviour as read_line(), however you may specify a second argument
 *     for every argument that gives an optional value which will be used if the
 *     tag cannot be found.
 *
 *   int count_tokens(char const *tag)
 *     On process 0, this program finds and reads a line from stdin, exactly
 *     as read_line(tag,..) does, and returns the number of tokens found on
 *     that line after the tag. Tokens are separated by white space (blanks,
 *     tabs or newline characters) and comments (text beginning with #) are
 *     ignored. On exit, the file pointer is positioned at the next line. If
 *     called on other processes, the program does nothing and returns 0.
 *
 *   long read_iprms(char const *tag, int n, int *iprms)
 *     On process 0, this program finds and reads a line from stdin, exactly
 *     as read_line(tag,..) does, reads n integer values from that line after
 *     the tag and assigns them to the elements of the array iprms. An error
 *     occurs if less than n values are found on the line. The values must be
 *     separated by white space (blanks, tabs or newline characters). The
 *     program returns the offset of the line from the beginning of the file and
 *     positions the file pointer to the next line. When called on other
 *     processes, the program does nothing.
 *
 *   long read_optional_iprms(char const *tag, int n, int *iprms)
 *     Same behaviour as read_iprms, however it does not throw an error if no
 *     section is found, rather, it returns No_Section_Found as the current
 *     location.
 *
 *   long read_dprms(char const *tag, int n, double *dprms)
 *     On process 0, this program finds and reads a line from stdin, exactly
 *     as read_line(tag,..) does, reads n double values from that line after
 *     the tag and assigns them to the elements of the array dprms. An error
 *     occurs if less than n values are found on the line. The values must be
 *     separated by white space (blanks, tabs or newline characters). The
 *     program returns the offset of the line from the beginning of the file and
 *     positions the file pointer to the next line. When called on other
 *     processes, the program does nothing.
 *
 *   long read_optional_dprms(char const *tag, int n, double *dprms)
 *     Same behaviour as read_dprms, however it does not throw an error if no
 *     section is found, rather, it returns No_Section_Found as the current
 *     location.
 *
 *   void copy_file(char const *in, char const *out)
 *     Copies the file "in" to the file "out" in binary mode. An error occurs
 *     if the file copy is not successful.
 *
 * Notes:
 *
 * Except for check_dir(), the programs in this module do not involve any
 * communications and can be called locally.
 *
 * The programs find_section() and read_line() serve to read structured
 * input parameter files (such as the *.in in the directory main; see
 * main/README.infiles).
 *
 * Parameter lines that can be read by read_line() must be of the form
 *
 *   tag v1 v2 ...
 *
 * where v1,v2,... are data values (strings, integers or floating-point
 * numbers) separated by blanks. If the tag is empty, the first data value
 * may not be a string. Such lines are read by calling
 *
 *   read_line(tag,format,&var1,&var2,...)
 *
 * where var1,var2,... are the variables to which the values v1,v2,... are
 * to be assigned. The format string must include the associated sequence
 * of conversion specifiers %s, %d, %f or %lf without any modifiers. Other
 * tokens are not allowed in the format string, except for additional blanks
 * and a newline character at the end of the string (none of these have any
 * effect).
 *
 * The programs find_section() and read_line() ignore blank lines and any text
 * appearing after the character #. Lines longer than NAME_SIZE-1 characters are
 * not permitted. Each section may occur at most once and, within each section,
 * a line tag may not appear more than once. The number of characters written
 * to the target string variables is at most NAME_SIZE-1. Buffer overflows are
 * thus excluded if the target strings are of size NAME_SIZE or larger.
 *
 *******************************************************************************/

#define MUTILS_C
#define OPENQCD_INTERNAL

#include "global.h"
#include "mpi.h"
#include "utils.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

long const No_Section_Found = -1L;

static char text[512];
static char line[NAME_SIZE + 1];
static char inum[3 * sizeof(int) + 4];

#define scan_int_str "%d"
#define scan_double_str "%lf"
#define scan_char_str "%c"

#define _read_prms_impl(type, name, name_str)                                  \
  static long read_##name##_impl(char const *tag, int n, type *iprms,          \
                                 int optional)                                 \
  {                                                                            \
    int my_rank, nc, ic;                                                       \
    type buffer;                                                               \
    long loc;                                                                  \
    char *s;                                                                   \
                                                                               \
    loc = No_Section_Found;                                                    \
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);                                   \
                                                                               \
    if (my_rank == 0) {                                                        \
      check_tag(tag);                                                          \
                                                                               \
      if (tag[0] != '\0') {                                                    \
        loc = find_tag_impl(tag, optional);                                    \
                                                                               \
        if (loc == No_Section_Found) {                                         \
          return loc;                                                          \
        }                                                                      \
                                                                               \
        s = get_line();                                                        \
        s += strspn(s, " \t");                                                 \
        s += strcspn(s, " \t\n");                                              \
      } else {                                                                 \
        s = get_line();                                                        \
      }                                                                        \
                                                                               \
      s += strspn(s, " \t\n");                                                 \
      nc = 0;                                                                  \
                                                                               \
      while ((s[0] != '\0') && (nc < n)) {                                     \
        ic = sscanf(s, scan_##type##_str, &buffer);                            \
                                                                               \
        if (ic == 1) {                                                         \
          iprms[nc] = buffer;                                                  \
          nc += 1;                                                             \
          s += strcspn(s, " \t\n");                                            \
          s += strspn(s, " \t\n");                                             \
        } else {                                                               \
          break;                                                               \
        }                                                                      \
      }                                                                        \
                                                                               \
      error_root(nc != n, 1, "read_" name_str "_impl [mutils.c]",              \
                 "Incorrect read count");                                      \
    }                                                                          \
                                                                               \
    return loc;                                                                \
  }

int find_opt(int argc, char *argv[], char const *opt)
{
  int my_rank, k;

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    for (k = 1; k < argc; k++) {
      if (strcmp(argv[k], opt) == 0) {
        return k;
      }
    }
  }

  return 0;
}

int fdigits(double x)
{
  int m, n, ne, k;
  double y, z;

  if (is_equal_d(x, 0.0)) {
    return 0;
  }

  y = fabs(x);
  z = DBL_EPSILON * y;
  m = floor(log10(y + z));
  n = 0;
  ne = 1;

  for (k = 0; k < (DBL_DIG - m); k++) {
    z = sqrt((double)(ne)) * DBL_EPSILON * y;

    if (((y - floor(y)) <= z) || ((ceil(y) - y) <= z)) {
      break;
    }

    y *= 10.0;
    ne += 1;
    n += 1;
  }

  return n;
}

void check_dir(char const *dir)
{
  int my_rank, nc, n;
  char *tmp_file;
  FILE *tmp;

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  nc = strlen(dir);
  tmp_file = malloc((nc + 7 + 3 * sizeof(int)) * sizeof(char));
  error(tmp_file == NULL, 1, "check_dir [mutils.c]",
        "Unable to allocate name string");
  sprintf(tmp_file, "%s/.tmp_%d", dir, my_rank);

  n = 0;
  tmp = fopen(tmp_file, "rb");

  if (tmp == NULL) {
    n = 1;
    tmp = fopen(tmp_file, "wb");
  }

  nc = sprintf(text, "Unable to access directory ");
  strncpy(text + nc, dir, 512 - nc);
  text[511] = '\0';
  error_loc(tmp == NULL, 1, "check_dir [mutils.c]", text);
  fclose(tmp);

  if (n == 1) {
    remove(tmp_file);
  }
  free(tmp_file);
}

void check_dir_root(char const *dir)
{
  int my_rank, nc, n;
  char *tmp_file;
  FILE *tmp;

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    nc = strlen(dir);
    tmp_file = malloc((nc + 6) * sizeof(char));
    error_root(tmp_file == NULL, 1, "check_dir_root [mutils.c]",
               "Unable to allocate name string");
    sprintf(tmp_file, "%s/.tmp", dir);

    n = 0;
    tmp = fopen(tmp_file, "rb");

    if (tmp == NULL) {
      n = 1;
      tmp = fopen(tmp_file, "wb");
    }

    error_root(tmp == NULL, 1, "check_dir_root [mutils.c]",
               "Unable to access directory %s from process 0", dir);

    fclose(tmp);
    if (n == 1) {
      remove(tmp_file);
    }
    free(tmp_file);
  }
}

int name_size(char const *format, ...)
{
  int my_rank, nlen, ie, n;
  double dmy;
  char const *pp, *pc;
  va_list args;

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    va_start(args, format);
    pc = format;
    nlen = strlen(format);
    ie = 0;
    n = 0;

    for (;;) {
      pp = strchr(pc, '%');

      if (pp == NULL) {
        break;
      }

      pc = pp + 1;

      if (pc[0] == 's') {
        nlen += (strlen(va_arg(args, char *)) - 2);
      } else if (pc[0] == 'd') {
        sprintf(inum, "%d", va_arg(args, int));
        nlen += (strlen(inum) - 2);
      } else if (pc[0] == '.') {
        if (sscanf(pc, ".%d", &n) != 1) {
          ie = 1;
          break;
        }

        sprintf(inum, ".%df", n);
        pp = strstr(pc, inum);

        if (pp != pc) {
          ie = 2;
          break;
        }

        nlen += (n + 1 - strlen(inum));
        dmy = va_arg(args, double);
        if (dmy < 0.0) {
          nlen += 1;
        }
      } else {
        ie = 3;
        break;
      }
    }

    va_end(args);
    error_root(ie != 0, 1, "name_size [mutils.c]",
               "Incorrect format string %s (ie=%d)", format, ie);
    return nlen;
  }

  return NAME_SIZE;
}

static int cmp_text(char const *text1, char const *text2)
{
  size_t n1, n2;
  char const *p1, *p2;

  p1 = text1;
  p2 = text2;

  while (1) {
    p1 += strspn(p1, " \t\n");
    p2 += strspn(p2, " \t\n");
    n1 = strcspn(p1, " \t\n");
    n2 = strcspn(p2, " \t\n");

    if (n1 != n2) {
      return 0;
    }
    if (n1 == 0) {
      return 1;
    }
    if (strncmp(p1, p2, n1) != 0) {
      return 0;
    }

    p1 += n1;
    p2 += n1;
  }
}

static char *get_line(void)
{
  char *s, *c;

  s = fgets(line, NAME_SIZE + 1, stdin);

  if (s != NULL) {
    error_root(strlen(line) == NAME_SIZE, 1, "get_line [mutils.c]",
               "Input line is longer than NAME_SIZE-1");

    c = strchr(line, '#');
    if (c != NULL) {
      c[0] = '\0';
    }
  }

  return s;
}

static long find_section_impl(char const *title, int optional)
{
  int my_rank, ie;
  long ofs, sofs;
  char *s, *pl, *pr;

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    rewind(stdin);
    sofs = No_Section_Found;
    ofs = ftell(stdin);
    s = get_line();

    while (s != NULL) {
      pl = strchr(line, '[');
      pr = strchr(line, ']');

      if ((pl == (line + strspn(line, " \t"))) && (pr > pl)) {
        pl += 1;
        pr[0] = '\0';

        if (cmp_text(pl, title) == 1) {
          error_root(sofs >= 0L, 1, "find_section [mutils.c]",
                     "Section [%s] occurs more than once", title);
          sofs = ofs;
        }
      }

      ofs = ftell(stdin);
      s = get_line();
    }

    error_root(!optional && (sofs == No_Section_Found), 1,
               "find_section [mutils.c]", "Section [%s] not found", title);

    if (sofs != No_Section_Found) {
      ie = fseek(stdin, sofs, SEEK_SET);
      error_root(ie != 0, 1, "find_section [mutils.c]",
                 "Unable to go to section [%s]", title);
      get_line();
    } else {
      fseek(stdin, ofs, SEEK_SET);
    }

    return sofs;
  } else {
    return No_Section_Found;
  }
}

long find_section(char const *title)
{
  return find_section_impl(title, 0);
}

long find_optional_section(char const *title)
{
  return find_section_impl(title, 1);
}

static void check_tag(char const *tag)
{
  if (tag[0] == '\0') {
    return;
  }

  error_root((strspn(tag, " 0123456789.") != 0L) ||
                 (strcspn(tag, " \n") != strlen(tag)),
             1, "check_tag [mutils.c]", "Improper tag %s", tag);
}

static long find_tag_impl(char const *tag, int optional)
{
  int ie;
  long tofs, lofs, ofs;
  char *s, *pl, *pr;

  ie = 0;
  tofs = No_Section_Found;
  lofs = ftell(stdin);
  rewind(stdin);
  ofs = ftell(stdin);
  s = get_line();

  while (s != NULL) {
    pl = strchr(line, '[');
    pr = strchr(line, ']');

    if ((pl == (line + strspn(line, " \t"))) && (pr > pl)) {
      if (ofs < lofs) {
        ie = 0;
        tofs = No_Section_Found;
      } else {
        break;
      }
    } else {
      pl = line + strspn(line, " \t");
      pr = pl + strcspn(pl, " \t\n");
      pr[0] = '\0';

      if (strcmp(pl, tag) == 0) {
        if (tofs != No_Section_Found) {
          ie = 1;
        }
        tofs = ofs;
      }
    }

    ofs = ftell(stdin);
    s = get_line();
  }

  error_root(!optional && (tofs == No_Section_Found), 1, "find_tag [mutils.c]",
             "Tag %s not found", tag);

  error_root(ie != 0, 1, "find_tag [mutils.c]",
             "Tag %s occurs more than once in the current section", tag);

  if (tofs != No_Section_Found) {
    ie = fseek(stdin, tofs, SEEK_SET);
    error_root(ie != 0, 1, "find_tag [mutils.c]",
               "Unable to go to line with tag %s", tag);
  } else {
    ie = fseek(stdin, lofs, SEEK_SET);
  }

  return tofs;
}

static long find_tag(char const *tag)
{
  return find_tag_impl(tag, 0);
}

static long read_line_impl(int optional, char const *tag, char const *format,
                           va_list args)
{
  int is, ic, use_optional;
  long tofs;
  char const *pl, *p;
  char *str_src;

  check_tag(tag);

  use_optional = 0;

  if (tag[0] != '\0') {
    tofs = find_tag_impl(tag, optional);

    if (tofs == No_Section_Found) {
      use_optional = 1;
      pl = NULL;
    } else {
      get_line();
      pl = line + strspn(line, " \t");
      pl += strcspn(pl, " \t\n");
    }
  } else {
    p = format;
    p += strspn(p, " ");
    error_root(strstr(p, "%s") == p, 1, "read_line [mutils.c]",
               "String data after empty tag");
    tofs = ftell(stdin);
    pl = get_line();
  }

  for (p = format;;) {
    p += strspn(p, " ");
    ic = 0;
    is = 2;

    if ((p[0] == '\0') || (p[0] == '\n')) {
      break;
    } else if (p == strstr(p, "%s")) {
      if (use_optional == 0) {
        ic = sscanf(pl, "%s", va_arg(args, char *));
      } else {
        str_src = va_arg(args, char *);
        strcpy(str_src, va_arg(args, char *));
      }
    } else if (p == strstr(p, "%d")) {
      if (use_optional == 0) {
        ic = sscanf(pl, "%d", va_arg(args, int *));
      } else {
        (*va_arg(args, int *)) = va_arg(args, int);
      }
    } else if (p == strstr(p, "%f")) {
      if (use_optional == 0) {
        ic = sscanf(pl, "%f", va_arg(args, float *));
      } else {
        (*va_arg(args, float *)) = (float)va_arg(args, double);
      }
    } else if (p == strstr(p, "%lf")) {
      is = 3;
      if (use_optional == 0) {
        ic = sscanf(pl, "%lf", va_arg(args, double *));
      } else {
        (*va_arg(args, double *)) = va_arg(args, double);
      }
    } else {
      error_root(1, 1, "read_line [mutils.c]",
                 "Incorrect format string %s on line with tag %s", format, tag);
    }

    if (use_optional == 0) {
      error_root(ic != 1, 1, "read_line [mutils.c]",
                 "Missing data item(s) on line with tag %s", tag);
    }

    p += is;
    if (use_optional == 0) {
      pl += strspn(pl, " \t");
      pl += strcspn(pl, " \t\n");
    }
  }

  return tofs;
}

long read_line(char const *tag, char const *format, ...)
{
  int my_rank;
  long tofs;
  va_list args;

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    va_start(args, format);
    tofs = read_line_impl(0, tag, format, args);
    va_end(args);

    return tofs;
  } else {
    return No_Section_Found;
  }
}

long read_optional_line(char const *tag, char const *format, ...)
{
  int my_rank;
  long tofs;
  va_list args;

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    va_start(args, format);
    tofs = read_line_impl(1, tag, format, args);
    va_end(args);

    return tofs;
  } else {
    return No_Section_Found;
  }
}

int count_tokens(char const *tag)
{
  int my_rank, n;
  char *s;

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    check_tag(tag);

    if (tag[0] != '\0') {
      find_tag(tag);
      s = get_line();
      s += strspn(s, " \t");
      s += strcspn(s, " \t\n");
    } else {
      s = get_line();
    }

    s += strspn(s, " \t\n");
    n = 0;

    while (s[0] != '\0') {
      n += 1;
      s += strcspn(s, " \t\n");
      s += strspn(s, " \t\n");
    }

    return n;
  } else {
    return 0;
  }
}

_read_prms_impl(int, iprms, "iprms")

long read_iprms(char const *tag, int n, int *iprms)
{
  return read_iprms_impl(tag, n, iprms, 0);
}

long read_optional_iprms(char const *tag, int n, int *iprms)
{
  return read_iprms_impl(tag, n, iprms, 1);
}

_read_prms_impl(double, dprms, "dprms")

long read_dprms(char const *tag, int n, double *dprms)
{
return read_dprms_impl(tag, n, dprms, 0);
}

long read_optional_dprms(char const *tag, int n, double *dprms)
{
  return read_dprms_impl(tag, n, dprms, 1);
}

_read_prms_impl(char, cprms, "cprms")

long read_cprms(char const *tag, int n, char *cprms)
{
  return read_cprms_impl(tag, n, cprms, 0);
}

long read_optional_cprms(char const *tag, int n, char *cprms)
{
  return read_cprms_impl(tag, n, cprms, 1);
}

void copy_file(char const *in, char const *out)
{
  int c;
  FILE *fin, *fout;

  fin = fopen(in, "rb");
  error_loc(fin == NULL, 1, "copy_file [mutils.c]",
            "Unable to open input file");

  fout = fopen(out, "wb");
  error_loc(fout == NULL, 1, "copy_file [mutils.c]",
            "Unable to open output file");

  c = getc(fin);

  while (feof(fin) == 0) {
    putc(c, fout);
    c = getc(fin);
  }

  if ((ferror(fin) == 0) && (ferror(fout) == 0)) {
    fclose(fin);
    fclose(fout);
  } else {
    error_loc(1, 1, "copy_file [mutils.c]", "Read or write error");
  }
}
