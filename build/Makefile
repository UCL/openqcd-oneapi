# User environment variables
# ------------------------------------------------------------------------------
# {{{

# }}}
#
# Code environment variables
# ------------------------------------------------------------------------------
# {{{

target_module_dir := modules
target_devel_dir  := devel
dep_dir           := .deps
source_main_dir    = $(code_dir)/main
source_module_dir  = $(code_dir)/modules
source_devel_dir   = $(code_dir)/devel

# }}}
#
# Variables read from compile_settings file
# ------------------------------------------------------------------------------
# {{{

settings_file := compile_settings.txt

# Function to load a parameter from the settings file
load-par-from-file = $(shell grep $(1) $(settings_file)| grep -v '\#' | awk '{print$$2;}')

NPROC0      := $(call load-par-from-file, NPROC0_TOT )
NPROC1      := $(call load-par-from-file, NPROC1_TOT )
NPROC2      := $(call load-par-from-file, NPROC2_TOT )
NPROC3      := $(call load-par-from-file, NPROC3_TOT )
L0          := $(call load-par-from-file, L0 )
L1          := $(call load-par-from-file, L1 )
L2          := $(call load-par-from-file, L2 )
L3          := $(call load-par-from-file, L3 )
NPROC0_BLK  := $(call load-par-from-file, NPROC0_BLK )
NPROC1_BLK  := $(call load-par-from-file, NPROC1_BLK )
NPROC2_BLK  := $(call load-par-from-file, NPROC2_BLK )
NPROC3_BLK  := $(call load-par-from-file, NPROC3_BLK )

code_dir    := $(shell grep CODELOC     $(settings_file)| grep -v '\#' | awk '{print$$2;}')
CC          := $(shell grep COMPILER    $(settings_file)| grep -v '\#' | awk '{print$$2;}')
MPI_INCLUDE := $(shell grep MPI_INCLUDE $(settings_file)| grep -v '\#' | awk '{print$$2;}')
USR_CFLAGS  := $(shell grep CFLAGS      $(settings_file)| grep -v '\#' | awk '{for (i=2; i<=NF; i++) print FS$$i;}')
USR_LDFLAGS := $(shell grep LDFLAGS     $(settings_file)| grep -v '\#' | awk '{for (i=2; i<=NF; i++) print FS$$i;}')

# }}}
#
# Compile flags
# ------------------------------------------------------------------------------
# {{{

SIZE_FLAGS := -DNPROC0=$(NPROC0) -DNPROC1=$(NPROC1)\
              -DNPROC2=$(NPROC2) -DNPROC3=$(NPROC3)\
              -DL0=$(L0) -DL1=$(L1) -DL2=$(L2) -DL3=$(L3)\
              -DNPROC0_BLK=$(NPROC0_BLK) -DNPROC1_BLK=$(NPROC1_BLK)\
              -DNPROC2_BLK=$(NPROC2_BLK) -DNPROC3_BLK=$(NPROC3_BLK)

CORRECTNESS := -pedantic -fstrict-aliasing -Wno-long-long -Wstrict-prototypes

CFLAGS     += -I$(code_dir)/include -I$(code_dir) -isystem $(MPI_INCLUDE)\
              $(SIZE_FLAGS) $(USR_CFLAGS) $(CORRECTNESS)

LDFLAGS    += $(USR_LDFLAGS) -lm

# The nompi targets needs one more include directory
$(target_devel_dir)/nompi/%: \
	CFLAGS := -I$(code_dir)/include/nompi $(CFLAGS)

$(target_module_dir)/nompi/%: \
	CFLAGS := -I$(code_dir)/include/nompi $(filter-out -I$(code_dir)/include,$(CFLAGS))

$(dep_dir)/nompi/%: \
	CFLAGS := -I$(code_dir)/include/nompi $(filter-out -I$(code_dir)/include,$(CFLAGS))

# }}}
#
# Module names and program names
# ------------------------------------------------------------------------------
# {{{
# List of all the modules in openQCD as well as all the tests.
#
# module_names: list of all modules in openQCD
# module_libs: the target static library names
# main_sources: all main executable source files
# main_programs: the corresponding target executables
# test_module_names: list of all testing modules in the devel/ folder
# test_module_dirs: the corresponding target "directories", as all tests are
#                   executable we set the targets as the folder containing them

# All the modules in openQCD
module_names := archive block dfl dirac field_com flags forces lattice linalg\
                linsolv little mdflds random ratfcts sap sflds stout_smearing\
                su3fcts sw_term tcharge uflds update utils vflds wflow

module_libs  := $(patsubst %,$(target_module_dir)/%.a,$(module_names))

# Collect all the possible simulation programs
main_sources  := $(wildcard $(source_main_dir)/*.c)
main_programs := $(patsubst $(source_main_dir)/%.c,%,$(main_sources))

# Collect all the tests
test_module_names := archive block dfl dirac field_com flags forces lattice linalg little\
                     random ratfcts sap sflds stout_smearing su3fcts sw_term\
                     tcharge uflds update utils vflds wflow

test_module_dirs  := $(patsubst %,$(target_devel_dir)/%,$(test_module_names))

## nompi tests and modules
## -----------------------------------------------------------------------------
## {{{
## The "nompi" modules and corresonding tests. The nompi libraries are only used
## for tests, and will not be compiled when making the main executables.
##
## nompi_module_names: list of all the moduels in the nompi folder
## nompi_module_libs: list of all target static libraries necessary to build a
##                    nompi test
## test_nompi_module_names: list of test modules in devel/nompi
## test_nompi_module_dirs: the corresponding target test folders

# And the nompi modules and tests
nompi_module_names := extras utils

# Create a list of modules without the nompi-libs
# This will be the set $(module_names)/$(nompi_module_names)
module_names_nompi_filter := $(module_names)
$(foreach elem,$(nompi_module_names),\
	$(eval module_names_nompi_filter := \
		$(filter-out $(elem),$(module_names_nompi_filter))))

# List of all the libs used to compile a nompi program
nompi_module_libs  := \
	$(patsubst %, $(target_module_dir)/nompi/%.a,$(nompi_module_names))\
	$(patsubst %,$(target_module_dir)/%.a,$(module_names_nompi_filter))

# List of the nompi test folders
test_nompi_module_names := extras forces linalg main random ratfcts su3fcts\
                           sw_term utils

test_nompi_module_dirs  := $(patsubst %,$(target_devel_dir)/nompi/%,$(test_nompi_module_names))

## }}}
# }}}
#
# Targets
# ------------------------------------------------------------------------------
# {{{

# Compile every main program
.PHONY: all
all: print-header $(main_programs)

# Compile every test in devel
.PHONY: tests
tests: print-header $(test_module_dirs) $(test_nompi_module_dirs)

# The main programs depend on all the modules
$(main_programs): print-header $(module_libs)

# Can build the modules independently if you wish
.PHONY: $(module_names)
$(module_names): %: $(target_module_dir)/%.a

# The nompi tests can also be built by itself
.PHONY: $(target_devel_dir)/nompi
$(target_devel_dir)/nompi: $(test_nompi_module_dirs)

# Targets for the individual test modules
.PHONY: print-header $(test_module_dirs)
$(test_module_dirs):

# Shows all the available targets
.PHONY: help
help:
	@echo "Build script for the openQCD software suite.\n"
	@echo "For information on how the $(settings_file) works,"
	@echo "please see the README file in the code.\n"
	@echo "Available targets:"
	@echo "  $(call header-print,all): build all openQCD executables"
	@echo "  $(call header-print,qcd1),$(call header-print,...): build any individual openQCD executable"
	@echo "  $(call header-print,[module]): make a static library of any individual module"
	@echo "  $(call header-print,tests): build all tests in the devel directory"
	@echo "  $(call header-print,devel/...): build a subset of tests"
	@echo "  $(call header-print,clean): delete all files generated by make"
	@echo "  $(call header-print,list-targets): list of all (main) targets"
	@echo "  $(call header-print,list-all-targets): list of all possible targets"

.PHONY: nop
.PHONY: list-targets
list-targets:
	@make --print-data-base --question nop |            \
  awk '/^[^.%][-A-Za-z0-9_]*:/                        \
         { print substr($$1, 1, length($$1)-1) }' |   \
  sort |                                              \
  pr --omit-pagination --width=80 --columns=4

.PHONY: list-all-targets
list-all-targets:
	@make --print-data-base --question nop |            \
  awk '/^[^.%][\/-A-Za-z0-9_]*:/                      \
         { print substr($$1, 1, length($$1)-1) }' |   \
  sort |                                              \
	pr --omit-pagination --width=120 --columns=2

# Print a header with the current lattice size
.PHONY: print-header
print-header:
	@echo "$(HEADER_COLOR)Compiling openQCD with lattice dimensions:"
	@echo "L:     $(strip $(L0))x$(strip $(L1))x$(strip $(L2))x$(strip $(L3))"
	@echo "NPROC: $(strip $(NPROC0))x$(strip $(NPROC1))x$(strip $(NPROC2))x$(strip $(NPROC3))"
	@echo "NBLK:  $(strip $(NPROC0_BLK))x$(strip $(NPROC1_BLK))x$(strip $(NPROC2_BLK))x$(strip $(NPROC3_BLK))$(NO_COLOR)"
	@echo

# Delete make-generated files
.PHONY: clean
clean: clean-modules clean-main clean-tests
	@rm -rf $(dep_dir)

.PHONY: clean-tests
clean-tests:
	@echo "Cleaning tests..."
	@rm -rf $(target_devel_dir)

.PHONY: clean-main
clean-main:
	@echo "Cleaning executables..."
	@rm -rf $(main_programs)

.PHONY: clean-modules
clean-modules:
	@echo "Cleaning modules..."
	@rm -rf $(target_module_dir)

# }}}
#
# Dependencies
# ------------------------------------------------------------------------------
# {{{

# Temporary "thin" static libraries
tmp_mod_lib       := $(target_module_dir)/module_library.a
tmp_nompi_mod_lib := $(target_module_dir)/nompi_module_library.a

# Make it so that a module depends on the .c files in it
# Also add the .d files to the .SECONDARY so that they aren't cleaned up by make
define expand-module-deps
.SECONDARY: \
	$(patsubst $(source_module_dir)/%.c,$(dep_dir)/%.d,$(wildcard $(source_module_dir)/$1/*.c))

$(target_module_dir)/$1.a: \
	$(patsubst $(source_module_dir)/%.c,$(target_module_dir)/%.o,$(wildcard $(source_module_dir)/$1/*.c))
endef

# Define dependencies for the test folders
# These also depend on any available .in file which will be copied over
define expand-devel-deps
$(target_devel_dir)/$1: \
	$(patsubst $(source_devel_dir)/%.c,$(target_devel_dir)/%,$(wildcard $(source_devel_dir)/$1/*.c)) \
	$(patsubst $(source_devel_dir)/%.in,$(target_devel_dir)/%.in,$(wildcard $(source_devel_dir)/$1/*.in))
endef

# Include all .d files for each of the .c files in a single module
define include-module-deps
-include $(patsubst $(source_module_dir)/%.c,$(dep_dir)/%.d,$(wildcard $(source_module_dir)/$1/*.c))
endef

# Add the appropriate dependencies
$(foreach mod,$(module_names),$(eval $(call expand-module-deps,$(mod))))
$(foreach mod,$(patsubst %,nompi/%,$(nompi_module_names)),$(eval $(call expand-module-deps,$(mod))))
$(foreach dev,$(test_module_names),$(eval $(call expand-devel-deps,$(dev))))
$(foreach dev,$(patsubst %,nompi/%,$(test_nompi_module_names)),$(eval $(call expand-devel-deps,$(dev))))

# No need to fetch the dependencies if we are building a PHONY target
nop_targets := clean clean-tests clean-main clean-modules help list-targets list-all-targets nop
ifeq ("$(findstring $(MAKECMDGOALS),"$(nop_targets)")","")
compiled_modules := $(module_names) $(patsubst %,nompi/%,$(nompi_module_names))
compiled_mains   := $(main_programs)
endif

# Include the gcc generated dependency files
# We do not generate ones for the tests as they should be generated correctly
-include $(patsubst %,$(dep_dir)/%.d,$(compiled_mains))
$(foreach mod,$(compiled_modules),$(eval $(call include-module-deps,$(mod))))

## Optimisation dependencies
## -----------------------------------------------------------------------------
## {{{

# The AVX512 instructions are stored in subfolders of their respective modules,
# these needs to be added if avx512 is used
ifneq (,$(findstring AVX512,$(CFLAGS)))

define expand-avx512-module-deps
.SECONDARY: \
	$(patsubst $(source_module_dir)/%.c,$(dep_dir)/%.d,$(wildcard $(source_module_dir)/$1/avx512/*.c))

$(target_module_dir)/$1.a: \
	$(patsubst $(source_module_dir)/%.c,$(target_module_dir)/%.o,$(wildcard $(source_module_dir)/$1/avx512/*.c))
endef

$(foreach mod,$(module_names),$(eval $(call expand-avx512-module-deps,$(mod))))
endif

# The AVX512_ASM instructions are stored as .s files, and they do not
# necessariy have the same name as their .c counterparts
ifneq (,$(findstring AVX512_ASM,$(CFLAGS)))

define expand-avx512-asm-module-deps
$(target_module_dir)/$1.a: \
	$(patsubst $(source_module_dir)/%.s,$(target_module_dir)/%.o,$(wildcard $(source_module_dir)/$1/avx512/*.c))
endef

$(foreach mod,$(module_names),$(eval $(call expand-avx512-asm-module-deps,$(mod))))
endif

## }}}
##
## Extra dependencies
## -----------------------------------------------------------------------------
## {{{

$(target_devel_dir)/stout_smearing: | $(target_devel_dir)/stout_smearing/configurations

$(target_module_dir)/utils.a: $(target_module_dir)/utils/version.o

$(test_nompi_module_dirs): $(nompi_module_libs)

# The version.c should always be rebuilt to have up to date information
.PHONY: $(target_module_dir)/utils/version.c

## }}}
# }}}
#
# Pattern rules
# ------------------------------------------------------------------------------
# {{{

# Create a thin archive of all the modules
.DELETE_ON_ERROR: $(tmp_mod_lib)
$(tmp_mod_lib): $(module_libs)
	@rm -f $@
	@ar rcsT $@ $^

# Thin archive for the nompi modules
.DELETE_ON_ERROR: $(tmp_nompi_mod_lib)
$(tmp_nompi_mod_lib): $(nompi_module_libs)
	@rm -f $@
	@ar rcsT $@ $^

# Compile a simulation program
$(main_programs): %: $(source_main_dir)/%.c $(tmp_mod_lib)
	$(mkdir-target)
	$(call link-and-print,\
		$(CC) $(CFLAGS) $< -o $@ $(tmp_mod_lib) $(LDFLAGS),\
		$(notdir $<))

# Compile a test
$(target_devel_dir)/%: $(source_devel_dir)/%.c $(tmp_mod_lib)
	$(mkdir-target)
	$(call link-and-print,\
		$(CC) $(CFLAGS) $< -o $@ $(tmp_mod_lib) $(LDFLAGS),\
		$@.c)

# Compile a nompi test
$(target_devel_dir)/nompi/%: $(source_devel_dir)/nompi/%.c $(tmp_nompi_mod_lib)
	$(mkdir-target)
	$(call link-and-print,\
		$(CC) $(CFLAGS) $< -o $@ $(tmp_nompi_mod_lib) $(LDFLAGS),\
		$@.c)

# Building the archives
$(target_module_dir)/%.a:
	$(mkdir-target)
	$(call link-and-print,ar rcs $@ $^,$(notdir $@))

# This rule has to come before the %.c -> %.o one so that it takes presedence
# in the case that we enable the ASM
ifneq (,$(findstring AVX512_ASM,$(CFLAGS)))
# Compiling an assembly file
$(target_module_dir)/%.o: $(source_module_dir)/%.s $(settings_file) Makefile
	$(mkdir-target)
	$(call compile-and-print,\
		$(CC) $(CFLAGS) -c $< -o $@,\
		"modules/$(subst $(source_module_dir)/,,$<)")
endif

# Compiling a single source file
$(target_module_dir)/%.o: $(source_module_dir)/%.c $(settings_file) Makefile
	$(mkdir-target)
	$(call compile-and-print,\
		$(CC) $(CFLAGS) -c $< -o $@,\
		"modules/$(subst $(source_module_dir)/,,$<)")


# Creating a dependency file from a source file
$(dep_dir)/%.d: $(source_module_dir)/%.c
	$(mkdir-target)
	@$(CC) \
		$(CFLAGS) -MM -MP \
		-MT $(patsubst $(dep_dir)/%.d,$(target_module_dir)/%.o,$@) \
		-MF $@ $<

# Creating a dependency file from a program source file
$(dep_dir)/%.d: $(source_main_dir)/%.c
	$(mkdir-target)
	@$(CC) \
		$(CFLAGS) -MM -MP \
		-MT $(patsubst $(dep_dir)/%.d,%,$@) -MF \
		$@ $<

# Copy the test infiles
$(target_devel_dir)/%.in: $(source_devel_dir)/%.in
	$(mkdir-target)
	@cp -f $< $@

## Specialised targets
## -----------------------------------------------------------------------------
## {{{

# Link the configs for the stout_smearing tests
$(target_devel_dir)/stout_smearing/configurations: \
		$(source_devel_dir)/stout_smearing/configurations
	${mkdir-target}
	@rm -f $@
	@ln -s $$(readlink -e $<) $@

# Compile version source file
$(target_module_dir)/utils/version.o: $(target_module_dir)/utils/version.c $(settings_file)
	$(mkdir-target)
	$(call compile-and-print,$(CC) $(CFLAGS) -c $< -o $@,"modules/$(notdir $<)")

# Fill the version.c file with current build information
$(target_module_dir)/utils/version.c: $(settings_file) Makefile
	@echo "#include \"version.h\"" > $@
	$(eval git_version := $(call git-version))
	@echo "const char * build_git_sha = \"$(git_version)\";" >> $@
	@current_date_time=$$(date "+%d-%m-%Y %H:%M:%S");\
	echo "const char * build_date = \"$${current_date_time}\";" >> $@
	$(eval sanitised_cflags := $(call sanitize-string,$(USR_CFLAGS)))
	@echo "const char * build_user_cflags = \"$(sanitised_cflags)\";" >> $@

## }}}
# }}}
#
# Make functions
# ------------------------------------------------------------------------------
# {{{

# Create the directory of the target
define mkdir-target
	@mkdir -p $(dir $@)
endef

# The shell if statement
define shell-if
	if $1; then $2; fi
endef

# Run a command only if it exists
define if-command-exists
	$(call shell-if,hash $1 2>/dev/null,$2)
endef

# Run a git command relative to $(code_dir)
define relative-git
	git --git-dir=$(code_dir)/.git --work-tree=$(code_dir) $1
endef

# Get the git version of the $(code_dir)
# Returns "unknown" if either $(code_dir) isn't a git repository or if git
# doesn't exist on the system.
define git-version
	$(shell \
		if hash git 2> /dev/null; then \
			if $(call relative-git,rev-parse --git-dir) > /dev/null 2>&1 ; then
				$(call relative-git,describe --abbrev=8 --dirty --always --tags); \
			else
				echo "unknown"; \
			fi
		else \
			echo "unknown"; \
		fi)
endef

## Pretty printing
## -----------------------------------------------------------------------------
## {{{

COMPILE_COLOR := \033[0;34m
LINK_COLOR    := \033[0;33m
HEADER_COLOR  := \033[0;32m
NO_COLOR      := \033[m

header-print = $(HEADER_COLOR)$(1)$(NO_COLOR)

# Clear last line if the tput command exists
define clear-last-line
	@$(call if-command-exists,tput,tput cuu 1 && tput el)
endef

# Pretty print a linking command and run it
define link-and-print
	@echo "$(LINK_COLOR)(LINK.c)$(NO_COLOR) $2"
	@$1
endef

# Pretty print a compilation command and run it
define compile-and-print
	@echo "$(COMPILE_COLOR)(COMPILE.c)$(NO_COLOR) $2"
	@$1
	@$(clear-last-line)
endef

## }}}

# Sanitise a string for a C++ file
# At the end of the makefile because of editor syntax highlighting
define sanitize-string
	$(strip $(subst ",\\\",$1))
endef

# }}}
