# openQCD-FASTSUM build system

The openQCD-FASTSUM build system is an improvement of the original openQCD build
system, and allows for compiling the code in a different folder to where the
source directory is located. This makes it easier to compile multiple versions
of the code, something that is often necessary due to the fact that the lattice
dimension is hard coded at compile time.

During compilation a source file containing the current date and the compile
flags used will be generated and this information will be prepended to the log
file of every run. If the code location also is a git repository, the current
SHA will also be added to the log (along with a release tag and a dirty
specifier if the source files have been modified).

## Targets

The makefile is written so that it is easy to compile exactly the component one
needs. There are two targets listing all targets:

 * `help`: prints a formatted help message with general information
 * `list-targets`: automatically generated list of non-rule targets

The main executables can either be specified individually (`qcd1`, `ms1`, `ym1`,
...), or compile all executables with the `all` target. `all` is also the
default target if no other target is specified.

The tests can be compiled with the `tests` target. One can also compile the
tests individually, for example

```
make devel/stout_smearing
```

will build all stout smearing tests, while for example

```
make devel/stout_smearing/test01_omega_matrices
```

will build a specific test in that module.

It is also possible to build an archive of a single module as every module is a
valid target. However this is not all that useful due to the way the modules are
structured.


## Configuration

The compilation is configured by editing the `compile_settings.txt` file. An
example file could read:

```
CODELOC ..
COMPILER /usr/bin/mpicc
MPI_INCLUDE /usr/include/mpi

CFLAGS -std=c89 -O2 -DAVX -DFMA3 -Werror -Wall
LDFLAGS

NPROC0_TOT   1
NPROC1_TOT   1
NPROC2_TOT   1 
NPROC3_TOT   1 

L0 8
L1 8
L2 8       
L3 8       

NPROC0_BLK 1
NPROC1_BLK 1
NPROC2_BLK 1
NPROC3_BLK 1
```

The options are as follows:

 * `CODELOC`: The location of the openqcd-fastsum root directory, accepts both
              relative and absolute paths

 * `COMPILER`: Specify which compiler to use

 * `MPI_INCLUDE`: Location of the `mpi.h` header file

 * `CFLAGS`: Compiler flags passed to the compilation stages, see the next
             section for a list of available flags for the openQCD software

 * `LDFLAGS`: Compiler flags passed to the linker

 * `NPROCx_TOT`: Number of nodes in the x'th direction. Corresponds to the
                 `NPROCx` macros in the source code.

 * `Lx`: Local number of lattice sites in the x'th direction. Corresponds to the
         `Lx` macros in the source code.

 * `NPROCx_BLK`: Processor block size. Corresponds to the `NPROCx_BLK` macros in
                 the source code.


## Compile flags

The code can be further specialised through a set of compiler flags specified
through the `CFLAGS` option in the `compile_settings.txt` file. There are two
different types of these flags: intrinsics, specifying which intrinsic
vectorisation instructions to use, and debugging, specifying the debugging
output.

### Intrinsics

Current Intel and AMD processors are able to perform arithmetic operations on
short vectors of floating-point numbers in just one or two machine cycles, using
SSE and/or AVX instructions.

Many programs in the module directories include inline-assembly SSE and AVX
code. Inline assembly is a GCC extension of the C language that may not be
supported by other compilers. On 64bit systems the code can be activated by
setting the compiler flags `-Dx64` or `-DAVX`, respectively. Furthermore, one
can enable AVX-512 instructions with the flag `-DAVX512`. These are only
available on modern architectures (currently Xeon Phi x200 and Skylake-X), and
are also only supported by modern Intel and GNU compilers. The BlueGeneQ
optimizations are enabled by using the flag `-DQPX`, and are only supported by
the IBM xlc compiler. In addition, SSE prefetch instructions will be used if
one of the following options is specified:

 * `-DP4`: Assume that prefetch instructions fetch 128 bytes at a time
           (Pentium 4 and related Xeons).

 * `-DPM`: Assume that prefetch instructions fetch 64 bytes at a time
           (Athlon, Opteron, Pentium M, Core, Core 2 and related Xeons).

 * `-DP3`: Assume that prefetch instructions fetch 32 bytes at a time
           (Pentium III).

These options have an effect only if `-Dx64` or `-DAVX` is set. The option
`-DAVX` implies `-Dx64`. If none of these options is set, the programs do not
make use of any C language extensions and are fully portable.

The latest x86 processors furthermore support fused multiply-add (FMA3)
instructions. OpenQCD makes use of these if the option `-DFMA3` is set in
addition to `-DAVX` (setting `-DFMA3` alone has no effect).

On recent x86-64 machines the recommended compiler flags are thus

```
-std=c89 -O -mno-avx -DAVX -DFMA3 -DPM
```

For older machines that do not support the AVX instruction set, the recommended
flags are

```
-std=c89 -O -mno-avx -Dx64 -DPM
```

Aggressive optimization levels such as `-O2` and `-O3` tend to have little
effect on the execution speed of the programs, but the risk of generating wrong
code is higher.

AVX instructions and the option `-mno-avx` may not be known to old versions of
the GCC compiler, in which case one may be limited to SSE accelerations with
option string `-std=c89 -O -Dx64 -DPM` (or no acceleration at all).

If compilers other than GCC are used together with the option `-Dx64` or
`-DAVX`, it is strongly recommended to verify the correctness of the compilation
using the check programs in the devel directory.


### Debugging

For troubleshooting and parameter tuning, it may helpful to switch on some
debugging flags at compilation time. The simulation program then prints a
detailed report to the log file on the progress made in specified subprogram.

The available flags are:

 * `-DCGNE_DBG`: CGNE solver.

 * `-DFGCR_DBG`: GCR solver.

 * `-DFGCR4VD_DBG`: GCR solver for the little Dirac equation.

 * `-DMSCG_DBG`: MSCG solver.

 * `-DDFL_MODES_DBG`: Deflation subspace generation.

 * `-DMDINT_DBG`: Integration of the molecular-dynamics equations.

 * `-DRWRAT_DBG`: Computation of the rational function reweighting factor.
