# openQCD Simulation Program

## LATTICE THEORY

Currently the common features of the supported lattice theories are the
following:

 * 4-dimensional hypercubic N0xN1xN2xN3 lattice with even sizes N0,N1,N2,N3.
   Open, Schrödinger functional (SF), open-SF or periodic boundary conditions
   in the time direction and periodic boundary conditions in the space
   directions.
 
 * SU(3) gauge group, plaquette plus planar double-plaquette gauge action
   (Wilson, Symanzik, Iwasaki,...).
 
 * O(a)-improved Wilson quarks in the fundamental representation of the gauge
   group. Among the supported quark multiplets are the classical ones (pure
   gauge, two-flavour theory, 2+1 and 2+1+1 flavour QCD), but doublets with a
   twisted mass and theories with many doublets, for example, are also
   supported.

 * Anisotropic lattice actions (currently only for periodic boundary
   conditions).

 * Stout smearing for the gauge links (currently only for periodic boundary
   conditions).

The O(a)-improvement includes the boundary counterterms required for the
improvement of the correlation functions near the boundaries of the lattice in
the time direction if open, SF or open-SF boundary conditions are chosen. For
the quark fields phase-periodic boundary conditions in the space directions
are implemented too.


## SIMULATION ALGORITHM

The simulation program is based on the HMC algorithm. For the heavier quarks,
a version of the RHMC algorithm is used. Several advanced techniques are
implemented that can be configured at run time:

 * Nested hierarchical integrators for the molecular-dynamics equations, based
   on any combination of the leapfrog, 2nd order Omelyan-Mryglod-Folk (OMF) and
   4th order OMF elementary integrators, are supported.
 
 * Twisted-mass Hasenbusch frequency splitting, with any number of factors
   and twisted masses. Optionally with even-odd preconditioning.
 
 * Twisted-mass determinant reweighting.
 
 * Deflation acceleration and chronological solver along the molecular-dynamics
   trajectories.
 
 * A choice of solvers (CGNE, MSCG, SAP+GCR, deflated SAP+GCR) for the Dirac
   equation, separately configurable for each force component and
   pseudo-fermion action.

All of these depend on a number of parameters, whose values are passed to the
simulation program together with those of the action parameters (coupling
constants, quark masses, etc.) through a structured input parameter file.


## PROGRAM FEATURES

All programs parallelize in 0,1,2,3 or 4 dimensions, depending on what is
specified at compilation time. They are highly optimized for machines with
current Intel or AMD processors, but will run correctly on any system that
complies with the ISO C89 (formerly ANSI C) and the MPI 1.2 standards.

For the purpose of testing and code development, the programs can also
be run on a desktop or laptop computer. All what is needed for this is
a compliant C compiler and a local MPI installation such as Open MPI.


## DOCUMENTATION

The simulation program has a modular form, with strict prototyping and a
minimal use of external variables. Each program file contains a small number
of externally accessible functions whose functionality is described at the top
of the file.

The data layout is explained in various README files and detailed instructions
are given on how to run the main programs. A set of further documentation
files are included in the doc directory, where the normalization conventions,
the chosen algorithms and other important program elements are described.


## COMPILATION

See the build/README.md file.


## RUNNING A SIMULATION

The simulation programs reside in the directory "main". For each program,
there is a README file in this directory which describes the program
functionality and its parameters.

Running a simulation for the first time requires its parameters to be chosen,
which tends to be a non-trivial task. The syntax of the input parameter files
and the meaning of the various parameters is described in some detail in
main/README.infiles and doc/parms.pdf. Examples of valid parameter files are
contained in the directory main/examples.


## EXPORTED FIELD FORMAT

The field configurations generated in the course of a simulation are written
to disk in a machine-independent format (see modules/misc/archive.c).
Independently of the machine endianness, the fields are written in little
endian format. A byte-reordering is therefore not required when machines with
different endianness are used for the simulation and the physics analysis.


## AUTHORS

The initial release of the openQCD package was written by Martin Lüscher and
Stefan Schaefer. Support for Schrödinger functional boundary conditions was
added by John Bulava. Phase-periodic boundary conditions for the quark fields
were introduced by Isabel Campos. Several modules were taken over from the
DD-HMC program tree, which includes contributions from Luigi Del Debbio,
Leonardo Giusti, Björn Leder and Filippo Palombi.

Anisotropic actions were developed by Jonas Rylund Glesaaen and Benjamin Jäger
while stout smearing was implemented by Jonas Rylund Glesaaen.

See the [CITATION.cff](CITATION.cff) file for a full list of contributions.


## ACKNOWLEDGEMENTS

In the course of the development of the openQCD code, many people suggested
corrections and improvements or tested preliminary versions of the programs.
The authors are particularly grateful to Isabel Campos, Dalibor Djukanovic,
Georg Engel, Leonardo Giusti, Björn Leder, Carlos Pena and Hubert Simma for
their communications and help.


## LICENSE

The software may be used under the terms of the GNU General Public Licence
(GPL).


## BUG REPORTS

If a bug is discovered, please file an issue on the [GitLab issue
tracker](https://gitlab.com/fastsum/openqcd-fastsum/issues).


## ALTERNATIVE PACKAGES AND COMPLEMENTARY PROGRAMS

There is a publicly available BG/Q version of openQCD that takes advantage of
the machine-specific features of IBM BlueGene/Q computers. The version is
available at <http://hpc.desy.de/simlab/codes/>.

The openQCD programs currently do not support reweighting in the quark
masses, but a module providing this functionality can be downloaded from
<http://www-ai.math.uni-wuppertal.de/~leder/mrw/>.

Full-fledged QCD simulation programs tend to have many adjustable parameters.
In the case of openQCD, most parameters are passed to the programs through a
human-readable structured file. Liam Keegan's sleek graphical editor for these
parameter files offers some guidance and complains when inconsistent parameter
values are entered (see <http://lkeegan.github.io/openQCD-input-file-editor>).


## ADD-ONS

The openQCD-FASTSUM code includes two optimisation additions.

### BLUEGENEQ OPTIMISATION

An optimized version of openQCD-1.2 for BlueGene/Q and other machines

#### DESCRIPTION

This software is a modification of the original openQCD code from
http://luscher.web.cern.ch/luscher/openQCD/openQCD-1.2.tar.gz which was written
by Martin Luscher and Stefan Schaefer 

The additional optimizations include:

1.  Use of intrinsics for QPX instructions for BlueGene/Q

    These instructions are only available on BlueGene/Q and with the IBM xlc
    compiler. They are enabled by default (by including `-DQPX` in CFLAGS in
    main/Makefile).

2.  implementation of global communications by different MPI functions

    Wrapper functions for global summation and broadcast allow to implement them
    by different MPI cunctions (e.g. using Reduce + Bcast or Allreduce for global
    sums). 

    This can be used also on other machines, e.g. SuperMUC, and may help to
    significantly reduce the time spent in global communications or to work
    around unpleasant features of some MPI implementations.

    The MPI functions used are selected in modules/utils/utils.c through the CPP
    macros `USE_MPI_BCAST` and `USE_MPI_ALLREDUCE`. The default setting has been
    found benificial on BlueGene/Q and SuperMUC.

#### AUTHORS

The original openQCD code has been written by Martin Luscher and Stefan Schaefer
with contributions from others (see http://luscher.web.cern.ch/luscher/openQCD/)

The optimizations and modifications with respect to the original openQCD code
have been implemented and tested by Dalibor Djukanovic, Mauro Papinutto, and
Hubert Simma.

#### LICENSE

The code may be used under the terms of the GNU General Public License (GPL)
http://www.fsf.org/licensing/licenses/gpl.html


### AVX512 OPTIMISATION

An optimized version of openQCD-1.2 for Intel processors with 512 bit vector
width

#### DESCRIPTION

This patch implements the Dirac operators with intel Intrincic operations in
order to use the full vector width on Intel Skylake, Knight Landing and other
processors. 

To enable the optimized version add -DAVX512 to the compiler flags.

#### AUTHORS

This patch extends the openQCD code written by Martin Luscher, Stefan Schaefer
and other contributors (see http://luscher.web.cern.ch/luscher/openQCD/).

The batch has been written by and tested by Jarno Rantaharju and Michele
Messiti.

#### LICENSE

The code may be used under the terms of the GNU General Public License (GPL)
http://www.fsf.org/licensing/licenses/gpl.html
