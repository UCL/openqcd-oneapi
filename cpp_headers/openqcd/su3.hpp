
/*******************************************************************************
 *
 * File su3.hpp
 *
 * Author (2018): Jonas Rylund Glesaaen
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef CPP_SU3_HPP
#define CPP_SU3_HPP

extern "C" {
#include "c_headers/su3.h"
}

namespace openqcd {

using complex = openqcd__complex;
using su3_vector = openqcd__su3_vector;
using su3 = openqcd__su3;
using su3_alg = openqcd__su3_alg;
using weyl = openqcd__weyl;
using spinor = openqcd__spinor;
using pauli = openqcd__pauli;
using u3_alg = openqcd__u3_alg;
using complex_dble = openqcd__complex_dble;
using su3_vector_dble = openqcd__su3_vector_dble;
using su3_dble = openqcd__su3_dble;
using su3_alg_dble = openqcd__su3_alg_dble;
using weyl_dble = openqcd__weyl_dble;
using spinor_dble = openqcd__spinor_dble;
using pauli_dble = openqcd__pauli_dble;
using u3_alg_dble = openqcd__u3_alg_dble;

} // namespace openqcd

#endif // CPP_SU3_HPP
