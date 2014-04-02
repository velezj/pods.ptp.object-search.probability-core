
#if !defined( __DISTRIBUTIONS_HPP__ )
#define __DISTRIBUTIONS_HPP__

#include "gamma.hpp"
#include "beta.hpp"
#include "poisson.hpp"
#include "negative_binomial.hpp"
#include "gaussian.hpp"
#include "discrete.hpp"
#include "gamma_conjugate_prior.hpp"
#include "uniform.hpp"
//#include "dirichlet.hpp"

//========================================================================

// Description:
// The basic API for distributions.
// There are two basic functions:
//   pdf( x, distribution )
//   sample_from( distribution )
//
// There is also a global random number generator
//   global_rng()
//
// We also supply math_function_t interfaces for all the pdfs
// by calling functions::X_pdf( .. )
//    e.g. functions::poisson_pdf( poisson_distribution_t x )

//=========================================================================


#endif

