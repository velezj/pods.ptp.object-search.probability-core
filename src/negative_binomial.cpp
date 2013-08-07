
#include "negative_binomial.hpp"
#include "core.hpp"

namespace probability_core {
  
  // Description:
  // Negative Binomial Distribution
  
  double
  pdf( const unsigned int& x,
       const negative_binomial_distribution_t& nb ) 
  {
    
    return gsl_ran_negative_binomial_pdf( x, nb.p, nb.r );
  }

  unsigned int
  sample_from( const negative_binomial_distribution_t& nb )
  {
    return gsl_ran_negative_binomial( global_rng(), nb.p, nb.r );
  }

  double
  mean( const negative_binomial_distribution_t& nb )
  {
    return nb.p * nb.r / ( 1 - nb.p );
  }

  double
  variance( const negative_binomial_distribution_t& nb )
  {
    return nb.p * nb.r / ( (1- nb.p) * (1 - nb.p) );
  }
}

