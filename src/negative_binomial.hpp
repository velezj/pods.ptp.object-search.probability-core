
#if !defined( __PROBABILITY_CORE_NEGATIVE_BINOMIAL_HPP__ )
#define __PROBABILITY_CORE_NEGATIVE_BINOMIAL_HPP__

#include "types.hpp"

namespace probability_core {
  
  // Description:
  // Negative Binomial Distribution
  
  double
  pdf( const unsigned int& x,
       const negative_binomial_distribution_t& nb );

  unsigned int
  sample_from( const negative_binomial_distribution_t& nb );

  double
  mean( const negative_binomial_distribution_t& d );
  
  double 
  variance( const negative_binomial_distribution_t& d );

  
  // Description:
  // Return the MLE esptimate for the negative binomail distribution
  // parmaeters given the set of observations k_i,
  // where k_i is the number of successes before a certain number of 
  // failures occur (specified by the r in the distribution)
  bool
  mle( const std::vector<size_t>& k, 
       negative_binomial_distribution_t& nb );

}



#endif

