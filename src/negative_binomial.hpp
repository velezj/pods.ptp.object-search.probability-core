
#if !defined( __PROBABILITY_CORE_NEGATIVE_BINOMIAL_HPP__ )
#define __PROBABILITY_CORE_NEGATIVE_BINOMIAL_HPP__

#include "lcmtypes/probability_core.hpp"

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

}



#endif

